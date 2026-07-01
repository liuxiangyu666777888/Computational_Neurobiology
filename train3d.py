#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_3d_dataset
from src.model import ResidualUNet3D


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D Residual U-Net for T1 MRI denoising.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/denoise_unet3d"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 96, 96), metavar=("D", "H", "W"))
    parser.add_argument("--patches-per-case", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return l1 + 0.2 * mse, l1, mse


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_items = 0

    with torch.set_grad_enabled(is_train):
        for noisy, clean in tqdm(loader, leave=False):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(noisy)
                loss, l1, mse = loss_fn(pred, clean)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = noisy.shape[0]
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_l1 += float(l1.item()) * batch_size
            total_mse += float(mse.item()) * batch_size

    return {"loss": total_loss / total_items, "l1": total_l1 / total_items, "mse": total_mse / total_items}


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_ds = build_3d_dataset(
        args.splits_dir / "train.csv",
        args.data_root,
        patch_size=tuple(args.patch_size),
        patches_per_case=args.patches_per_case,
        max_cases=args.max_cases,
    )
    val_ds = build_3d_dataset(
        args.splits_dir / "val.csv",
        args.data_root,
        patch_size=tuple(args.patch_size),
        patches_per_case=max(1, min(args.patches_per_case, 4)),
        max_cases=args.max_cases,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = ResidualUNet3D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_path = args.out_dir / "metrics.csv"
    best_val = float("inf")
    stale_epochs = 0

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_l1", "train_mse", "val_loss", "val_l1", "val_mse", "lr", "best"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            LOGGER.info("Epoch %d/%d", epoch, args.epochs)
            train_metrics = run_epoch(model, train_loader, device, optimizer, scaler=scaler, amp_enabled=amp_enabled, grad_clip=args.grad_clip)
            val_metrics = run_epoch(model, val_loader, device, amp_enabled=amp_enabled)
            is_best = val_metrics["loss"] < best_val - args.min_delta
            if is_best:
                best_val = val_metrics["loss"]
                stale_epochs = 0
                save_checkpoint(args.out_dir / "best.pt", model, optimizer, epoch, best_val, args)
            else:
                stale_epochs += 1
            save_checkpoint(args.out_dir / "last.pt", model, optimizer, epoch, best_val, args)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_l1": train_metrics["l1"],
                "train_mse": train_metrics["mse"],
                "val_loss": val_metrics["loss"],
                "val_l1": val_metrics["l1"],
                "val_mse": val_metrics["mse"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best": int(is_best),
            }
            writer.writerow(row)
            f.flush()
            LOGGER.info("epoch=%d train_loss=%.6f val_loss=%.6f best=%s stale=%d", epoch, row["train_loss"], row["val_loss"], bool(is_best), stale_epochs)
            if args.patience > 0 and stale_epochs >= args.patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    LOGGER.info("Training complete. Best validation loss: %.6f", best_val)


if __name__ == "__main__":
    main()
