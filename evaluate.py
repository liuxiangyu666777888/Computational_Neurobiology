#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from tqdm import tqdm

from src.dataset import load_pair, read_cases, valid_slice_indices
from src.model import ResidualUNet2D


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate T1 MRI denoising on held-out cases.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/denoise_unet/eval"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-slices-per-case", type=int, default=None)
    parser.add_argument("--save-figures", type=int, default=3)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mae = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr(mse),
        "ssim": float(structural_similarity(target, pred, data_range=1.0)),
    }


def resize_slice(arr: np.ndarray, target_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)


def predict_case(
    model: torch.nn.Module,
    noisy: np.ndarray,
    clean: np.ndarray,
    indices: List[int],
    target_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    noisy_slices: List[np.ndarray] = []
    clean_slices: List[np.ndarray] = []
    pred_slices: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            noisy_t = resize_slice(noisy[:, :, idx], target_size).to(device)
            clean_t = resize_slice(clean[:, :, idx], target_size)
            pred_t = model(noisy_t).cpu()
            noisy_slices.append(noisy_t.cpu().numpy()[0, 0])
            clean_slices.append(clean_t.numpy()[0, 0])
            pred_slices.append(pred_t.numpy()[0, 0])

    return np.stack(noisy_slices), np.stack(pred_slices), np.stack(clean_slices)


def save_figure(caseid: str, noisy: np.ndarray, pred: np.ndarray, clean: np.ndarray, out_path: Path) -> None:
    mid = noisy.shape[0] // 2
    error = np.abs(pred[mid] - clean[mid])
    panels = [
        ("Noisy", noisy[mid], "gray"),
        ("Denoised", pred[mid], "gray"),
        ("Clean", clean[mid], "gray"),
        ("Abs Error", error, "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Case {caseid}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", args.base_channels))
    target_size = int(ckpt_args.get("target_size", args.target_size))
    model = ResidualUNet2D(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"])

    records = read_cases(args.split, args.data_root, max_cases=args.max_cases)
    rows = []

    for fig_count, record in enumerate(tqdm(records), start=0):
        noisy, clean = load_pair(str(record.noisy_path), str(record.clean_path))
        indices = valid_slice_indices(clean, min_foreground_ratio=0.01)
        if args.max_slices_per_case is not None and len(indices) > args.max_slices_per_case:
            positions = np.linspace(0, len(indices) - 1, args.max_slices_per_case).round().astype(int)
            indices = [indices[int(pos)] for pos in positions]

        noisy_eval, pred_eval, clean_eval = predict_case(model, noisy, clean, indices, target_size, device)
        noisy_metrics = metrics(noisy_eval, clean_eval)
        model_metrics = metrics(pred_eval, clean_eval)

        row = {"caseid": str(record.caseid), "slices": int(len(indices))}
        row.update({f"noisy_{key}": value for key, value in noisy_metrics.items()})
        row.update({f"model_{key}": value for key, value in model_metrics.items()})
        rows.append(row)

        if fig_count < args.save_figures:
            save_figure(record.caseid, noisy_eval, pred_eval, clean_eval, figures_dir / f"{record.caseid}.png")

    case_df = pd.DataFrame(rows)
    if not case_df.empty:
        case_df["caseid"] = case_df["caseid"].astype(str)
    case_df.to_csv(args.out_dir / "case_metrics.csv", index=False)

    summary_rows = []
    for prefix in ["noisy", "model"]:
        summary = {"method": prefix}
        for metric_name in ["mae", "mse", "psnr", "ssim"]:
            summary[metric_name] = float(case_df[f"{prefix}_{metric_name}"].mean())
            summary[f"{metric_name}_std"] = float(case_df[f"{prefix}_{metric_name}"].std(ddof=0))
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "metrics_summary.csv", index=False)
    LOGGER.info("Evaluation summary:\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
