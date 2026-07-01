#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/validation loss curves from metrics.csv.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.metrics.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {args.metrics}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    best_rows = [row for row in rows if row.get("best") == "1"]
    best_epoch = int(best_rows[-1]["epoch"]) if best_rows else epochs[val_loss.index(min(val_loss))]
    best_val = min(val_loss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, train_loss, marker="o", linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=1.8, label="Validation loss")
    ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.0, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curve (best val loss={best_val:.6f})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
