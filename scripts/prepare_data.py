#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare case-level train/val/test splits for T1 denoising.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to cn_project_t1_noise2.")
    parser.add_argument("--out-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    rows = []
    for case_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        noisy = case_dir / "T1_noisy.nii.gz"
        clean = case_dir / "T1_clean.nii.gz"
        if noisy.exists() and clean.exists():
            rows.append(
                {
                    "caseid": case_dir.name,
                    "noisy_path": str(noisy.relative_to(data_root)),
                    "clean_path": str(clean.relative_to(data_root)),
                }
            )

    if len(rows) < 3:
        raise ValueError(f"Need at least 3 paired cases, found {len(rows)} under {data_root}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_total = len(rows)
    n_train = max(1, int(round(n_total * args.train_ratio)))
    n_val = max(1, int(round(n_total * args.val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)

    splits = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_caseids = set()
    for name, split_rows in splits.items():
        caseids = {row["caseid"] for row in split_rows}
        overlap = all_caseids.intersection(caseids)
        if overlap:
            raise RuntimeError(f"Case leakage detected in {name}: {sorted(overlap)[:5]}")
        all_caseids.update(caseids)
        pd.DataFrame(split_rows).sort_values("caseid").to_csv(args.out_dir / f"{name}.csv", index=False)

    summary = pd.DataFrame(
        [{"split": name, "cases": len(split_rows)} for name, split_rows in splits.items()]
    )
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Prepared {n_total} cases in {args.out_dir}")


if __name__ == "__main__":
    main()
