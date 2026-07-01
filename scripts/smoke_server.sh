#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gmemory/lxy/计算神经学final_task}"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-data/cn_project_t1_noise2}"

.venv/bin/python scripts/prepare_data.py --data-root "$DATA_ROOT" --out-dir splits

.venv/bin/python train.py \
  --data-root "$DATA_ROOT" \
  --splits-dir splits \
  --epochs 1 \
  --batch-size 2 \
  --max-cases 8 \
  --max-slices-per-case 8 \
  --device cuda \
  --patience 0 \
  --out-dir runs/smoke

.venv/bin/python evaluate.py \
  --data-root "$DATA_ROOT" \
  --split splits/test.csv \
  --ckpt runs/smoke/best.pt \
  --max-cases 3 \
  --max-slices-per-case 8 \
  --out-dir runs/smoke/eval
