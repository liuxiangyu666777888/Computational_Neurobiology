#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gmemory/lxy/计算神经学final_task}"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-data/cn_project_t1_noise2}"

if [[ ! -d "$DATA_ROOT" ]]; then
  mkdir -p data
  cat data_parts/cn_project.tar.part_* | tar -xf - -C data
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/prepare_data.py --data-root "$DATA_ROOT" --out-dir splits

.venv/bin/python train.py \
  --data-root "$DATA_ROOT" \
  --splits-dir splits \
  --epochs 40 \
  --batch-size 24 \
  --lr 1e-4 \
  --device cuda \
  --num-workers 2 \
  --grad-clip 1.0 \
  --out-dir runs/denoise_unet

.venv/bin/python evaluate.py \
  --data-root "$DATA_ROOT" \
  --split splits/test.csv \
  --ckpt runs/denoise_unet/best.pt \
  --save-figures 3 \
  --out-dir runs/denoise_unet/eval

.venv/bin/python scripts/plot_training_curve.py \
  --metrics runs/denoise_unet/metrics.csv \
  --output runs/denoise_unet/eval/training_curve.png

.venv/bin/python scripts/fill_report.py \
  --metrics runs/denoise_unet/eval/metrics_summary.csv \
  --case-metrics runs/denoise_unet/eval/case_metrics.csv \
  --splits-summary splits/summary.csv \
  --training-metrics runs/denoise_unet/metrics.csv \
  --figures-dir runs/denoise_unet/eval/figures \
  --training-curve runs/denoise_unet/eval/training_curve.png \
  --run-dir runs/denoise_unet \
  --output report_final.md

if command -v pandoc >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then
  pandoc report_final.md -o report_final.pdf --pdf-engine=xelatex --resource-path=. -V CJKmainfont="${CJK_FONT:-Noto Sans CJK SC}"
else
  echo "pandoc/xelatex not found; report_final.md was generated, PDF export skipped."
fi
