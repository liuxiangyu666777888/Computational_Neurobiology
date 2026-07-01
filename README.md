# T1 MRI Denoising Experiment

This project implements the selected final-task option: supervised T1 MRI denoising from `T1_noisy.nii.gz` to `T1_clean.nii.gz`.

## Authoritative Deliverables

The full-data experiment has been completed on all 600 cases.

Use these files for submission:

```text
report_final.md
report_final.pdf
runs/denoise_unet/eval/metrics_summary.csv
runs/denoise_unet/eval/case_metrics.csv
runs/denoise_unet/eval/figures/
```

Historical reports are kept only for traceability:

```text
report.md                 initial template
report_final_24case.md    earlier 24-case subset run
report_final_mini.md      earlier mini smoke run
```

The reproducible full-run configuration is recorded in `configs/denoise_full.json`.

## Server Setup

```bash
cd /home/gmemory/lxy/计算神经学final_task
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

If exact environment reproduction is required, install from `requirements-lock.txt` after it has been generated on the server with:

```bash
.venv/bin/pip freeze > requirements-lock.txt
```

## Data

Upload the local split tar files to the server:

```bash
rsync -avP data_cn/cn_project.tar.part_* gmemory@10.176.40.144:/home/gmemory/lxy/计算神经学final_task/data_parts/
```

Extract on the server:

```bash
mkdir -p data
cat data_parts/cn_project.tar.part_* | tar -xf - -C data
```

The expected data root is:

```text
data/cn_project_t1_noise2
```

## Run

```bash
.venv/bin/python scripts/prepare_data.py --data-root data/cn_project_t1_noise2 --out-dir splits
.venv/bin/python train.py --data-root data/cn_project_t1_noise2 --splits-dir splits --epochs 40 --batch-size 24 --lr 1e-4 --device cuda --grad-clip 1.0 --out-dir runs/denoise_unet
.venv/bin/python evaluate.py --data-root data/cn_project_t1_noise2 --split splits/test.csv --ckpt runs/denoise_unet/best.pt --out-dir runs/denoise_unet/eval
```

Optional training features:

```bash
--amp             enable CUDA mixed precision
--augment         enable paired random horizontal/vertical flips
--lr-scheduler    enable ReduceLROnPlateau
--patience 10     stop after 10 epochs without validation improvement
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Smoke Test

```bash
.venv/bin/python train.py --data-root data/cn_project_t1_noise2 --splits-dir splits --epochs 1 --batch-size 2 --max-cases 8 --max-slices-per-case 8 --device cuda --out-dir runs/smoke
.venv/bin/python evaluate.py --data-root data/cn_project_t1_noise2 --split splits/test.csv --ckpt runs/smoke/best.pt --max-cases 3 --max-slices-per-case 8 --out-dir runs/smoke/eval
```

## Completed 24-Case Run

A 24-case subset experiment was completed on the server after transferring `subsets/cn_denoise_subset.tar`.

Outputs:

```text
report_final_24case.md
runs/denoise_unet_24case/metrics.csv
runs/denoise_unet_24case/eval/metrics_summary.csv
runs/denoise_unet_24case/eval/figures/
splits_24/
```

Split:

```text
train 17, val 4, test 3
```

Test metrics:

```text
Noisy baseline: MAE 0.0036, MSE 0.000025, PSNR 46.8310, SSIM 0.9801
Residual U-Net: MAE 0.0027, MSE 0.000015, PSNR 48.4807, SSIM 0.9943
```

Reproduce the completed 24-case run:

```bash
cd /home/gmemory/lxy/计算神经学final_task
rm -rf data/cn_project_t1_noise2 splits_24 runs/denoise_unet_24case
mkdir -p data
tar -xf subsets/cn_denoise_subset.tar -C data
.venv/bin/python scripts/prepare_data.py --data-root data/cn_project_t1_noise2 --out-dir splits_24
.venv/bin/python train.py --data-root data/cn_project_t1_noise2 --splits-dir splits_24 --epochs 20 --batch-size 12 --max-slices-per-case 24 --device cuda --out-dir runs/denoise_unet_24case
.venv/bin/python evaluate.py --data-root data/cn_project_t1_noise2 --split splits_24/test.csv --ckpt runs/denoise_unet_24case/best.pt --max-slices-per-case 24 --save-figures 3 --out-dir runs/denoise_unet_24case/eval
.venv/bin/python scripts/plot_training_curve.py --metrics runs/denoise_unet_24case/metrics.csv --output runs/denoise_unet_24case/eval/training_curve.png
```

The final report is `report_final.md`. If PDF export is required, open `report_final.md` in a Markdown editor that supports local images and export to PDF, or use:

```bash
pandoc report_final.md -o report_final.pdf --pdf-engine=xelatex --resource-path=.
```

## Completed Mini Run

Because the full 31GB data upload was slow over the current connection, a reproducible 8-case mini experiment was completed on the server.

Outputs:

```text
report_final_mini.md
runs/denoise_unet_mini_zero/metrics.csv
runs/denoise_unet_mini_zero/eval/metrics_summary.csv
runs/denoise_unet_mini_zero/eval/figures/1000509.png
```

Mini-test metrics:

```text
Noisy baseline: MAE 0.0031, MSE 0.000018, PSNR 47.4241, SSIM 0.9787
Residual U-Net: MAE 0.0023, MSE 0.000011, PSNR 49.6866, SSIM 0.9943
```
