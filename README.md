# T1 MRI Denoising Experiment

This repository contains the course final-project experiment for supervised T1 MRI denoising:

```text
T1_noisy.nii.gz -> T1_clean.nii.gz
```

The current version compares three residual U-Net variants on the same 600-case dataset:

- `train.py` / `evaluate.py`: 2D Residual U-Net
- `train25d.py` / `evaluate25d.py`: 2.5D Residual U-Net with adjacent axial slices as channels
- `train3d.py` / `evaluate3d.py`: patch-based 3D Residual U-Net

The NeurIPS-style English report is in `report_nips/main.tex`, with the compiled PDF at `report_nips/main.pdf`.

## Server Setup

Experiments were run on:

```text
/home/gmemory/lxy/计算神经学final_task
```

Create the environment:

```bash
cd /home/gmemory/lxy/计算神经学final_task
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
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

Prepare case-level splits:

```bash
.venv/bin/python scripts/prepare_data.py --data-root data/cn_project_t1_noise2 --out-dir splits
```

## Training and Evaluation

2D:

```bash
.venv/bin/python train.py --data-root data/cn_project_t1_noise2 --splits-dir splits --epochs 40 --batch-size 24 --lr 1e-4 --device cuda --out-dir runs/denoise_unet
.venv/bin/python evaluate.py --data-root data/cn_project_t1_noise2 --split splits/test.csv --ckpt runs/denoise_unet/best.pt --out-dir runs/denoise_unet/eval
```

2.5D:

```bash
.venv/bin/python train25d.py --data-root data/cn_project_t1_noise2 --splits-dir splits --epochs 40 --batch-size 24 --lr 1e-4 --device cuda --patience 6 --out-dir runs/denoise_unet25d
.venv/bin/python evaluate25d.py --data-root data/cn_project_t1_noise2 --split splits/test.csv --ckpt runs/denoise_unet25d/best.pt --out-dir runs/denoise_unet25d/eval
```

3D:

```bash
.venv/bin/python train3d.py --data-root data/cn_project_t1_noise2 --splits-dir splits --epochs 8 --batch-size 2 --patch-size 32 96 96 --device cuda --out-dir runs/denoise_unet3d
.venv/bin/python evaluate3d.py --data-root data/cn_project_t1_noise2 --split splits/test.csv --ckpt runs/denoise_unet3d/best.pt --max-slices-per-case 48 --out-dir runs/denoise_unet3d/eval
```

## Results

Aggregate result CSVs used by the report are copied into `report_nips/results/`:

```text
report_nips/results/metrics_2d.csv
report_nips/results/metrics_25d.csv
report_nips/results/metrics_3d.csv
```

The best model in the completed comparison is the 2.5D Residual U-Net:

```text
MAE 0.002758
MSE 1.747987e-05
PSNR 47.849837 dB
SSIM 0.993208
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Report Build

```bash
cd report_nips
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
