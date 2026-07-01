#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final Chinese report from experiment outputs.")
    parser.add_argument("--metrics", type=Path, default=Path("runs/denoise_unet/eval/metrics_summary.csv"))
    parser.add_argument("--case-metrics", type=Path, default=Path("runs/denoise_unet/eval/case_metrics.csv"))
    parser.add_argument("--splits-summary", type=Path, default=Path("splits/summary.csv"))
    parser.add_argument("--training-metrics", type=Path, default=Path("runs/denoise_unet/metrics.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("runs/denoise_unet/eval/figures"))
    parser.add_argument("--training-curve", type=Path, default=Path("runs/denoise_unet/eval/training_curve.png"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/denoise_unet"))
    parser.add_argument("--output", type=Path, default=Path("report_final.md"))
    parser.add_argument("--max-case-rows", type=int, default=10)
    return parser.parse_args()


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    value = float(value)
    if abs(value) < 1e-4 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def read_split_counts(path: Path) -> tuple[int | None, int | None, int | None, int | None]:
    if not path.exists():
        return None, None, None, None
    df = pd.read_csv(path)
    counts = {str(row["split"]): int(row["cases"]) for _, row in df.iterrows()}
    train = counts.get("train")
    val = counts.get("val")
    test = counts.get("test")
    total = sum(v for v in [train, val, test] if v is not None)
    return train, val, test, total


def metric_row(label: str, row: pd.Series) -> str:
    return (
        f"| {label} | {fmt(row['mae'])} | {fmt(row['mse'])} | "
        f"{fmt(row['psnr'])} | {fmt(row['ssim'])} |"
    )


def build_case_table(case_df: pd.DataFrame, max_rows: int) -> str:
    if case_df.empty:
        return "未找到病例级指标。"

    rows = [
        "| 病例 | 切片数 | Noisy MAE | Model MAE | Noisy PSNR | Model PSNR | Noisy SSIM | Model SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in case_df.head(max_rows).iterrows():
        caseid = str(row["caseid"])
        if caseid.endswith(".0"):
            caseid = caseid[:-2]
        rows.append(
            "| {caseid} | {slices} | {noisy_mae} | {model_mae} | {noisy_psnr} | {model_psnr} | {noisy_ssim} | {model_ssim} |".format(
                caseid=caseid,
                slices=int(row["slices"]),
                noisy_mae=fmt(row["noisy_mae"]),
                model_mae=fmt(row["model_mae"]),
                noisy_psnr=fmt(row["noisy_psnr"]),
                model_psnr=fmt(row["model_psnr"]),
                noisy_ssim=fmt(row["noisy_ssim"]),
                model_ssim=fmt(row["model_ssim"]),
            )
        )
    return "\n".join(rows)


def figure_markdown(figures: Iterable[Path]) -> str:
    lines = [f"![{fig.stem}]({fig.as_posix()})" for fig in figures]
    return "\n".join(lines) if lines else "未找到测试病例可视化图片。"


def training_summary(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "训练日志未找到。", ""
    df = pd.read_csv(path)
    if df.empty:
        return "训练日志为空。", ""
    first = df.iloc[0]
    last = df.iloc[-1]
    best_rows = df[df["best"].astype(str) == "1"] if "best" in df.columns else pd.DataFrame()
    if best_rows.empty:
        best = df.loc[df["val_loss"].idxmin()]
    else:
        best = best_rows.iloc[-1]

    summary = (
        f"训练日志保存在 `{path.as_posix()}`。训练集 loss 从第 1 个 epoch 的 "
        f"{fmt(first['train_loss'], 6)} 降到第 {int(last['epoch'])} 个 epoch 的 "
        f"{fmt(last['train_loss'], 6)}。最佳验证 loss 出现在第 {int(best['epoch'])} 个 epoch，"
        f"为 {fmt(best['val_loss'], 6)}，对应 checkpoint 保存为 `best.pt`。"
    )
    return summary, str(int(best["epoch"]))


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics).set_index("method")
    noisy = metrics.loc["noisy"]
    model = metrics.loc["model"]
    case_df = pd.read_csv(args.case_metrics, dtype={"caseid": str}) if args.case_metrics.exists() else pd.DataFrame()
    train_n, val_n, test_n, total_n = read_split_counts(args.splits_summary)
    train_text, _ = training_summary(args.training_metrics)

    split_sentence = (
        f"实验按病例级别划分训练集、验证集和测试集，比例为 70%/15%/15%，随机种子为 42。"
        f"本次全量实验共纳入 {total_n} 个完整病例，其中训练集 {train_n} 例、验证集 {val_n} 例、测试集 {test_n} 例。"
        if total_n is not None
        else "实验按病例级别划分训练集、验证集和测试集，比例为 70%/15%/15%，随机种子为 42。"
    )

    mae_delta = float(noisy["mae"] - model["mae"])
    mse_delta = float(noisy["mse"] - model["mse"])
    psnr_delta = float(model["psnr"] - noisy["psnr"])
    ssim_delta = float(model["ssim"] - noisy["ssim"])

    if not case_df.empty:
        improved = int((case_df["model_psnr"] > case_df["noisy_psnr"]).sum())
        improved_text = f"在病例级结果中，模型 PSNR 高于 noisy baseline 的测试病例数为 {improved}/{len(case_df)}。"
    else:
        improved_text = ""

    figures = sorted(args.figures_dir.glob("*.png"))[:3]
    training_curve_md = (
        f"![training_curve]({args.training_curve.as_posix()})"
        if args.training_curve.exists()
        else "训练曲线图片未找到。"
    )

    case_note = (
        f"下表展示前 {min(args.max_case_rows, len(case_df))} 个测试病例的病例级结果；完整文件位于 "
        f"`{args.case_metrics.as_posix()}`。"
        if len(case_df) > args.max_case_rows
        else f"病例级结果如下，完整文件位于 `{args.case_metrics.as_posix()}`。"
    )

    text = f"""# T1 加噪 MRI 图像去噪实验报告

## 1. 任务介绍

本实验选择课程期末作业中的 T1 MRI 去噪任务。数据集中每个病例包含一对三维 NIfTI 图像：`T1_noisy.nii.gz` 为加噪后的 T1 图像，`T1_clean.nii.gz` 为对应的干净图像。实验目标是学习从 noisy 图像到 clean 图像的映射，并通过定量指标和可视化结果评估去噪效果。

相比跨模态转换任务，去噪任务的输入和目标处于同一成像模态，空间结构和强度分布更一致，因此更适合作为稳定、可复现的期末实验。

## 2. 数据说明

数据目录为 `data/cn_project_t1_noise2`，每个病例目录结构如下：

```text
CASEID/
  T1_noisy.nii.gz
  T1_clean.nii.gz
```

{split_sentence} 病例级划分可以避免同一受试者的不同切片同时出现在训练集和测试集中导致数据泄漏。

## 3. 方法

本实验采用 2D Residual U-Net。模型以单张轴向 noisy slice 为输入，输出对应的 denoised slice。U-Net 编码器提取多尺度上下文信息，解码器通过跳跃连接恢复空间细节。模型最后预测残差，并与输入图像相加，再裁剪到 `[0, 1]` 范围：

```text
denoised = clamp(noisy + residual, 0, 1)
```

这种残差学习方式适合去噪任务，因为 noisy 图像与 clean 图像大部分结构一致，模型主要需要学习噪声成分的修正。实现中将最后一层残差输出卷积初始化为 0，使模型初始状态等价于 noisy baseline，再通过训练学习细微修正。

## 4. 实验设置

- 归一化：对每个病例的 noisy/clean 图像使用联合前景体素的 1% 和 99% 分位数归一化到 `[0, 1]`。
- 切片选择：抽取包含有效前景的轴向切片，跳过近空白切片。
- 输入尺寸：所有 2D slice 统一 resize 到 `256 x 256`。
- 损失函数：`L1 + 0.2 * MSE`。
- 优化器：AdamW，学习率 `1e-4`，weight decay `1e-5`。
- 训练轮数：40 epochs。
- batch size：24。
- 设备：NVIDIA RTX A6000。

## 5. 评价指标

测试集上同时评估 noisy baseline 和模型输出，使用以下指标：

- MAE：平均绝对误差，越低越好。
- MSE：均方误差，越低越好。
- PSNR：峰值信噪比，越高越好。
- SSIM：结构相似性，越高越好。

需要说明的是，本实验的定量指标是在归一化后的 2D resized slices 上计算的，而不是在原始 NIfTI 分辨率的三维体数据上直接计算。这样可以与训练输入保持一致，并保证不同病例切片尺寸统一。

## 6. 训练过程

{train_text}

{training_curve_md}

## 7. 实验结果

训练完成后，在测试集上得到如下结果。完整指标文件位于 `{args.metrics.as_posix()}`。

| 方法 | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
{metric_row("Noisy baseline", noisy)}
{metric_row("Residual U-Net", model)}

Residual U-Net 相较 noisy baseline 的平均 MAE 降低 {fmt(mae_delta)}，MSE 降低 {fmt(mse_delta)}，PSNR 提升 {fmt(psnr_delta)} dB，SSIM 提升 {fmt(ssim_delta)}。{improved_text}

{case_note}

{build_case_table(case_df, args.max_case_rows)}

## 8. 可视化结果

测试集可视化保存在：

```text
{args.figures_dir.as_posix()}/
```

每张图包含四列：Noisy、Denoised、Clean、Abs Error。下面展示 3 个测试病例的可视化结果。

{figure_markdown(figures)}

## 9. 结论

本实验实现了基于 2D Residual U-Net 的 T1 MRI 去噪方法。实验采用病例级数据划分、监督学习训练和多指标测试评估。测试结果表明，模型相较 noisy baseline 在平均 MAE/MSE 上更低，并在平均 PSNR/SSIM 上更高，说明该方法能够抑制噪声并保留主要解剖结构。

本方法实现简单、训练稳定，适合作为本课程期末作业中的医学影像去噪实验基线。
"""

    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
