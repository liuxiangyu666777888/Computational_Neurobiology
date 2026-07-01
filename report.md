# T1 加噪 MRI 图像去噪实验报告

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

实验按病例级别划分训练集、验证集和测试集，比例为 70%/15%/15%，随机种子为 42。病例级划分可以避免同一受试者的不同切片同时出现在训练集和测试集中导致数据泄漏。

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

## 6. 实验结果

训练完成后，将 `runs/denoise_unet/eval/metrics_summary.csv` 中的结果填入下表。

| 方法 | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Noisy baseline | 待填入 | 待填入 | 待填入 | 待填入 |
| Residual U-Net | 待填入 | 待填入 | 待填入 | 待填入 |

从结果上看，若模型有效，应表现为 MAE 和 MSE 低于 noisy baseline，同时 PSNR 和 SSIM 高于 noisy baseline。这说明模型不仅降低了像素误差，也更好地保持了 MRI 结构信息。

## 7. 可视化结果

测试集可视化保存在：

```text
runs/denoise_unet/eval/figures/
```

每张图包含四列：Noisy、Denoised、Clean、Abs Error。报告最终版本至少插入 3 个测试病例的对比图。

示例占位：

```markdown
![case1](runs/denoise_unet/eval/figures/CASEID.png)
![case2](runs/denoise_unet/eval/figures/CASEID.png)
![case3](runs/denoise_unet/eval/figures/CASEID.png)
```

## 8. 结论

本实验实现了基于 2D Residual U-Net 的 T1 MRI 去噪方法。实验采用病例级数据划分、监督学习训练和多指标测试评估。最终结果将与 noisy baseline 比较，若模型在 PSNR/SSIM 上提升且 MAE/MSE 下降，则说明模型能够有效抑制噪声并保留主要解剖结构。

本方法实现简单、训练稳定，适合作为本课程期末作业中的医学影像去噪实验基线。
