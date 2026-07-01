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

由于本地到服务器的全量数据上传速度较慢，本次已完成的可复现实验先使用从原始压缩包中抽取的 8 个完整病例作为 mini 子集。实验按病例级别划分训练集、验证集和测试集，实际划分为训练集 6 例、验证集 1 例、测试集 1 例，随机种子为 42。病例级划分可以避免同一受试者的不同切片同时出现在训练集和测试集中导致数据泄漏。

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
- 训练轮数：10 epochs。
- batch size：8。
- 每个病例最多抽取 16 张有效轴向切片用于本次 mini 实验。
- 设备：NVIDIA RTX A6000。

## 5. 评价指标

测试集上同时评估 noisy baseline 和模型输出，使用以下指标：

- MAE：平均绝对误差，越低越好。
- MSE：均方误差，越低越好。
- PSNR：峰值信噪比，越高越好。
- SSIM：结构相似性，越高越好。

## 6. 实验结果

训练完成后，在测试集上得到如下结果。完整指标文件位于 `runs/denoise_unet_mini_zero/eval/metrics_summary.csv`。

| 方法 | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Noisy baseline | 0.0031 | 0.0000 | 47.4241 | 0.9787 |
| Residual U-Net | 0.0023 | 0.0000 | 49.6866 | 0.9943 |

可以看到，Residual U-Net 的 MAE 和 MSE 低于 noisy baseline，同时 PSNR 和 SSIM 高于 noisy baseline。这说明在当前 mini 子集实验中，模型能够在保持结构相似性的同时进一步减小加噪图像与干净图像之间的误差。

## 7. 可视化结果

测试集可视化保存在：

```text
runs/denoise_unet_mini_zero/eval/figures/
```

每张图包含四列：Noisy、Denoised、Clean、Abs Error。本次 mini 实验测试集为 1 个病例，因此展示该测试病例的可视化结果。

![1000509](runs/denoise_unet_mini_zero/eval/figures/1000509.png)

## 8. 结论

本实验实现了基于 2D Residual U-Net 的 T1 MRI 去噪方法。实验采用病例级数据划分、监督学习训练和多指标测试评估。在已完成的 mini 子集实验中，模型相较 noisy baseline 降低了 MAE/MSE，并提高了 PSNR/SSIM，说明该方法能够有效抑制噪声并保留主要解剖结构。

本方法实现简单、训练稳定，适合作为本课程期末作业中的医学影像去噪实验基线。全量数据上传完成后，可直接使用项目中的 `scripts/run_server.sh` 在相同代码和参数框架下复现实验并生成全量报告。
