# train-opt

这是 DINOv3 + EoMT 语义分割训练优化工程。原项目源码、训练脚本、推理项目、数据集和权重保持只读；优化通过 `train-opt/common` 的 harness、ops 注册表和各 round patch 注入。

## 已完成

- 数据转换写入 `/root/autodl-tmp/train-opt-tmp/data/sam3_head`，训练集 3672 张、验证集 477 张；抽样 mask 与 COCO 转换逐像素一致。
- 本地 backbone 权重逐 tensor 校验通过；固定 batch 为 ViT-L/16=`7`、ViT-B/16=`14`。
- 公共 harness、RngTape、双 GPU 锁、计时、nsys 分析、roofline、结果汇总和 round10 CLI 已实现。
- round0：两模型一轮原版训练、三阶段 GPU step/e2e、六份 nsys profile、训练形状 GEMM 峰值。
- round3：`batched_torch` 和 `fused` 在两个模型、三个阶段的单步 B 级通过。
- round4：`sdpa_rope` 在两个模型、三个阶段单步 B 级通过。
- round5：ViT-L 三阶段、ViT-B mid 的 `ln_fuse` 单步 B 级通过。
- round1：ViT-B/L mid 的 `nosync+adamw_fused` 单步 B 级通过。
- round2：ViT-B/L mid 的 segmented Graph 真实捕获及单步 B 级通过。
- round6/7/10：训练安全候选、capturable optimizer、padded loss 和 full Graph contract 已实现并通过 CPU 回归测试。
- 真正 CUDA C++ baseline 已加入：round3 `cuda_ops.cpp/.cu`（point loss + T≤4 matcher）、round4 `cuda_attention.cu`（packed QKV attention fwd/dQ/dK/dV）、round7 `cuda_adamw.cu`（device norm/clip/AdamW/step）。三者均通过 nvcc 13.0 / sm_120 编译，其中 round3/4/7 的小形状 A smoke 已通过。

## 运行与验证

```bash
PYTHONPATH=train-opt /root/miniconda3/bin/python -m pytest -q train-opt/round0_baseline/tests train-opt/round1_torch_cleanup/tests train-opt/round2_cuda_graph/tests train-opt/round3_loss_matcher/tests train-opt/round4_attention/tests train-opt/round5_block_fusion/tests train-opt/round6_head_fusion/tests train-opt/round7_optimizer/tests train-opt/round10_integration/tests
PYTHONPATH=train-opt /root/miniconda3/bin/python -m common.summarize
```

原版或 fast loop 入口：

```bash
PYTHONPATH=train-opt /root/miniconda3/bin/python train-opt/round10_integration/train.py --model dinov3/vitb16-eomt --epochs 1 --batch-size 14 --loop lightly
PYTHONPATH=train-opt /root/miniconda3/bin/python train-opt/round10_integration/train.py --model dinov3/vitb16-eomt --epochs 1 --batch-size 14 --loop fast --patches round1_torch_cleanup,round3_loss_matcher,round4_attention --patch-kwargs '{"round1_torch_cleanup":{"variant":"nosync"},"round3_loss_matcher":{"variant":"batched_torch"},"round4_attention":{"variant":"sdpa_rope"}}'
```

## 当前实测基线

| 模型 | batch | GPU step 加权吞吐 | e2e 吞吐 | 6 epoch 双 seed基线 |
|---|---:|---:|---:|---|
| ViT-B/16 | 14 | 37.80 img/s | 33.14 img/s | mIoU 0.83105±0.00061；head IoU 0.66262±0.00123 |
| ViT-L/16 | 7 | 16.25 img/s | 14.94 img/s | 待运行 |

完整表格和图在 [`results/summary.md`](results/summary.md) 与 [`results/throughput.png`](results/throughput.png)。原始 nsys CSV/摘要在 [`round0_baseline/profile`](round0_baseline/profile)。

## 限制

当前没有声称 round9 FP8、round6 head、round7 optimizer 或 round10 full Graph 已通过完整 C/D 门槛；FP8 的训练形状峰值不代表整步收益。ViT-L 的 20 epoch 双 seed、ViT-B/L 最终组合的 6/20 epoch 对照、导出模型 GPU `load_model().predict()` 兼容性和整步 Graph 长训练仍需独占 GPU 队列验证。所有未实测字段在 `results.json` 中保留为 `null`。

CUDA C++ kernel 当前是 correctness-first baseline：round3/4 使用每问题/每行的简单实现，round7 先按参数张量逐个 launch。它们已证明能编译、运行和对照 Torch，但尚未达到方案要求的最终融合性能；后续优化包括 flat/multi-tensor AdamW、FlashAttention tile、matcher/loss 融合和静态寄存器/SASS 分析。
