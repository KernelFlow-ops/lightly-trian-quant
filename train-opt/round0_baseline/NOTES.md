# Round0：基线与测量体系（负责：Agent A，2026-10-02）

## 1. 目标与假设

建立原版 Lightly 训练路径、公共 harness、数据/权重准备和性能测量口径。H1-H10 的完整 nsys 证据待正式 profile；当前先记录可复现的环境、batch 和一轮基线。

## 2. 改动

- `common/data.py` 将 COCO semantic mask 写入 `/root/autodl-tmp/train-opt-tmp/data/sam3_head`，源数据只读。
- `common/model_factory.py` 从 `vitl-best.pth`/`vitb-best.pth` 抽出 backbone，逐 tensor 校验。
- `common/harness.py`、`fast_forward.py`、`ops.py`、`rng.py` 提供原版等价训练步和 patch 接口。
- `B_AUTO` 按原脚本 probe 固定为 ViT-L=7、ViT-B=14。

## 3. nsys 分析过程

正式采集覆盖两个模型的 early/mid/late，各预热 20 步、捕获 5 步。命令示例（其余五组只替换模型和阶段，原始命令同名日志中保存）：

```bash
bash train-opt/common/profile_nsys.sh /root/autodl-tmp/train-opt-tmp/A/nsys/r0_vitb14_mid train-opt/round0_baseline/run.py --model vitb16 --variant B0-harness --action profile --phase mid > /root/autodl-tmp/train-opt-tmp/A/r0_vitb14_mid_nsys.log 2>&1
/root/miniconda3/bin/python -B train-opt/common/nsys_analyze.py /root/autodl-tmp/train-opt-tmp/A/nsys/r0_vitb14_mid
```

| model | phase | kernels/step | syncs/step | GPU busy | gap >20us ms/step | loss CPU ms | mask CPU ms |
|---|---|---:|---:|---:|---:|---:|---:|
| B14 | early | 6098.8 | 215 | 81.4% | 38.6 | 77.7 | 69.0 |
| B14 | mid | 6105.8 | 216 | 82.3% | 32.7 | 74.9 | 68.7 |
| B14 | late | 6105.4 | 215 | 81.9% | 35.6 | 77.3 | 69.3 |
| L7 | early | 6492.6 | 131 | 87.7% | 25.7 | 45.8 | 79.8 |
| L7 | mid | 6498.8 | 132 | 88.1% | 23.1 | 45.2 | 81.2 |
| L7 | late | 6500.0 | 131 | 88.5% | 22.1 | 45.1 | 81.7 |

证据：`profile/r0_*_summary.md`、同前缀的 kernel/API/NVTX CSV；逐调用位置和空泡归因在 `/root/autodl-tmp/train-opt-tmp/A/nsys/r0_*_analysis.json`。CSV、SQLite 和 kernel summary 的行数经分析器一致性检查。保留 6 份原始 `.nsys-rep` 于 A 的 tmp，项目 profile 不存大文件。

时间线观察：ViT-B mid 的 loss 段约有 196 次同步/步，22.1 ms/步的 >20us 空泡落在 loss；`mask_L9/L10/L11` 的 CPU 时间合计 68.7 ms，但 GPU 约 1.8 ms。`fmha_cutlassB_f32...` 单个 attention 反向热点 63.4 ms/步，ViT-L 为 85.8 ms/步。推断：H1/H3 为 CPU 同步瓶颈，H2 为 FP32 joint attention GPU 瓶颈；各段 CPU/GPU 可重叠，不能把这些数直接相加为节省时间。受控实验将使用 r1 `nosync`/`mask_skip`、r3 `batched_torch`、r4 `sdpa_rope` 单开与叠加计时核实。
ViT-B loss 的 196 次/步包括约 140 次 `cudaStreamSynchronize` 和 56 次 DtoH 相关 `cudaMemcpyAsync`；ViT-L 的 112 次/步包括 84+28。56/28 恰为 4 层 × batch 的 matcher 子问题数；nsys 此处无可用 Python backtrace，具体行以源码的 `cost_matrix.cpu()` 推断。

| 假设 | 当前判断与实测上界 | 证据/后续受控实验 |
|---|---|---|
| H1 matcher/loss CPU | 成立；B loss CPU 74.9 ms/step、196 sync，L 45.2 ms、112 sync | r3 单开对照，mid profile |
| H2 joint mask attention | 成立；FP32 attention fwd+bwd Top kernel 82.1 ms B、111.1 ms L | r4 单开对照，Top kernel |
| H3 mask 退火/disable 同步 | 成立；3 个 mask 段 CPU 68.7 ms B、81.2 ms L，GPU 分别约 1.8 ms | r1 nosync/mask_skip 对照，NVTX |
| H4 RoPE 每层 kernel | 生成部分 L 的 24 次调用合计 360 GPU ops，但仅 0.378 GPU ms、6.89 CPU ms；应用部分仍在 block 内 | `profile/r0_vitl7_mid_rope_summary.md`，r4 单开对照 |
| H5 head | 成立；4 次 predict 前向 GPU 34.8 ms B、25.9 ms L，反向需按 cuDNN kernel 归因 | r6 单开对照，NVTX/Top kernel |
| H6 启动/逐元素 | 成立；6106/6499 kernels/step，>20us 空泡 32.7/23.1 ms B/L | r2 graph、r5 fusion 对照 |
| H7 opt/clip | 有空间；opt GPU 6.0/12.9 ms B/L，CPU 发射开销与 bwd 重叠 | r1 fused AdamW、r7 对照 |
| H8 data | 触发 r8：B GPU 37.80 对 e2e 33.14 img/s（差 12.3%，data wait 14.1 ms）；L 16.25 对 14.94（差 8.1%，wait 17.0 ms） | `results.json` e2e + 原训练日志；先拆其他主循环开销，再调 loader |
| H9 val/ckpt | val 占整轮约 10.0% B、6.6% L；ckpt 尚未单独计时 | 原训练日志，后续 epoch 分段 |
| H10 GEMM | 训练形状 f32/TF32 最高 B 117.9、L 115.5 TFLOPS；bf16 234.2/225.4；FP8 683.4/578.0（wgrad 的 K 补齐至 16 倍数，未计量化） | `results/peaks_train.json`，r9 单开测试 |

## 4. 结果（当前实测）

| model | batch | variant | train epoch | val mIoU | val head IoU | train throughput | peak memory |
|---|---:|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 14 | B0-lightly | 262 steps | 0.786675 | 0.574158 | 29 img/s | 29473.8 MB |
| ViT-L/16 | 7 | B0-lightly | 524 steps | 0.813653 | 0.627894 | 13 img/s | 29283.7 MB |

ViT-B/16 的 6 epoch、两个 seed 短预算基线：seed0 `mIoU=0.831658, head IoU=0.663849`，seed1 `mIoU=0.830435, head IoU=0.661392`；均值 `mIoU=0.831046±0.000611`、`head IoU=0.662621±0.001229`。日志：`/root/autodl-tmp/train-opt-tmp/A/baseline_vitb16_e6_s0_retry.log`、`baseline_vitb16_e6_s1.log`。

GPU 预加载 8 个真实 batch、每阶段预热 20 步、测 50 步 × 3 的固定阶段计时：

| model | early ms | mid ms | late ms | 加权 img/s | late fwd/loss/bwd/opt ms |
|---|---:|---:|---:|---:|---|
| ViT-B/16 B=14 | 370.45 | 370.03 | 371.44 | 37.80 | 99.00 / 38.17 / 223.23 / 10.24 |
| ViT-L/16 B=7 | 430.27 | 430.72 | 430.93 | 16.25 | 117.24 / 21.53 / 268.79 / 22.29 |

上述 GPU step 是固定阶段、无 dataloader 的主口径；原版一轮日志是从训练第 1 步累计的吞吐，口径不同。命令和输出分别在 `/root/autodl-tmp/train-opt-tmp/A/r0_vit{b14,l7}_{early,mid,late}_bench.log`。

ViT-B 的固定真实 loader 100 步同口径对照：直接调用原 `training_step` 为 **33.2626 img/s**（数据等待 12.52 ms），harness 为 **33.1423 img/s**（数据等待 14.06 ms），差异 **0.36%**，满足 A2 的 ±3%。原一轮训练日志中的 29 img/s 是从第 1 步开始累计的平均值，含启动阶段，不能拿它与稳态 100 步直接相除。对照命令输出：`/root/autodl-tmp/train-opt-tmp/A/r0_vitb14_source_step_e2e.log`、`/root/autodl-tmp/train-opt-tmp/A/r0_vitb14_e2e.log`。

日志：`/root/autodl-tmp/train-opt-tmp/A/b0_lightly_vitb16_epoch1.log`。输出目录：`/root/autodl-tmp/train-opt-tmp/runs/b0_lightly_vitb16_seed0_epoch1`。
ViT-L 日志：`/root/autodl-tmp/train-opt-tmp/A/b0_lightly_vitl16_epoch1_retry1.log`。输出目录：`/root/autodl-tmp/train-opt-tmp/runs/b0_lightly_vitl16_seed0_epoch1_retry1`。

## 5. 正确性

- 数据 mask 抽样 10 张与源 COCO 转换逐像素一致。
- ViT-B/L backbone state_dict 逐 tensor 与导出 checkpoint 相等。
- default harness forward、loss、RNG 及更新后的参数与原训练步对齐；控制组 B 级 `grad_cos_min=0.99258`，`loss_rel=3.76e-4`，`grad_norm_rel=1.68e-4`。
- `tests/test_harness_equivalence.py` 在同权重、同真实 batch、同 RNG 下逐位验证单步 loss 和更新后的参数；`test_rng.py` 验证回放后续随机流，合计 2 tests passed。
- 20 步真实数据等价检查使用同一单进程 loader 产出的 CPU batch，再分别送入原版与 harness；`same_batches=true`、最大逐步 loss 相对差 `1.1971e-5`，通过。多 worker loader 会独立预取并消耗增强 RNG，不能用来做逐步 batch 顺序断言；其吞吐单独按 e2e 指标记录。
- RngTape replay 会先消费同形状随机数，再返回录制值；回归测试通过。

## 6. 结论、失败尝试、下一个瓶颈

- 单 GPU Fabric 的 `bf16-mixed` 配置下，直接调用 Lightly `training_step` 的 logits 实测为 fp32；显式 autocast 作为 round1 实验开关，不能默认视为等价。
- ViT-L 首轮入口遗漏原脚本 probe 使用的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，第 3 步 OOM；补齐后 524 步训练完成，保留失败日志 `/root/autodl-tmp/train-opt-tmp/A/b0_lightly_vitl16_epoch1.log`。
- round3 batched loss 的同 logits 梯度通过；RngTape 修复后完整 ViT-B mid B 级 `grad_cos_min=0.999999344`、`loss_rel=0`、通过（详见 round3 profile）。
- 下一步完成三阶段 GPU profile、harness 吞吐等价与 H1-H10 瓶颈排序。
