# train-opt：DINOv3 + EoMT 训练加速（最终状态，2026-10-03）

原项目源码、训练脚本、推理项目、数据集和权重保持只读；优化通过 `train-opt/common` 的 harness、ops 注册表和各 round 的 patch 注入，不改 Parameter 身份，不改 `state_dict`。

## 结果

同一台机器（RTX 5090，575 W）、同一套 harness（`common.bench.time_steps`，三相 early/mid/late 按 0.2/0.6/0.2 加权，warmup 20，50×3，GPU 独占）、B 固定（ViT-L/16 B=7，ViT-B/16 B=14）。最终栈 = round19 的 `FINAL`，在整步 CUDA graph 下运行。

| 模型 | R0（原版，本轮重测） | **FINAL（graph）** | 倍数 | mid 步时 | 显存 |
|---|---:|---:|---:|---|---|
| ViT-L/16，B=7 | 16.31 img/s | **68.87 img/s** | **4.22×** | 429 → 102 ms | 28.8 GB → 16.5 GB |
| ViT-B/16，B=14 | 37.77 img/s | **161.94 img/s** | **4.29×** | 371 → 87 ms | 29.2 GB → 12.5 GB |

真实 DataLoader 的端到端（200 步，稳态）：ViT-L 68.3 img/s、ViT-B 155.8 img/s，是 GPU-only 的 98%；loader 不是瓶颈（workers 16 即可，pin_memory 必须开）。**一个 epoch 里 val + checkpoint 现在占 29–33%**（ViT-L val 12.2 s + checkpoint 9.4 s，ViT-B 8.7 s + 2.7 s）。

### 瀑布（加权 img/s；R0、FINAL 是 round19 同机实测，中间阶段是各轮 NOTES 发布的数字，同类硬件不同时段，±2–3% 漂移）

| 阶段 | ViT-L | 倍数 | ViT-B | 倍数 | 来源 |
|---|---:|---:|---:|---:|---|
| R0 原版 Lightly 步（fp32/TF32） | 16.31 | 1.00× | 37.77 | 1.00× | round19 |
| P0（r1+r3+r4+r5+r7，fp32 清理） | 18.0 | 1.10× | 44.4 | 1.18× | CONTRACTS |
| TB1（BF16 autocast + flash/cuDNN attention + triton RoPE + head LN bf16） | 41.4 | 2.54× | 96.0 | 2.54× | round11 |
| + r13 融合块 | 43.7 | 2.68× | 100.7 | 2.67× | round13 |
| + r17 MXFP8（fc+proj+qkv_dw，qkv 前向 bf16） | 51.7 | 3.17× | 109.9 | 2.91× | round17 / 16 |
| + r15 扁平优化器（eager） | 56.0 | 3.43× | 116.6 | 3.09× | round15 / 16 |
| + r16 整步 CUDA graph | 59.5 | 3.65× | 123.1 | 3.26× | round16 |
| + r14 融合 head（eager，P2+r14+r15） | 65.46 | 4.01× | 152.08 | 4.03× | round19 |
| **+ r14 融合 head，graph = FINAL** | **68.87** | **4.22×** | **161.94** | **4.29×** | round19 |

完整表（三相、显存、门槛、e2e、最大 batch）在 [`round19_final/summary.md`](round19_final/summary.md)，过程和 nsys 在 [`round19_final/NOTES.md`](round19_final/NOTES.md)，机器可读结果在 `round19_final/results.json`。

## 最终栈是什么

`common.config.P2R14R15_STACK` + `P2R14R15_PATCH_KWARGS`（`round19_final/impl/presets.py` 里叫 `FINAL`）：

| 补丁（按顺序） | 变体 | 作用 |
|---|---|---|
| `round1_torch_cleanup` / `round3_loss_matcher` / `round5_block_fusion` / `round7_optimizer` | `nosync+mask_skip+adamw_fused` / `batched_torch` / `ln_fuse` / `capturable_torch` | 去同步、padded loss/matcher（可 graph）、LN 融合 |
| `round11_bf16_mixed` | `autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16` | BF16 autocast、flash / cuDNN attention、triton RoPE（= TB1） |
| `round13_gemm_fusion_sm120` | `full` | QuACK GEMM 融合块（LN/GELU/残差/γ 的 epilogue） |
| `round17_mxfp8_sm120` | `fc+proj+qkv_dw` | MXFP8（E4M3 + UE8M0 1×32）block-scaled GEMM；qkv 前向保持 bf16 |
| `round14_head_sm120` | `fused` | head：ConvT 当 GEMM + 3 个融合 CUDA kernel |
| `round15_optimizer_flat` | `flat` | 扁平 fp32 master/grad，2 个 kernel 完成 clip + AdamW，设备端 lr 日程 |
| 运行器 `round16_full_graph/impl/graph_step.py::StepGraphRunner` | `RngTape("off")`，`max_targets=2` | 前向 + loss + 反向 + 优化器整步一次 `replay()` |

## 怎么跑

```bash
ssh infer-5090
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536                      # >=32 个 DataLoader workers 需要；默认 1024 会报 "received 0 items of ancdata"
cd /root/lightly-train/train-opt/round19_final

# 最终栈，真实 loader，200 步（脚本自己取双锁；--val 末尾做一次 val，--save-ckpt DIR 末尾写 checkpoint + export）
python run_final.py --model vitl16 --steps 200 --workers 16 --persistent --log-every 50
python run_final.py --model vitb16 --steps 200 --workers 16 --persistent --val --save-ckpt /root/autodl-tmp/runs/final_vitb16_test

# 整个 20 epoch 日程（每个 epoch 末尾 val + checkpoint；这条完整循环在 round19 没有跑过，只跑了上面的 N 步路径）
python run_final.py --model vitl16 --steps 0 --epochs 20 --workers 16 --persistent --val --save-ckpt /root/autodl-tmp/runs/final_vitl16

# 重现实测 / 门槛
python run.py --model vitl16 --stack FINAL --runner graph --action bench          # 三相加权；--runner eager 是不带 graph 的对照；--stack R0 是原版
python run.py --model vitl16 --stack FINAL --runner graph --ref R0 --action gate_s # 50 步 2 batch，graph 对 R0
python run.py --model vitl16 --stack FINAL --action graph_equiv                    # graph 对 eager
python tests/gate_b_multi.py --model vitl16 --cands FINAL --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_vitl16_FINAL.json
python run.py --model vitl16 --stack FINAL --runner graph --action e2e --loader-settings 16:2:1:1 32:2:1:1 --tag x
```

要点：
- `--epochs` 定义 lr 日程和 mask 退火日程（默认 20，与原脚本一致）；`--steps N` 只跑日程的前 N 步。
- 每张图的目标数不能超过 `--max-targets`（默认 2）。本数据集是背景 + head 两个类，最大 2（round19 扫描）；加类别时设成 类别数，且 ≤4；超过容量的批退回 eager 步（可用但慢，ViT-L 上峰值 26.2 GB）。
- 训练 loader 推荐 `workers=16, prefetch=2, pin_memory=True, persistent_workers=True`；**不要关 pin**（ViT-L −7%、ViT-B −26%）。
- 稳妥档：把 `--stack FINAL` 换成 `FINAL_fc`（r17 策略 `fc`，只有 fc1/fc2 走 MXFP8；ViT-B 的梯度余弦更高，见下）。`FINAL_noMX`（全 bf16）也可用。对 FINAL 的吞吐：`FINAL_fc` 64.65 / 156.67 img/s（−6.1% / −3.3%，对 R0 3.96× / 4.15×），`FINAL_noMX` 54.37 / 139.63 img/s（对 R0 3.33× / 3.70×）。
- 检查点：带 r15 的栈需要 `run_final.py::fix_scheduler_state`（round15 的 scheduler state_dict 补丁会让 `torch.save` 失败，见 NOTES §5.5；`run_final.py` 已自动处理，自己写循环时要调用它）。
- 导出的模型与原版完全兼容：`lightly_train.load_model(export)` 直接可用，checkpoint 能在原版（未打补丁）路径上 resume（已验证）。

## 验证状态（few-batch 门槛，没有跑 epoch 训练）

| 门 | ViT-L | ViT-B |
|---|---|---|
| A（op 级，沿用各轮测试） | 全过 | 全过 |
| B-lite，8 批，对 R0 / 对 P1：全局梯度余弦 中位数 / 最小（规则：≥0.995 / ≥0.98） | 0.9965 / 0.9887，0.9969 / 0.9881，**过** | **0.9945** / 0.9824，**0.9942** / 0.9823：最小过、中位数差 0.0005，**未过**；原因是 MXFP8 本身（去掉 r14/r15 不变，`fc` 档 0.9947，全 bf16 0.9994），见 NOTES §5.2 |
| S（50 步 × 2 batch，graph 对 R0，4 个场景，EMA 偏差 ≤5%） | ≤2.05e-3 过 | ≤2.26e-3 过 |
| graph 对 eager（同 RNG） | 首步逐位相同，之后在 eager 自身噪声内 | 同左 |
| 导出 / resume / `load_model` | 全过 | 全过 |
| 数据集最大目标数 | 2（`max_targets=2` 够用） | 同左 |

**没有做**：epoch 级训练和 mIoU / head IoU 对照（Level D）——最终栈的精度只有上面的 few-batch 证据。

## 限制与已知问题

1. ViT-B 的 B-lite 中位数门没过（上表），需要协调者裁定是接受、改 `FINAL_fc` 还是全 bf16。
2. round15 的 scheduler 补丁让 checkpoint 无法 `torch.save`（已在 `run_final.py` 绕过，round15 未改）。
3. val 仍是未打补丁的 fp32 路径，checkpoint 同步写（round18 没做）：占 epoch 的 29–33%。
4. 固定 B（7 / 14）；加大 batch 吞吐几乎不变（ViT-L B=7…17 都在 69–70 img/s）。
5. graph 模式下随机数来自 torch 默认生成器，与 R0 的随机流不同；逐元素的 B 检查只能用 eager + RngTape。
6. `RngTape("off")`，train metrics 惰性更新（日志步才更新），host 不在步间同步。
7. 数据集每图最多 2 个目标（扫描 11016 个增强样本）；理论上可能出现 0 目标的样本（源码最多重采样 20 次），r3 padded 路径对 0 目标没有单独测。
8. 只在 sm_120（RTX 5090）、torch 2.12 / CUDA 13 / QuACK 0.6.5 上验证；冷启动（JIT 缓存被清空）的首次编译时间没有测。
9. NVTX 在 graph 回放内部是空的；分析 graph 用 `round16_full_graph/tests/trace_busy.py` 和 `round19_final/tests/nsys_final_report.py`。

## 失败的尝试 / 没有采用的

- r6 的 head 融合（Triton，比 torch 慢，round14 / round11 NOTES）、r10 旧的整步 graph（`as_tensor(float, device=cuda)` 的 H2D 导致捕获失败，r16 已修）：见各轮 NOTES。
- r17 `all`（qkv 前向也走 MXFP8）：吞吐更高但梯度余弦掉到 0.96–0.99，不采用。
- r14 `mx8` / `mx8_bwd`（head ConvT 走 MXFP8）：门没跑全，不发布；`mx8` 前向 mask logits 相对误差 5–6e-2。
- round12 融合 attention：kernel 写了，没有集成、没有整步数据。
- 更大的 batch：无收益（见上）。
- loader：workers 超过 16 没有收益（只增加启动延迟），prefetch 2/4/8 无差别，关 pin 大幅变慢。

## 各轮目录

| 目录 | 内容 |
|---|---|
| `common/` | harness、ops 注册表、RngTape、bench（双锁）、check（门）、config（栈）、nsys 分析、roofline |
| `round0_baseline` | 原版基线（R0）与 nsys |
| `round1_torch_cleanup` … `round10_integration` | 第一阶段（fp32 语义下的清理、padded loss、attention、优化器、整步 graph 的早期尝试、导出兼容） |
| `round11_bf16_mixed` | TB1：BF16 autocast + attention/RoPE/head LN |
| `round12_attention_sm120` | 融合 attention（kernel 已写，**未集成**） |
| `round13_gemm_fusion_sm120` | QuACK GEMM 融合块 |
| `round14_head_sm120` | 融合 head |
| `round15_optimizer_flat` | 扁平优化器 |
| `round16_full_graph` | 整步 CUDA graph 运行器 `StepGraphRunner` |
| `round17_mxfp8_sm120` | MXFP8 量化器 + block-scaled GEMM |
| `round19_final` | 最终栈：`run_final.py`（训练入口）、`run.py`（实测/门槛）、`tests/`、`profile/`、`NOTES.md`、`summary.md`、`results.json` |

## 下一步（按优先级，估计值，详见 NOTES §7.5）

1. val BF16 + graph 快路径、checkpoint 异步（round18）：每个 epoch 省约 19–21%（ViT-L）。
2. head ConvT 的 dgrad/wgrad 走 MXFP8（r14 `mx8_bwd`，补门）：约 −4 ms（L）/ −5 ms（B）每步。
3. 量化融合进生产者：约 −2 到 −3 ms。
4. wgrad 直接累加进 flat grad buffer：约 −1.5 到 −2 ms。
5. round12 融合 attention 集成：约 −2 到 −3 ms。
6. 小 kernel 清理（cat / copy / fill / reduce）：约 −1 到 −1.5 ms。
7. FP8 attention：约 −3 到 −5 ms，风险高。
