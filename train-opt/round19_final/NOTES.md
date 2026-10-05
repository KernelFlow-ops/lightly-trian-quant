# Round19：最终栈验证与打包（负责：协调者角色代理，2026-10-03）

> 全部数字为 `infer-5090`（RTX 5090，575 W 上限）上本轮实测；B 固定（ViT-L 7 / ViT-B 14），三相 early/mid/late 按 0.2/0.6/0.2 加权，warmup 20，50×3，`common.bench.time_steps`，GPU 独占（每个 bench json 的 `gpu_snapshot_before/after` 在计时期间只有本进程）。原始 json 在 `profile/`，nsys 原始文件在 `/root/autodl-tmp/train-opt-tmp/round19_final/nsys/`，日志在 `/root/autodl-tmp/train-opt-tmp/round19_final/logs/`。
> 本轮没有改 `common/`、其它 round、`src/`、`infer-opt/`，也没有新增自研 kernel；只新增了运行器、门槛脚本和分析脚本（§2）。

## 0. 摘要

- **最终候选栈 FINAL** = `common.config.P2R14R15_STACK`（`r1,r3,r5,r7,r11` = TB1 BF16 + `r13:full` 融合块 + `r17:fc+proj+qkv_dw` MXFP8 + `r14:fused` 融合 head + `r15:flat` 扁平优化器），在 `round16 StepGraphRunner` 下整步捕获，`RngTape(mode="off")`。
- **整步（三相加权，同一台机器、同一套 harness、GPU 独占）**

| 模型 | R0（本轮重测） | FINAL eager | **FINAL graph** | graph 对 R0 | 显存（R0 → FINAL） |
|---|---:|---:|---:|---:|---|
| ViT-L B=7 | 16.31 img/s（429 ms） | 65.46（107.1 ms） | **68.87 img/s（101.8 ms）** | **4.22×** | 28.8 GB allocated → 15.6 GB eager 峰值 / 16.5 GB reserved（graph） |
| ViT-B B=14 | 37.77 img/s（371 ms） | 152.08（92.3 ms） | **161.94 img/s（86.6 ms）** | **4.29×** | 29.2 GB → 12.0 GB / 12.5 GB |

- **门（FINAL 栈，两个模型）**：见 §5。要点——Gate A 沿用各轮测试全过；**B-lite（≥8 批，对 R0 和对 P1）：ViT-L 通过（对 R0 中位数 0.9965、最小 0.9887；对 P1 中位数 0.9969、最小 0.9881，loss_rel ≤2.0e-3）；ViT-B 在"中位数 ≥0.995"这一条上差 0.0005（对 R0 中位数 0.9945，对 P1 0.9942；最小 0.9824 ≥0.98，loss_rel ≤2.6e-3）**，二分结果见 §5.2；S 门 8/8 通过（graph 对 R0，EMA 偏差 ≤2.3e-3）；graph 对 eager 等价通过。
- **数据集最大目标数 = 2**（背景 + head），`max_targets=2` 够用（§5.4）。
- GPU 忙碌率 0.982 / 0.984（nsys，graph 回放），每步 2475 / 1641 个 kernel，graph 内 0 次同步。

## 1. 目标与假设

- 目标：把 r11–r17 的各个部件在**同一个栈**上验证并打包：(1) 实测最终栈整步（graph 与 eager 两种，两个模型）和 R0 同机重测；(2) 最终栈上的 A/B-lite/S/graph-vs-eager 门；(3) 数据集目标数检查；(4) 真实 loader 的 200 步端到端、loader 扫描、一次 val、一次 checkpoint；(5) 导出兼容；(6) 最终 nsys + roofline 估计；(7) README / NOTES / results / summary。
- 假设（各轮分别验证过，本轮验证组合）：r14（`ops.predict`）、r17（`ops.block`）、r15（`ops.opt_step`）覆盖不同的 ops，互不依赖；r14 的 head 是一个 `autograd.Function`，r14 在 eager 下实测可被 graph 捕获；各部件的收益在各轮分别测过（P2+r15+graph 59.5 / 123.1 img/s；r14 在 P2 + r7 优化器 eager 上 +13.7% / +26.8%），组合从没有一起测过，所以这里实测而不是外推。
- 风险点：r14 的门 B/S 是在 `TB1_R13` 上跑的，没覆盖 late 相和 r17 MXFP8 叠加；r3 padded loss 的 `max_targets` 没有对照数据集；ViT-L 溢出批次回退 eager 的显存。

## 2. 改动（全部在 `train-opt/round19_final/`，没有 kernel）

| 文件 | 内容 |
|---|---|
| `run_final.py` | **用户入口**：最终栈 + `StepGraphRunner` + 真实 Lightly DataLoader 的训练脚本（`--steps N`，`--epochs` 定义 lr/mask 退火日程，`--val`，`--save-ckpt DIR`，loader 的 workers/prefetch/pin/persistent 可调）；同时导出 `final_cfg / make_loader / make_graph_runner / train_loop / validate_and_save` 供测试脚本复用。只有短路径（N 步 + 末尾 val + checkpoint）在本轮跑过，完整的逐 epoch val/ckpt 循环未跑。 |
| `run.py` | 本轮实测入口，是 `round16_full_graph/run.py` 的薄封装（输出重定向到本目录）：`bench / profile / graph_equiv / gate_b`（沿用 round16）+ 新增 `gate_s`（`--stack` 的 graph 路径对 `--ref`，默认 R0）、`maxbatch`、`e2e`。 |
| `impl/presets.py` | 预设：`R0 / TB1 / P2 / P2R15 / P2R14 / FINAL`，以及二分用的 `FINAL_noR14 / FINAL_noR15 / FINAL_fc / FINAL_noMX`。 |
| `tests/gate_b_multi.py` | 多批门 B-lite（同一进程、同一批、同一 RngTape：R0 / TB1 / 候选各跑一步），按 Amendment 2 §10.2 + Amendment 4 汇总（中位数/最小/loss_rel，对 R0 和对 P1；逐参数偏差只记录）。 |
| `tests/check_export.py` | 导出兼容：ckpt/export 的键、形状、dtype 对 R0 和对原始训练导出；优化器/调度器结构；NaN 检查；`lightly_train.load_model` + predict；在原始 R0 路径上 resume 一步。 |
| `tests/test_fallback.py` | 溢出批次（一张图 3 个目标 > `max_targets=2`）在最终栈上的 eager 回退与显存。 |
| `tests/scan_dataset.py` | 数据集目标数扫描（COCO、语义 mask PNG、真实 pipeline 含增强 3 遍）。 |
| `tests/ro_run.py` | 运行旧轮次的 Gate A 脚本而不覆盖它们的结果文件（把写入重定向到 `profile/gateA_*`）。 |
| `tests/nsys_final_report.py`、`tests/roofline_final.py` | graph 模式 nsys 报告（类别、top-15、忙碌率、同步）和 roofline 估计。`tests/nsys_queue.sh`、`queue_bench.sh`、`queue2.sh`、`queue3.sh` 是实际跑过的队列。 |
| `tests/make_results.py`、`tests/make_summary.py` | 由 `profile/*.json` 生成 `results.json` 和 `summary.md`。 |


## 3. nsys 分析过程

**命令**（持双锁；沿用 round16 的 `locked_run.sh` + `nsys_inner.sh`，即 `nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node --capture-range=cudaProfilerApi`，窗口 = 预热 20 步之后的 5 个 mid 相 graph 回放；原始 `.nsys-rep/.sqlite` 在 tmp，项目里只放 csv/json/summary）：

```bash
cd round19_final
tests/nsys_queue.sh v1 vitl16:FINAL:graph vitb16:FINAL:graph
#  = round16 tests/locked_run.sh bash round16_full_graph/tests/nsys_inner.sh <prefix> run.py --model M --stack FINAL --runner graph --action profile --phases mid --warmup 20
#  + python round16_full_graph/tests/trace_busy.py <prefix>        # 忙碌率/间隙由原始 kernel trace 算（common/nsys_analyze.py 对 graph 回放不适用，round16 §3）
python tests/nsys_final_report.py --json profile/nsys_report.json --label L=<prefix> --label B=<prefix> --label Lpre=<round16 P2+r15 graph v2> --label Bpre=<…>
python tests/roofline_final.py --report profile/nsys_report.json --json profile/roofline_final.json
```
完整输出在 `profile/nsys_report.txt`、`profile/roofline_final.txt`；"before" = round16 的 `P2+r15 graph v2`（最终栈去掉 r14，同一容器，同一方法）。

**3.1 整步（mid 相，5 步窗口）**

| 模型 / 栈 | kernel 数/步 | kernel 总时间 ms/步 | 窗口 ms/步 | GPU 忙碌率 | >20 µs 间隙 个/步 → ms/步 | 同步（窗口内） |
|---|---:|---:|---:|---:|---|---|
| ViT-L P2+r15 graph（before，round16） | 2573 | 115.82 | 117.94 | 0.981 | 1.0 → 0.75 | 1（窗口末尾的 `torch.cuda.synchronize`） |
| **ViT-L FINAL graph** | **2475** | **98.64** | **100.32** | **0.982** | 1.0 → 0.44 | 1（同上，即 0 次/步） |
| ViT-B P2+r15 graph（before，round16） | 1767 | 112.81 | 114.46 | 0.984 | 2.2 → 0.77 | 1 |
| **ViT-B FINAL graph** | **1641** | **85.14** | **86.34** | **0.984** | 1.8 → 0.39 | 1（即 0 次/步） |

**3.2 NVTX / 同步 / host**：graph 回放内部没有 NVTX 区间（kernel 以 `--cuda-graph-trace=node` 展开，GPU 投影为空），NVTX 只剩 5 个 CPU 侧 `step_NN` 区间：ViT-L 8.6 / 4.1 / 6.3 / 6.7 / 6.7 ms，ViT-B 5.2 / 3.5 / 3.1 / 4.0 / 4.0 ms（一步的 host 发射时间：D2D 取批、`fill_` 概率、`cudaGraphLaunch`，平均 6.5 / 4.0 ms，远小于 100 / 86 ms 的 GPU 时间，host 提前一步以上）。`cuda_api_sum`：5 步里 `cudaGraphLaunch` 5 次，`cudaDeviceSynchronize` 1 次（窗口末尾），`cudaMemcpyAsync` 70 / 105 次（每步 14 / 21 次，批输入和 mask 的 D2D/staging），没有 `cudaStreamSynchronize`；即 **0 次/步同步**。忙碌率 0.982 / 0.984，窗口比 kernel 总时间多 1.7 / 1.2 ms（间隙 0.44 / 0.39 ms 加上小间隙），所以 graph 之后剩下的只有 kernel 时间。

**3.3 前后类别（ms/步，kernel 时间）**

| 类别 | ViT-L before → FINAL | ViT-B before → FINAL |
|---|---|---|
| cuDNN 卷积（before：head 的 ConvT/dwconv；after：只剩 patch embed） | 22.28 → 0.71 | 30.45 → 1.09 |
| head：QuACK ConvT fwd/dgrad + einsum bwd | – → 11.77 | – → 13.35 |
| head：cutlass sm80 ConvT wgrad / einsum / mask head | 6.20 → 7.23 | 6.21 → 8.21 |
| r14 融合 kernel（GELU+dwconv+LN2D fwd/bwd1/bwd2） | – → 6.64 | – → 10.30 |
| 逐元素 / copy / reduce / 其它 | 15.84 → 6.91 | 22.92 → 7.75 |
| LN / 行 kernel | 11.04 → 5.89 | 13.28 → 5.04 |
| MXFP8 GEMM（骨干） | 24.90 → 24.38 | 14.84 → 14.47 |
| attention（flash + RoPE） | 16.33 → 16.05 | 12.90 → 12.70 |
| 优化器（r15） | 7.72 → 7.72 | 2.25 → 2.25 |
| MXFP8 量化 | 5.11 → 5.07 | 4.09 → 4.10 |
| bf16 骨干 GEMM（qkv 前向） | 4.91 → 4.77 | 2.77 → 2.70 |
| loss / matcher / mask | 1.50 → 1.49 | 3.10 → 3.17 |
| **合计** | **115.82 → 98.64** | **112.81 → 85.14** |

r14 的收益全部来自 head：把 cuDNN 的 ConvT/dwconv/LN2D 路径（22.3 / 30.5 ms）换成"GEMM + 3 个融合 kernel"，同时带走了约 9 / 15 ms 的逐元素和约 5.2 / 8.2 ms 的 LN 类 kernel；净 −17.2 / −27.7 ms kernel 时间，窗口 −17.6 / −28.1 ms。其余类别与 before 相同（r14 与 r17/r15 没有干涉）。

**3.4 前 15 个 kernel（FINAL graph，mid；完整表在 `profile/nsys_report.txt`）**

ViT-L：

| # | ms/步 | % | 次/步 | 平均 µs | 类别 | kernel |
|---|---:|---:|---:|---:|---|---|
| 1 | 10.00 | 10.1 | 24 | 416.5 | attention | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` |
| 2 | 9.88 | 10.0 | 16 | 617.6 | head GEMM bf16 | QuACK `quackgemm_default`（ConvT fwd/dgrad，每次 predict 4 个） |
| 3 | 9.36 | 9.5 | 96 | 97.5 | MXFP8 GEMM | QuACK `quackgemm_default`（fc/proj/qkv 的 dgrad/wgrad） |
| 4 | 6.64 | 6.7 | 1 | 6641.8 | 优化器 | r15 `update_kernel` |
| 5 | 5.59 | 5.7 | 8 | 698.5 | head GEMM bf16 | `cutlass_80_tensorop_s16816gemm_bf16_64x256_32x4_nt`（ConvT wgrad） |
| 6 | 4.77 | 4.8 | 24 | 198.7 | bf16 骨干 GEMM | QuACK `bias_epi`（qkv 前向，bf16） |
| 7 | 4.70 | 4.8 | 48 | 98.0 | MXFP8 GEMM | QuACK `res_epi`（proj/fc2 前向 + γ + 残差） |
| 8 | 4.66 | 4.7 | 72 | 64.7 | MXFP8 GEMM | QuACK `quackgemm_default` |
| 9 | 4.12 | 4.2 | 24 | 171.5 | attention | `flash_fwd_kernel` |
| 10 | 3.93 | 4.0 | 168 | 23.4 | MXFP8 量化 | `quant_dual_kernel<bf16>` |
| 11 | 2.80 | 2.8 | 21 | 133.3 | MXFP8 GEMM | QuACK `dgelu_epi`（fc1 dgrad） |
| 12 | 2.53 | 2.6 | 8 | 316.5 | r14 | `bwd2_kernel<1024,4,2>` |
| 13 | 2.47 | 2.5 | 24 | 103.0 | MXFP8 GEMM | QuACK `gelu_epi`（fc1 前向） |
| 14 | 2.39 | 2.4 | 8 | 299.2 | r14 | `bwd1_kernel<1024,4,2>` |
| 15 | 1.92 | 1.9 | 24 | 80.1 | LN | `rowbwd_kernel<1024,…>`（r13） |

ViT-B 的前 5 个：QuACK head GEMM 11.25 ms（16 次，703 µs）、flash bwd 7.47 ms、`cutlass_80_…128x64_64x3_nt` head wgrad 6.12 ms、QuACK MXFP8 `default` 5.49 ms、r14 `bwd1_kernel<768>` 3.90 ms（其后是 r14 `bwd2` 3.71、量化 3.69、flash fwd 3.02）。before（round16 P2+r15 graph）的头几名是 flash bwd 10.21、MXFP8 `default` 9.52、优化器 6.65、cuDNN ConvT wgrad `sm80_xmma_wgrad…` 5.46、cuDNN dgrad 5.39、cuDNN fprop 4.24 和一个 31 次 × 150 µs 的 `direct_copy_kernel`（4.64 ms，推断为 head 的 NCHW↔NHWC permute 拷贝，未逐调用核对），这些在 FINAL 里都没有了。

**3.5 结论**：(1) graph 把 host 开销全部移出关键路径（忙碌率 0.98，0 次同步/步），r14 之后仍然成立，所以 r14 的 kernel 时间收益几乎 1:1 变成墙钟（ViT-L 窗口 117.9 → 100.3 ms，bench mid 117.8 → 101.8 ms）。(2) 现在 ViT-L 的 98.6 ms kernel 时间里，MXFP8+bf16 骨干 GEMM 29.2 ms（30%）、head 的 GEMM + 融合 kernel 25.6 ms（26%）、attention 16.1 ms（16%）、优化器 7.7 ms（8%）、其余（量化 5.1、LN 5.9、逐元素 6.9、loss 1.5）19.4 ms。ViT-B 上 head 的 GEMM + 融合 kernel 是 31.9 ms（37%），已经是最大块。(3) 下一步只能靠减少 kernel 时间，见 §7 的 roofline 估计。


## 4. 实测结果

**4.1 整步（三相加权；括号为 mid 步时 ms；GPU 独占；同一容器、同一 harness）**

| 栈 / runner | ViT-L B=7 | ViT-B B=14 |
|---|---|---|
| **R0**（原版 Lightly 步，本轮重测） | 16.31 img/s（429.1 ms） | 37.77 img/s（370.7 ms） |
| FINAL eager（P2+r14+r15，无 graph） | 65.46（107.1 ms） | 152.08（92.3 ms） |
| **FINAL graph** | **68.87（101.8 ms）** | **161.94（86.6 ms）** |
| FINAL graph 对 R0 | **4.22×** | **4.29×** |
| FINAL graph 对 FINAL eager | 1.052× | 1.065× |
| P2+r15 graph（同时段漂移核对；已发布 59.5 / 123.1） | 59.37（118.1 ms） | 123.75（113.3 ms） |
| TB1 eager（同时段漂移核对；已发布 41.4 / 96.0） | 41.94（167.0 ms） | 97.93（143.1 ms） |

三相（early / mid / late）：ViT-L FINAL graph 101.8 / 101.8 / 101.2 ms（68.8 / 68.8 / 69.2 img/s），ViT-B 87.2 / 86.6 / 85.4 ms；R0 三相相同（429 / 429 / 430 ms，370 / 371 / 372 ms），三个 repeat 的 mid 均值很稳（ViT-L graph 101.54 / 101.74 / 102.01 ms，ViT-B 86.36 / 86.53 / 86.78 ms）。完整三相、峰值显存、repeat 在 `profile/bench_*_all.json`，瀑布表在 `summary.md`。

**4.2 瀑布（R0 → P0 → TB1 → +r13 → +r17 → +r15/r16 → +r14；加权 img/s，对 R0 的倍数；除"本轮"外沿用各轮 NOTES 的数字，同类硬件、不同时段，±2–3% 漂移）**

| 阶段 | ViT-L | 倍数 | ViT-B | 倍数 | 来源 |
|---|---:|---:|---:|---:|---|
| R0 | 16.31 | 1.00× | 37.77 | 1.00× | 本轮 |
| P0（r1+r3+r4+r5+r7，fp32） | 18.0 | 1.10× | 44.4 | 1.18× | CONTRACTS 9.3 |
| TB1（BF16 autocast + flash/cuDNN attention + triton RoPE + head LN bf16） | 41.4 | 2.54× | 96.0 | 2.54× | CONTRACTS 9.3 |
| + r13 融合块 | 43.7 | 2.68× | 100.7 | 2.67× | CONTRACTS 10.1 |
| + r17 MXFP8（fc+proj+qkv_dw） | 51.7 | 3.17× | 109.9 | 2.91× | round16 §4.1 |
| + r15 扁平优化器（eager） | 56.0 | 3.43× | 116.6 | 3.09× | round16 §4.1 |
| + r16 整步 CUDA graph | 59.5 | 3.65× | 123.1 | 3.26× | round16 §4.1 |
| + r14 融合 head（eager，P2+r14+r15） | 65.46 | 4.01× | 152.08 | 4.03× | 本轮 |
| **+ r14 融合 head，graph = FINAL** | **68.87** | **4.22×** | **161.94** | **4.29×** | 本轮 |

（r14 单独在 P2 + r7 优化器 eager 上测得 59.07 / 139.95，round14 §0；r14 在 graph 之上的增量是 59.5 → 68.87（1.157×）/ 123.1 → 161.94（1.315×）。）

**4.3 r17 精度档位的代价（graph，3 相加权；用于 §5.2 的 B 门裁定）**

| 栈（graph） | ViT-L img/s（mid ms） | 对 FINAL | ViT-B img/s（mid ms） | 对 FINAL | reserved GB（L / B） | ViT-B B-lite 严格规则 |
|---|---|---:|---|---:|---|---|
| **FINAL**（r17 `fc+proj+qkv_dw`） | 68.87（101.8） | 1.000 | 161.94（86.6） | 1.000 | 16.50 / 12.52 | FAIL（中位 0.9924–0.9945） |
| FINAL_fc（r17 `fc`） | 64.65（108.4） | 0.939 | 156.67（89.6） | 0.967 | 16.12 / 12.43 | 对 R0 中位 0.9947（差 0.0003），对 P1 过 |
| FINAL_noMX（全 bf16：r13 + r14 + r15） | 54.37（128.9） | 0.789 | 139.63（100.5） | 0.862 | 16.03 / 12.76 | PASS（0.9994） |

即：对 R0 的倍数，FINAL_fc 是 3.96× / 4.15×，FINAL_noMX 是 3.33× / 3.70×。ViT-L 上 FINAL 已过严格规则，不需要降档。

**4.4 显存**：R0 峰值 allocated 28.81 / 29.20 GB（reserved 30.02 / 29.70 GB，已经贴着 32 GB）；FINAL eager 峰值 allocated 15.56 / 11.95 GB；FINAL graph 捕获后 reserved 16.50 / 12.52 GB（`bench.time_steps` 之后的 allocated 不含 graph 私有池，round16 §5.5；`nvidia-smi` 计时期间 16.9 GB / 13.2 GB）。r14 把 P2+r15 graph 的 ViT-L reserved 从 18.1 GB 降到 16.5 GB。

**4.5 最大 batch（ViT-L，FINAL graph，mid 相，warmup 20 + 30 步，次要指标；每个 B 一个进程）**

| B | mid 步时 ms | img/s | reserved GB |
|---:|---:|---:|---:|
| 7（3 相 bench 的 mid） | 101.8 | 68.79 | 16.50 |
| 9 | 130.0 | 69.24 | 19.30 |
| 10 | 144.2 | 69.36 | 20.69 |
| 11 | 159.3 | 69.06 | 22.02 |
| 12 | 172.5 | 69.58 | 23.43 |
| 13 | 185.4 | 70.11 | 24.88 |
| 14 | 199.7 | 70.10 | 26.25 |
| 15 | 213.5 | 70.26 | 27.58 |
| 16 | 228.0 | 70.17 | 29.08 |
| **17** | 241.9 | 70.27 | 30.43（`mem_get_info` 已用 31.12 / 31.36 GB） |
| 18 | OOM（`torch.OutOfMemoryError`，allocated 30.2 GB） | – | – |

**最大 batch = 17**（R0 在 B=7 就要 28.8 GB allocated，更大的 B 没测）。吞吐 B=7…17 一直在 68.8–70.3 img/s，B=17 只比 B=7 高 2.1%，每多 1 张图 reserved +1.4 GB：GEMM 的 M=7203 已经饱和，唯一随 B 摊薄的只有 7.7 ms 的优化器；另外 B≠7 会改变 `total_steps = 3672//B × epochs` 的含义（lr / 退火日程）。结论：大 batch 不是收益来源。

**4.6 端到端（真实 Lightly DataLoader，FINAL graph，200 步）**

| 模型 | GPU-only（early，预载批） | e2e 稳态（第 21–200 步） | 稳态 / GPU-only | 含启动（200 步，`time_e2e` 口径） |
|---|---|---|---:|---|
| ViT-L B=7 | 69.40 img/s（100.9 ms） | **68.26 img/s**（16 workers，pin on，prefetch 2） | 0.984 | 64.62 img/s |
| ViT-B B=14 | 158.50 img/s（88.3 ms） | **155.78 img/s**（64 workers；16 workers 155.68） | 0.983 | 139.11 img/s（16 workers） |

loader 不是瓶颈：workers 16 / 32 / 64、prefetch 2 / 4 / 8 的稳态吞吐彼此相差 <0.5%（ViT-L 67.6–68.3、ViT-B 155.4–155.8），都是 GPU-only 的 97.4–98.4%；多出来的 workers 只增加启动延迟（首个 batch：ViT-L 1.15 / 2.04 / 4.85 s，ViT-B 2.15 / 2.73 / 8.32 s，对应 16 / 32 / 64 workers）。**`pin_memory=False` 代价很大**：ViT-L 64.77 img/s（0.933），ViT-B 116.91（0.738），host 等待 107–118 ms/步——必须保持 pin 开。`persistent_workers=True` 稳态相同、host 等待更小（ViT-L 1.1 ms 对 3.5–4.0 ms），而且省掉每个 epoch 开头重新拉起 workers 的 1–5 s（epoch 只有 ViT-L 54 s / ViT-B 24 s）。**推荐设置：workers=16、prefetch=2、pin_memory=True、persistent_workers=True，训练前 `ulimit -n 65536`**（默认 1024 的软限制在 ≥32 workers 时会报 `received 0 items of ancdata`，本轮 dataset 扫描第一次就撞上了）。`run_final.py --steps 200 --workers 16 --persistent` 的实跑：ViT-L 67.87 img/s 稳态（103.1 ms/步），ViT-B 155.78（89.9 ms/步），1 次捕获、0 次 eager 回退，峰值 reserved 16.57 / 12.82 GB。

**4.7 一次 val 和一次 checkpoint（20 个真实步之后，Lightly 的 `save_checkpoint` / `export_model`，val 走未打补丁的 fp32 路径，477 张）**

| 模型 | val 一遍 | checkpoint（last.ckpt） | export（exported_last.pt） | 大小 |
|---|---:|---:|---:|---|
| ViT-L | 12.2 s | 7.94 s | 1.50 s | 3.52 GB + 1.18 GB |
| ViT-B | 8.7 s | 2.36 s | 0.34 s | 1.03 GB + 0.35 GB |

**由实测分量估计的单个 epoch（不是实测 epoch，没有跑 epoch 训练）**：ViT-L 训练 3668 张 / 68.3 img/s = 53.7 s，加 val 12.2 s、checkpoint 9.4 s，约 75 s；R0 训练 3668 / 16.31 = 225 s，加同样的 val 和 checkpoint，约 247 s，即 epoch 级约 3.3×（训练部分 4.2×）。ViT-B：23.6 s + 8.7 + 2.7 ≈ 35 s，对 R0 的 97 s + 11 ≈ 108 s，约 3.1×（训练部分 4.3×）。**val + checkpoint 现在占 epoch 的 29% / 33%**，这是 §7 里排第一的"非 kernel"优化项（val 走 BF16 + graph 快路径、checkpoint 异步）。

## 5. 正确性

**5.1 Gate A（op 级，沿用各轮测试，输出由 `tests/ro_run.py` 重定向到 `profile/gateA_*`，没有覆盖原轮次文件）**

| 测试（来源） | 覆盖的 op | 结果 |
|---|---|---|
| `round17 test_quant.py` | MXFP8 双输出量化器对 QuACK `to_mx`（rceil），bf16/fp32 输入，M∈{7203,7903,1029,14406,15806,128,33}，normal/spiky/scale8/tiny/large/zeros/pow2 | 98 例，**0 个不一致**，NaN/Inf 的 scale 字节一致 |
| `round17 test_gemm_mx8.py --quick` | MXFP8 GEMM + epilogue（fwd bias / γ+残差 / GELU，dgrad，dgrad GELU'，wgrad），ViT-L M=7203，normal + spiky | 8 例，对反量化 fp32 乘积 rel L2 ≤1.73e-3（0 例 >5e-3）；对原始 fp32 乘积 fwd/dgrad 3.6–3.9%、wgrad 0.95–5.1%（量化误差，spiky 的 wgrad 偏低是尖峰通道主导范数） |
| `round17 test_block_mx8.py --variants fc+proj+qkv_dw` | MXFP8 整块 fwd+bwd，两模型 × plain/joint × normal/spiky | 8 例 ALL PASS：out 余弦 ≥0.99993、dx ≥0.9999、全局梯度余弦 0.99837–0.99904、单参数最低 0.9953（r13 的 bf16 块全局 0.99999） |
| `round13 test_fused_block.py` | r13 bf16 融合块（qkv 前向仍走它） | ALL PASS（fused/fp32 余弦 ≥0.999992） |
| `round14 test_gateA.py` | 融合 head 对 fp32 源码实现（输出 + 全部参数梯度 + x 梯度），gauss 与 scale-8 尖峰，两模型 | GATE A PASS：mask 余弦 0.99997 / 0.99997，grad cos_min ≥0.99996（比 torch bf16 head 更小误差）；`test_graph.py` 的 graph 回放对 eager：mask/class 余弦 1.0，grad cos ≥0.9999999，每次 predict 6.87 ms（L）/ 8.67 ms（B） |
| `round15 test_flat_synthetic.py` | flat 优化器 100 步对 oracle / torch AdamW，lr 调度，ckpt 往返，确定性，冻结参数 | pass（ckpt 往返逐位一致，`deterministic_bitwise: true`） |
| `round11 test_round11_gpu.py`（pytest） | plain / joint(masked) attention 对 fp32，含 spiky，N=1029/1129 | 56 passed |
| `round3 test_padded_graph.py`（pytest） | r3 padded loss 前向+反向可 graph 捕获与回放 | 1 passed |

没有重跑的：round12 的融合 attention（未集成）、r13 的 `test_block_kernels.py` / `test_round13_cpu.py`、r14 的 Triton 版测试，因为它们不在最终栈的执行路径上。

**5.2 Gate B-lite（≥8 批，同一进程内同权重/同批/同 RngTape 的 R0 / P1 / FINAL(eager) 各一步；准则 Amendment 2 §10.2：全局 grad 余弦中位数 ≥0.995 且最小 ≥0.98，loss_rel ≤2e-2，分别对 R0 和对 P1；Amendment 4：对 R0 余弦本身 <0.9 的参数只看全局，其余逐参数只记录；批次按 (进程, seed) 计，每个 seed 是不同的真实 batch，`batch_sum` 逐个不同）**

| 模型 | 批数 | 参考 | 全局 grad 余弦 中位 / 最小 | loss_rel 最大 | grad_norm_rel 最大 | 结论 |
|---|---:|---|---|---:|---:|---|
| ViT-L | 8 | R0 | 0.99649 / 0.98865 | 1.9e-3 | 0.011 | **PASS** |
| ViT-L | 8 | P1 | 0.99694 / 0.98805 | 2.0e-3 | 0.010 | **PASS** |
| ViT-B | 8 | R0 | **0.99451** / 0.98240 | 2.6e-3 | 0.017 | **FAIL（中位数差 0.0005）** |
| ViT-B | 8 | P1 | **0.99419** / 0.98230 | 2.6e-3 | 0.018 | **FAIL（中位数差 0.0008）** |

逐批全局余弦（对 R0）：ViT-L 0.9992 / 0.9982 / 0.9902 / 0.9988 / 0.9976 / 0.9954 / 0.9943 / 0.9887；ViT-B 0.9986 / 0.9950 / 0.9938 / 0.9933 / 0.9942 / 0.9824 / 0.9966 / 0.9948；同一批上 P1 对 R0 是 0.9977–1.0000（ViT-L）/ 0.9969–0.9997（ViT-B）。loss_rel、grad_norm_rel、update 余弦（1.0000）都远在门槛内。逐参数偏差（相对 `cos(P1,R0)−0.05` 且对 P1 <0.99，记录不卡）：ViT-L 每批 2–61 个参数，ViT-B 0–44 个，集中在 block 0–1 的 LN / proj / qkv / ls，与 Amendment 4 描述的敏感参数一致。

**二分（ViT-B，12 批，所有候选在同一批上；ViT-L 的 FINAL 已过，没有做二分——跑了一半为节省时间停掉，半成品 json 已删）**

| 候选 | 对 R0：中位 / 最小 | 对 P1：中位 / 最小 | loss_rel 最大（对 R0） | 严格规则 |
|---|---|---|---:|---|
| FINAL（r17 `fc+proj+qkv_dw`） | 0.99241 / 0.98803 | 0.99409 / 0.98873 | 3.3e-3 | FAIL |
| FINAL_noR14（去掉 r14，= P2+r15） | 0.99180 / 0.98795 | 0.99347 / 0.98836 | 3.4e-3 | FAIL |
| FINAL_fc（r17 策略换成稳妥档 `fc`） | 0.99467 / 0.99151 | 0.99587 / 0.99204 | 3.3e-3 | 对 R0 中位差 0.0003（FAIL），对 P1 过 |
| FINAL_noMX（r13 bf16 块 + r14 + r15，无 MXFP8） | 0.99938 / 0.99853 | 0.99946 / 0.99768 | 3.5e-4 | PASS |

结论：**未过的原因是 r17 的 MXFP8（任何策略），不是 r14 或 r15**——去掉 r14 中位数不变（0.9918 vs 0.9924），去掉 MXFP8 回到 0.9994；`fc` 档接近但在对 R0 的中位数上仍差 0.0003。这和 round17 自己的统计一致（ViT-B `fc+proj+qkv_dw` 6 批均值 0.9938、`fc` 20 批均值 0.9960，r17 §5）。门的"中位数 ≥0.995"对 ViT-B 的 MXFP8 栈本来就是边缘的；所有其它指标（最小 ≥0.98、loss_rel ≤3.3e-3、grad_norm_rel ≤0.03、S 门偏差 ≤2.3e-3）都有大余量。**需要协调者裁定：接受 ViT-B 的 0.9945 / 0.9924（FINAL）、或 ViT-B 改用 `FINAL_fc`、或要求全 bf16**；性能代价见 §4.3。

**5.3 Gate S（`run.py --action gate_s`：FINAL graph 对 R0 eager，2 个真实 batch，50 步，`gate_c(interval=10, max_rel=0.05)`，无 NaN/Inf；R0 走原版 list-targets 路径，graph 走 padded targets）**

| 模型 | schedule（真实日程） | fixed mid | fixed late | compressed（退火压缩，7 个签名、7 次捕获） |
|---|---:|---:|---:|---:|
| ViT-L | 1.77e-3 PASS | 1.32e-3 PASS | 2.05e-3 PASS | 1.64e-3 PASS |
| ViT-B | 2.26e-3 PASS | 9.39e-4 PASS | 1.78e-3 PASS | 7.93e-4 PASS |

数字是 EMA(0.9) 损失在第 10–50 步的最大相对偏差，阈值 5e-2；grad_norm 比值 0.95–1.05。8/8 通过。首步 loss：R0 29.97 / FINAL 29.92（ViT-B schedule）、31.56 / 31.59（ViT-L schedule）。mid / late 相和 compressed 都覆盖了（r14 原先的 S 门没跑 late）。

**5.4 graph 对 eager，同一 RNG（`--action graph_equiv`，每步 `torch.cuda.manual_seed(7000+k)`）**：首步 loss 在全部 6 个场景里逐位相同；之后 graph 对 eager 的 loss 相对差最大 1.7e-4，更新相对 L2 ≤7.5e-3、更新余弦 ≥0.99997，与 eager 自己两次运行的差异同量级（eager 对 eager：loss 相对差最大 3.6e-4，更新相对 L2 最大 2.4e-2；flash bwd / split-K 原子加是非确定的）。表在 `summary.md`。

**5.5 导出兼容（`run_final.py --steps 20 --val --save-ckpt` 写 checkpoint + export，再由 `tests/check_export.py` 在新进程里对原版 R0 路径检查；两个模型都 `pass: true`）**

- `train_model` 的键、形状、dtype 与新建的 R0 `state_dict()` **完全一致**（ViT-L 388 个键，ViT-B 208 个）；export 字典的键/形状既等于 R0 的 `get_export_state_dict()` 也等于原始训练导出 `/root/autodl-tmp/vit{l,b}-best.pth`；`model_class_path` / `model_init_args` 相同；导出权重全部有限；
- checkpoint 顶层键 = Lightly 的 `{train_model, optimizer, scheduler, train_dataloader, step, model_class_path, model_init_args, license_info}`；优化器的 param_groups（ViT-L 235 组 / ViT-B 127 组）、state 键（`step/exp_avg/exp_avg_sq`）和形状与原版 AdamW 一步之后的结构一致，调度器 state 键一致；
- 权重确实在动：20 步后 19 个张量变化（最大 |Δ|=2.7e-5 / 3.8e-5，head 参数；骨干在 `TwoStageWarmupPolySchedule` 的 head 预热阶段 lr 为 0，所以没变）；
- `lightly_train.load_model(export, device="cuda")` 加载原版 `DINOv3EoMTSemanticSegmentation` 并对一张 valid 图 `predict` 成功（输出 [1008,1008] int32）；
- **在原版（R0、未打补丁）路径上 resume**：`load_state_dict(strict=True)` 全部匹配，优化器和调度器 state 载入，一步原版 eager 训练 loss 有限（26.9 / 26.2）。
- **发现并绕过 round15 的一个 bug**：`install_checkpoint_compat` 把 `scheduler.state_dict/load_state_dict` 换成局部闭包，而 torch `LRScheduler.state_dict()` 返回整个实例 `__dict__`（含这两个闭包），所以 `torch.save` 报 `AttributeError: Can't pickle local object 'install_checkpoint_compat.<locals>.sched_state'`，**带 r15 的栈无法存 checkpoint**（r15/r16 的测试只做了字典级往返，没有真正 `torch.save`）。本轮不能改 round15，`run_final.py::fix_scheduler_state` 在返回的字典里过滤掉这两个键（已验证存/载/resume 通过）。修法见 §7 契约变更请求。

**5.6 数据集目标数（`tests/scan_dataset.py`，`profile/dataset_targets.json`）**

| 视角 | 结果 |
|---|---|
| COCO 标注（train 3672 张 / valid 477 张） | 每张图 0 或 1 个实例：train 154 张 0 个 + 3518 张 1 个（只有类别 `head`），valid 38 + 439；`iscrowd` 0 |
| 训练实际用的语义 mask PNG | 每张图的不同像素值个数：1（只有背景，154 张）或 2（背景 + head，3518 张）；全集只有值 0 和 1 |
| 真实训练 pipeline（Lightly 数据集 + EoMT 变换含随机裁剪/缩放/增强 + collate，3 遍共 11016 个样本）的 `len(binary_masks)` | 每图 1 个：721 / 731 / 849（三遍），2 个：2951 / 2941 / 2823；**最大 2**，这 11016 个样本里没有出现 0 个目标的（源码对没有有效类的 mask 最多重采样 20 次，找不到时仍返回最后一次，所以理论上可能有 0 个目标；r3 padded 路径对 0 目标的行为本轮没有单独测）；label 0（背景）在每个样本里都有 |

**结论：`max_targets=2` 刚好覆盖数据集（语义分割下每图目标数 = 出现的类别数 ≤ 类别数 2），r3 的 4 个上限有余量。** 如果以后加类别（K 个前景类 → 每图最多 K+1 个目标），要把 `StepGraphRunner(max_targets=…)`（`run_final.py --max-targets`）设成 K+1，且 ≤4（r3 padded loss 上限）；超出容量的批会退回 eager 步。**回退实测**（`tests/test_fallback.py`，一张图 4 个目标）：ViT-B 回退成功（graph 驻留 reserved 12.5 GB，回退期间峰值 22.6 GB，loss 有限，回退后同签名继续回放、无重捕获）；**ViT-L 也成功**（graph 驻留 16.5 GB，峰值 26.2 GB < 31.4 GB，loss 有限，无重捕获）——round16 担心的 ViT-L OOM 在 r14 把显存降下来之后不再出现，但 26.2 GB 离上限只剩 5 GB，更大的 B 或更多的目标就会 OOM。


## 6. 静态分析

本轮没有新的自研 kernel，没有跑 `kernel_static.sh`。各 kernel 的静态分析在其所属轮次（r13 / r14 / r15 / r17）。roofline 估计见 §7.2。

## 7. 结论、失败项、契约变更请求、roofline 与下一步

### 7.1 结论

- **FINAL（`P2R14R15`）在 graph 下实测 ViT-L 68.87 img/s（101.8 ms）、ViT-B 161.94 img/s（86.6 ms），对同机重测的 R0（16.31 / 37.77）是 4.22× / 4.29×，对"graph 之前的最佳"P2+r15 graph（59.5 / 123.1）是 1.157× / 1.315×**；显存 28.8 / 29.2 GB → 16.5 / 12.5 GB（reserved）。r14、r15、r16、r17 组合没有干涉：kernel 类别表里 r17 / r15 的类别与 r14 之前完全相同，r14 的收益在 graph 之下几乎 1:1 变成墙钟。
- **门**：Gate A 全过；S 8/8 过（EMA 偏差 ≤2.3e-3）；graph 对 eager 等价；导出/resume/`load_model` 全过；B-lite **ViT-L 过、ViT-B 在"中位数 ≥0.995"差 0.0005（0.9945）**，原因是 r17 的 MXFP8（`fc` 档 0.9947 也差 0.0003，去掉 r14 或 r15 不改变，去掉 MXFP8 才过），其余指标有大余量。
- **端到端**：loader 不是瓶颈（e2e 稳态是 GPU-only 的 98%），一个 epoch 的 val + checkpoint 反而占 29–33%。
- **数据集每图最多 2 个目标，`max_targets=2` 够用**，ViT-L / ViT-B 的溢出回退在最终栈上都不会 OOM。

### 7.2 roofline 式的剩余估计（graph，mid；`profile/roofline_final.txt`；下限是由 FLOP / 字节数和给定峰值算的**解析估计**，不是更好 kernel 的实测）

峰值假设：BF16 HMMA(fp32 累加) 230 TFLOPS，MXFP8 block-scaled 600 TFLOPS（r17 整 GEMM 实测 314–638），DRAM 1.55 TB/s。

| 部分 | ViT-L 实测 ms | ViT-L 下限 ms | ViT-B 实测 ms | ViT-B 下限 ms | 依据 / 说明 |
|---|---:|---:|---:|---:|---|
| 骨干 GEMM（MXFP8 + qkv 前向 bf16） | 29.15 | 24.97 | 17.17 | 14.21 | 13.2 / 7.5 TFLOP（其中 12.1 / 6.9 TFLOP 在 MXFP8，实测 MXFP8 GEMM 平均约 497 TFLOPS）；全部 bf16 是 57.4 / 32.7 ms，全部 MXFP8 是 22.0 / 12.5 ms |
| attention（flash fwd/bwd + RoPE） | 16.05 | 11.37 | 12.70 | 8.74 | 2.62 / 2.01 TFLOP 在 230 TFLOPS；RoPE（`_rotate`）1.27 ms 不在下限里 |
| head GEMM（ConvT×2 fwd/dgrad/wgrad、einsum、mask head，4 次 predict） | 19.01 | 17.14 | 21.57 | 19.74 | 3.94 / 4.54 TFLOP，**已贴着 bf16 限速**；若 ConvT 走 MXFP8，下限 6.6 / 7.6 ms |
| r14 融合 kernel（GELU+dwconv+LN2D） | 6.64 | – | 10.30 | – | DRAM 受限（r14：stage2 达到 DRAM 带宽的 72–78%） |
| 优化器（r15） | 7.72 | 7.32 | 2.25 | 2.02 | 36 B/float，已贴着带宽 |
| MXFP8 量化 | 5.07 | – | 4.10 | – | DRAM 受限，只能靠融合进生产者来消 |
| LN / 行 kernel | 5.89 | – | 5.04 | – | DRAM 受限 |
| 逐元素 / copy / 其它 | 6.91 | – | 7.75 | – | 见下 |
| loss / matcher / mask | 1.49 | – | 3.17 | – | |
| cuDNN conv（patch embed） | 0.71 | – | 1.09 | – | |
| **合计（kernel）** | **98.64** | | **85.14** | | 窗口 100.32 / 86.34 ms |

有解析下限的四部分（骨干 GEMM、attention、head GEMM、优化器）合计实测 71.9 / 53.7 ms，下限 60.8 / 44.7 ms，**差 11.1 / 9.0 ms**（骨干 GEMM 4.2 / 3.0，attention 4.7 / 4.0，head GEMM 1.9 / 1.8，优化器 0.4 / 0.2）；没有下限的 DRAM 受限部分（r14 融合 kernel、量化、LN、逐元素、loss、conv）合计 26.7 / 31.5 ms。所以**在不改精度的前提下，kernel 时间最多还能再省约 11 ms（11%）加上融合带来的一部分 DRAM 受限时间；再往下的大头只有 head 的 ConvT 改 MXFP8（下限 −10.5 / −12.1 ms，但要新门）和 FP8 attention**。逐元素类里最大的单项（`mid`，ViT-L）：434 个 `CUDAFunctor_add<float>`（3.8 µs/个，共 1.65 ms，推断为梯度累加，未逐调用核对）、`CatArrayBatchedCopy` 31 次共 1.05 ms、`reduce_kernel` 51 次 0.60 ms、bf16 cast / copy 约 1.0 ms、`FillFunctor` 25 次 0.19 ms、patch embed 的 `precomputed_convolve_sgemm` 0.38 ms。

### 7.3 失败的尝试 / 发现的问题

1. **r15 的 checkpoint bug**（§5.5）：带 flat 优化器的栈在 round15/16 里从没真正存过 checkpoint；本轮第一次 `torch.save` 就失败。已在 `run_final.py` 绕过，**round15 需要修**（契约变更请求 1）。
2. **ViT-B 的 B-lite 中位数 0.9945**（§5.2）：r17 MXFP8 的固有噪声，不是组合造成的；是否放宽需要协调者决定。
3. 第一次 `check_export.py` 在 ViT-L 上 OOM：R0（fp32，B=7）峰值 28.8 GB，再加上 resume 载入的优化器 state 就超过 31.4 GB；改用 2 张图的 mini-batch 做 R0 的两步（批大小对这项检查无关）。
4. 多 seed / 多进程交替使用 GPU 时，`assert_gpu_exclusive` 会被"另一个进程在两次取锁之间留着空闲 CUDA 上下文"打断：本轮 `maxbatch B=15` 因此失败（RuntimeError，不是 OOM），已重跑（契约变更请求 3，沿用 r16/r17）。
5. dataset 扫描的 loader 在默认 `ulimit -n 1024` 下，48 个 workers 就报 `received 0 items of ancdata`；`run_final.py` 的 docstring 和 README 都写了 `ulimit -n 65536`。
6. 没有做：epoch 级训练和 mIoU / head IoU（Level D）对照；round18（val 快路径、异步 checkpoint、loader 的 GPU 上 normalize）；round12 融合 attention 的集成；PDL；冷启动（QuACK / CuTe / Triton JIT 缓存被清空时）的首次编译时间；`run_final.py` 的逐 epoch val/ckpt 循环（只跑了"N 步 + 末尾一次 val + 一次 checkpoint"）；r15 的 `check_real_flat.py`（graph_equiv / gate S / 导出检查已覆盖它的内容）。

### 7.4 契约变更请求

1. **round15**：`install_checkpoint_compat` 的 `sched_state` / `sched_load` 不要作为实例属性挂在 scheduler 上（或在返回的 `state_dict()` 里删掉 `state_dict` / `load_state_dict` 两个键，或用 `types.MethodType` + 模块级函数）；并加一个真正 `torch.save` 的测试。
2. **`common/config.py`**：发布 `BEST_STACK_V2 = P2R14R15_STACK`、`BEST_STACK_V2_KWARGS = P2R14R15_PATCH_KWARGS`（当前为空），并注明运行器是 `round16_full_graph.impl.graph_step.StepGraphRunner`（`RngTape("off")`，`max_targets=2`），入口 `round19_final/run_final.py`；ViT-B 的 B 门裁定后再决定是否同时发布 `FINAL_fc`（`r17:fc`）。
3. **`common/bench.assert_gpu_exclusive`**：只看锁持有者 / 空闲上下文（沿用 r16 / r17 的请求），否则多进程队列会误报。
4. **CONTRACTS §7 / Amendment 2**：对 MXFP8 栈，"中位数 ≥0.995"在 ViT-B 上没有任何 MXFP8 策略能稳定通过（`fc` 0.9947、`fc+proj+qkv_dw` 0.9924，12 批）；建议改为"中位数 ≥0.99 且最小 ≥0.98 且 S 门通过"，或明确 ViT-B 用 bf16 / `fc` 档。

### 7.5 下一步（按优先级；估计的 ms 是 ViT-L（ViT-B）每步，依据写在括号里，**均为估计，未实现**）

1. **val 走 BF16 + graph 快路径，checkpoint 异步（round18）**：不是 kernel，但 val + checkpoint 占 epoch 的 29%（33%）：val 12.2 s（8.7 s）是未打补丁的 fp32 路径，checkpoint 同步 9.4 s（2.7 s）。val 换 BF16（TB1 对 R0 的 fp32→bf16 倍数约 2.5×）估计省 5–7 s（4–5 s），checkpoint 放后台线程省 9 s（2.7 s）：**每个 epoch 省约 14–16 s（6–8 s），即 epoch 缩短 19–21%（17–23%）**，比下面任何 kernel 项都大，风险低。
2. **head 的 ConvT 用 MXFP8（r14 已有 `mx8_bwd` 变体，dgrad/wgrad 走 MXFP8、前向保持 bf16）：−4 ms（−5 ms）**。依据：ConvT 的 dgrad + wgrad 现在约 10.5 ms（ViT-L，bf16 在限速上限），MXFP8 按 ~450 TFLOPS 估计约 5.2 ms，再扣 ~1 ms 量化；r14 在 eager 上测到 +4.0% / +2.7%。**需要补门**：现在只有 ViT-L 3 批的 B（`mx8_bwd` 3/3），ViT-B、≥8 批规则、S、graph 捕获都没做；`mx8` 全 GEMM（前向也量化）的 mask logits 相对误差 5–6e-2，不要用。
3. **量化融合进生产者（fc1 的 GELU epilogue 直接出 h 的行副本、LN 输出、`γ⊙dX`、权重量化接到 r15 的 `register_weight_writer`）：−2 到 −3 ms（−1.5 到 −2.5 ms）**。依据：量化类现在 5.07 ms（4.10 ms），r17 估计合计 2–3 ms；现在 graph 之下才兑现。
4. **wgrad 直接累加进 flat grad buffer，去掉梯度累加和清零写：−1.5 到 −2 ms（−1.2 到 −1.5 ms）**。依据：434（266）个 3.8 µs 的 `add<float>` kernel 共 1.65 ms（1.08 ms），`update_kernel` 里清零写 4 B/float ≈ 0.4 ms（0.1 ms）；需要 QuACK 的 `gemm_add` 做 beta=1 的 fp32 wgrad 输出，并改 r15 的 `adopt_grads` 约定（r15 §7 已提）。
5. **round12 融合 attention 集成（kernel 已写，没有实测数据，NOTES 是进行中版本）：−2 到 −3 ms（−1.5 到 −2.5 ms）**，上限 −4.7 ms（−4.0 ms）= 实测 16.05 ms 对 BF16 下限 11.37 ms。依据：128 行 tile 对 N=1029 浪费约 11–25% FLOPs，融合 RoPE（`_rotate` 1.27 ms）、packed 布局、joint 的 query bitmask；要新 kernel 的 A/B/S 门。
6. **小 kernel 清理：−1 到 −1.5 ms（−1 到 −1.5 ms）**：`CatArrayBatchedCopy`（31 次 1.05 ms）、bf16 cast / copy（约 1.0 ms）、`FillFunctor`（0.19 ms）、`reduce_kernel`（0.6 ms）改成预分配输出或并入相邻 kernel；ViT-B 另有 loss / matcher 3.17 ms（r3 的 batched torch 版，换成 CUDA 融合版可能省一半，未测）。
7. **FP8 attention（SageBwd / Spark 风格）：−3 到 −5 ms（−2.5 到 −4 ms）**，风险高（精度，且要先做 5），排最后。
8. **不用做**：加大 batch——ViT-L 在 B=7…17 吞吐都在 68.8–70.3 img/s（§4.5），B=17 只比 B=7 高 2.1%，显存却从 16.5 GB 涨到 30.4 GB；loader 调参——e2e 已是 GPU-only 的 98%（唯一要做的是 pin 保持开和 persistent workers）。

把上面 2–6 全做到的话，ViT-L 的 101.8 ms 大约能到 88–92 ms（约 76–80 img/s），再加 7 可到约 85–88 ms；这是把各项估计加起来的结果，不是承诺——每一项都要自己的门，且收益会相互重叠。
