# Round20（R20a）：FP4 可行性探针，速度矩阵 + 量化开销 + 真实张量误差表 + 每步节省估计（负责：Sonnet，第三阶段）

> 全部数字都在 `infer-5090`（RTX 5090）上实测，CUDA graph 回放、同一进程内候选交替运行、3 轮取中位数、持双锁；没测的写 null / not measured。**本文件只覆盖 R20a**，R20b（仿真门扫描）是下一个任务，这里没有任何 B/S 门结果，也没有整步实测。
> 参考基线 FINAL = `BEST_STACK_V2`（P2R14R15）+ r16 `StepGraphRunner`：ViT-L 101.8 ms/步，ViT-B 86.6 ms/步（PLAN_v3 §0，同机）。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 做了什么 | (1) `impl/fp4_ref.py`：nv4 / mx4 的纯 torch 参考（RN、SR、RHT、2D/1D 权重 scale、token 轴填充、128×4 blocked scale、dequant），自测通过；(2) 60 个真实 GEMM 形状 × 约 12 个候选的速度矩阵（QuACK nv4/mx4 含 epilogue、cuBLASLt、Spark、QuTLASS、r17 MXFP8、bf16）；(3) 现成量化 kernel 的实测开销 + 带宽下限；(4) 两个模型各 16 个真实张量 × 3 种 GEMM × 7–11 种方案的相对 L2 误差表；(5) 每步节省估计与 go/no-go 速度条件 |
| fp4_ref 自测 | nv4 RN 与 Spark 的 CUDA 量化器**逐字节一致**（3 类数据：高斯 / 尖峰 / 微小块，数据字节与 scale 字节 100%）；mx4（floor 模式）与 QuACK `to_mxfp4`、Spark 逐字节一致；fp4_ref 的操作数喂 QuACK FP4 GEMM（fp32 输出）对 fp64 反量化乘积相对 L2 = 0 – 6e-9，喂 cuBLASLt NVFP4 的输出恰好等于 bf16(参考)（相对 L2 1.66e-3 = bf16 舍入下限），即布局和 scale 约定与 GEMM 消费方式一致（门槛 ≤2e-3）；SR 在不饱和元素上无偏（偏差/期望噪声 = 1.004 nv4、0.997 mx4） |
| GEMM（最重要的数） | QuACK FP4 在训练形状上 0.86–1.32 PFLOPS（大形状；小输出和带残差 epilogue 的形状更低）：**head dgrad+wgrad nv4 对 bf16 5.15×（L）/ 5.07×（B）**；**骨干 dgrad+wgrad nv4 对 MXFP8 1.89× / 1.94×**；fc1/fc2 前向（带 epilogue）1.72× / 1.47×；qkv 前向 MXFP8 对 bf16 2.65× / 2.89×；head 前向 MXFP8 对 bf16 2.70× / 2.39×。mx4 比 nv4 只快 3–7% |
| 量化器 | 现成的行方向 FP4 量化器（Spark / QuTLASS）在 HBM 读下限上；**所有现成的转置副本路径比下限慢 6–10×**，直接用它们整步会亏 23–32 ms。**必须自写一次读入同时出行副本和转置副本的 FP4 量化器**（R21）。按字节比例估计的融合量化器开销：骨干 dgrad+wgrad 项约 0.3 ms（ViT-L）/ 0.65 ms（B）（amax 由生产者给出时），需要单独 amax 遍历时 3.0 / 3.4 ms |
| 误差 | 单 GEMM 相对 L2 中位数（骨干，ViT-L）：nv4 前向 0.114 / dgrad 0.135 / wgrad 0.080；mx8 0.029 / 0.034 / 0.022；mx4 约为 nv4 的 1.5–1.7 倍。**发现**：block 0 的 fc1/fc2 有 massive activation（cls/register token 行，X 最大值 1992 对 rms 9），FP4 wgrad 误差 0.26–0.60，RHT 使它恶化到 9–25（两次捕获之间 dY 的位不同，结果在这个范围内波动）；把 cls+register（joint 块还有 query）token 排除出 FP4 token 规约、用小 bf16 GEMM 精确补上，误差降到 0.04–0.06 |
| 每步节省估计（融合量化器、amax 由生产者给出） | S1（head dgrad+wgrad nv4 + 骨干 dgrad+wgrad nv4）：ViT-L 省 15.1 ms（14.8%），ViT-B 省 11.3 ms（13.1%）；S2（再加 fc 前向 nv4、qkv 前向 MXFP8、head 前向 MXFP8）：24.2 ms（23.8%）/ 18.2 ms（21.0%）。**这是 GEMM 实测 + 量化器估计，没有任何精度门** |
| go/no-go（只套速度和节省两条，仿真门是 R20b） | 满足：① head dw nv4、③ 骨干 dw nv4、②a head 前向 MXFP8。**不满足 ≥3 ms 或 ≥3%（单独）**：④ fc 前向 nv4（L 2.92 ms / 2.9%，B 1.93 ms / 2.2%，且 ViT-B 速度 1.47× < 1.5×）、⑤a qkv 前向 MXFP8（L 2.88 ms / 2.8%，B 1.92 ms / 2.2%）；两个 ViT-L 项贴着 3 ms / 3% 的线，两次完整重跑之间在 2.7–2.9% 波动。ViT-B 的③在"需要单独 amax 遍历"时只省 1.54 ms（1.8%），所以 nv4 的 amax 必须由生产者给出 |
| 推荐 | 格式 nv4（mx4 作对照臂）；后端 QuACK（epilogue 全部可用，nv4 的每张量 scale 作为设备标量参数）；R20b 候选见 §7 |
| 契约变更请求 | 见 §7 |

## 1. 目标与假设

PLAN_v3 R20a：用实测数字替换 PLAN_v3 §2 的预估，回答 (a) FP4 GEMM 在我们的形状上到底有多快（后端对比），(b) 量化器要多少钱，(c) 单 GEMM 误差有多大、哪些层危险，(d) 每步能省多少，并给 R20b 筛选候选。假设与核对：

| 假设 | 实测 |
|---|---|
| FP4 实际 GEMM 约 1.2–1.4 PFLOPS（Spark / cuBLASLt 报告） | QuACK 大形状 1.1–1.26 PFLOPS，小输出形状（proj 的 N=K=768/1024）和带 epilogue 的 fp32 残差流（proj/fc2 前向）受带宽限制，只有 0.25–0.8 PFLOPS |
| 骨干 MXFP8 → FP4 ≥1.5× | dgrad+wgrad 合计 1.89× / 1.94× 成立；fc 前向 1.72×（L）成立、1.47×（B）不成立 |
| head bf16 → FP4 ≥3× | 4.9–5.7× 成立（bf16 在 217 TFLOPS 限速上，FP4 1.1 PFLOPS） |
| 量化器开销可忽略 | **不成立**：必须自写融合量化器，否则亏 |
| 4 bit 操作数只能 K-major，所以三个 GEMM 都写成 TN | 已验证：QuACK 接受 TN 的 FP4 操作数；wgrad 用 `[N,Mpad]×[K,Mpad]ᵀ`，Mpad=128 的倍数（QuACK 通过，Spark 要 256 的倍数） |

## 2. 改动（全部在 `train-opt/round20_fp4_probe/`，没有改任何旧 round / common / PLAN / CONTRACTS）

| 文件 | 内容 |
|---|---|
| `impl/fp4_ref.py` | 参考实现。`quantize(x, fmt, row, col, rht_row, rht_col, sr, gen, u_row, u_col)` → `FP4Tensor`（`q/sf/ts` 行副本、`qt/sft/tst` 转置副本）；`quantize_weight(w, fmt, two_d, rht_row, rht_col)`（W 的 `q` 沿 K、`qt` 为 Wᵀ 沿 N，2D 时二者逐元素一致）；`quantize_matrix`、`quantize_matrix_2d`、`dequant_copy/row/col`、`rht_apply`（FWHT 蝶形 h=1,2,4,8 → ×1/4 → ×符号向量）、`rht_signs`、`make_gen`、`to_blocked/from_blocked`。**约定**见下 |
| `impl/fp4_cands.py` | 候选包装：`quack_fp4`、`scaled_mm_fp4`（cuBLASLt）、`spark_fp4`，`operand()`（QuACK `BlockScaledOperand`）、`epilogues()`（r13） |
| `impl/fp4_epi.py` | NVFP4 版 r13 epilogue：`bias_ts / res_ts / gelu_ts / dgelu_ts / plain_ts`，每张量 scale 以设备标量 `ts = ts_a·ts_b` 折进 epilogue（无 host 同步）；mx4 无每张量 scale，直接用 r13 的 epilogue |
| `impl/spark_fp4_ext.py` | Spark `fp4gemm / fp4quant`（MIT，a401bfa9e363）的最小 torch 绑定，原地编译 third_party（不复制代码），build dir 在 tmp |
| `impl/qutlass_ext.py` | QuTLASS（Apache-2.0，e74319e3405c）对着 CUTLASS 4.2 头文件的限时（15 分钟）构建，实际约 3 分钟编完，GEMM 与量化器可用 |
| `tests/` | `test_fp4_ref.py`（自测）、`bench_fp4_gemm.py`（矩阵）、`bench_qutlass_gemm.py`、`bench_fp4_quant.py`、`fp4_error_table.py`、`debug_wgrad.py`、`debug_prefix.py`、`summarize_*.py`、`make_*.py`、`queue_*.sh`、`probe_quack.py`、`run.py` |
| `profile/` | `fp4_ref_selftest.json`、`fp4_gemm_matrix.json`（合并）+ 分文件、`fp4_gemm_qutlass.json`、`fp4_quant_cost.json`、`fp4_error_table.json`（+分模型）、`per_step_estimate.json`、`debug_*.json`、`tables_r20a.md` |

**fp4_ref 的约定（R21/R22 gate A 逐位参照；与 CONTRACTS 11.2 的差别在下面标出）**

- nv4：`ts = amax/(6·448)`（fp32 除法，amax=0 → ts=1）；`s_blk = e4m3_RN(min((amax_blk/6)/ts, 448))`（**运算顺序是先除 6 再除 ts**，与 QuACK、Spark 相同；契约写的 `amax_blk/(6·ts)` 数学上相同，位上可能差 1 ulp）；`v = x/(e4m3(s_blk)·ts)`（fp32 除法，Spark 的算法）→ e2m1 RN-even，饱和 ±6，符号位取 `signbit`；scale 为 0 的块（全 0，或 scale 下溢）数据全 0、scale 字节 0。
- mx4：`byte = clamp(biased_exp(amax_blk/6) + (mantissa != 0), max=254)`（RCEIL，不会饱和）；`v = x·2^-(byte-127)`（精确）；**全 0 块约定：scale 字节 0、数据 0**（与 r17 的 pad 行一致）；`mx_round="floor"` 仅用于和 QuACK `to_mxfp4`、Spark 对拍。
- RHT：`y = signs ⊙ (FWHT16(x)/4)`（先 FWHT 后乘符号，符号向量 16 个 ±1，种子 20251003，`rht_signs()` 给出：`[1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1]`）；两个操作数沿收缩轴用同一符号向量；FWHT 用显式蝶形而不是矩阵乘，保证 fp32 下可复现。
- SR：`u ∈ [0,1)`，取上邻点当且仅当 `u < (|v|-lo)/(hi-lo)`；`u` 来自专用 `torch.Generator`（`make_gen`），或由调用方传入 `u_row/u_col`（kernel 用同一 Philox 流时逐位对拍）；默认 CPU/CUDA 生成器未被触碰（自测 e 项）。行副本先抽、转置副本后抽。

## 3. nsys 分析过程

**本任务没有 nsys**：R20a 是算子级速度矩阵（CUDA graph 回放 × 5 次嵌套 × 10 次重放，事件计时）和量化器 / 误差测量，没有整步改动，所以没有 NVTX 表、整步 kernel 表和 gap 分析（R21/R22 才有）。作为对照，用矩阵给出的"FINAL 当前"各 GEMM 时间乘以每步次数，和 r19 §3.3 的 kernel 表交叉核对（同一台机器）：

| 项（ms/步） | 矩阵加权和 L / B | r19 §3.3 L / B |
|---|---:|---:|
| 骨干 MXFP8 GEMM（fwd proj/fc + dgrad + wgrad） | 24.8 / 15.2 | 24.4 / 14.5 |
| qkv 前向 bf16 | 4.84 / 3.06 | 4.8 / 2.7 |
| head GEMM（bf16，fwd+dgrad+wgrad） | 16.4 / 17.8 | 19.0 / 21.6（含 ConvT 以外的 head GEMM，如 einsum，矩阵未含） |

**命令**（每段持双锁，≤10 分钟，后台 + 日志轮询）：

```bash
python tests/test_fp4_ref.py                                   # 自测 -> profile/fp4_ref_selftest.json
tests/queue_matrix.sh   # vitl16 bb:7203 / vitb16 head / vitb16 bb:14406 / vitl16 bb:7903 / vitb16 bb:15806 (+ vitl16 head 单独跑)
python tests/bench_qutlass_gemm.py ; python tests/bench_fp4_quant.py
python tests/fp4_error_table.py vitl16 --save ; python tests/fp4_error_table.py vitb16 ; python tests/debug_wgrad.py ; python tests/debug_prefix.py
python tests/summarize_r20a.py ; python tests/summarize_error.py ; python tests/make_tables.py ; python tests/make_results20a.py
```

## 4. 实测结果

### 4.1 fp4_ref 自测（`profile/fp4_ref_selftest.json`）

| 项 | 结果 |
|---|---|
| (a) nv4 RN 对 Spark CUDA 量化器 | 高斯 / 尖峰 scale-8 / 微小块（1e-5 倍，含全 0 块）三类，2048×1024：数据字节、scale 字节 100% 一致（0 个差异） |
| (a) nv4 RN 对 QuACK `to_nvfp4` | 高斯：scale 100%，数据字节 99.907%（978 个差，原因：QuACK 用 `(1/ts)/s` 的倒数乘法，fp4_ref 用除法）；尖峰：100%；微小块：scale 92.2%、数据 93.7% 一致（QuACK 把 block scale 下限夹到 e4m3 最小正规数 2^-6，Spark / fp4_ref 允许 subnormal，scale 下溢为 0 时整块为 0） |
| (a) mx4 floor 模式对 QuACK / Spark | 三类数据全部逐字节一致；RCEIL 性质：`amax/scale ≤ 6`（无饱和）且 `scale/2 < amax/6`（紧）成立 |
| (a) 布局 | `to_blocked` 与 QuACK `pack_scale_2d_to_blocked_contig` 逐字节相同；往返无损 |
| (b) GEMM 消费 | 形状 fwd 7203×4096×1024、dgrad（7203×1024，收缩 4096）、wgrad（4096×1024，收缩 Mpad=7296，dY 用 SR + 两个操作数沿 token 轴 RHT）。QuACK（fp32 输出）相对 L2：nv4 0 / 0 / 6.3e-9，mx4 0 / 0 / 0；QuACK 操作数 `.dequantize()` 与 fp4_ref 反量化逐元素相等；cuBLASLt NVFP4（bf16 输出）相对 L2 = bf16 舍入下限（1.66e-3，输出等于 bf16(参考)）；Spark（bf16 输出）可用（变体需显式指定）。**cuBLASLt 的 MXFP4 在 sm_120 上不可用**（`NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300`） |
| (c) SR 无偏性 | 64×1024 高斯，2000 次抽样：不饱和元素的 rms 偏差 / rms(x) = 0.003126（nv4）/ 0.003640（mx4），无偏时的期望噪声 0.003114 / 0.003651，比值 1.004 / 0.997；RN 的 rms 误差 0.095 / 0.114。**nv4 有 3.4% 的元素饱和**（e4m3 scale 向下取整使 \|x\|/denom > 6），这些元素 SR 也无法无偏（rms 截断偏差 = 5.5% rms(x)）；mx4 RCEIL 无饱和 |
| (d) RHT | 点积保持相对误差 1.1e-9，范数保持 0；FWHT 蝶形等于 Hadamard 矩阵乘 |
| (f) 2D 权重 | W 与 Wᵀ 逐元素相同（nv4、mx4 都是），ts 相同；1D 时不同。W 的量化相对 L2：nv4 2D 0.111 / 1D 0.095；mx4 2D 0.130 / 1D 0.117 |
| (g) 填充 | 转置副本 Rpad=7296，填充 token 全 0，blocked scale 形状 nv4 `(2,114,32,4,4)`（对 256 通道）、mx4 `(2,57,32,4,4)` |
| (e) 默认 RNG | 默认 CPU / CUDA 生成器状态在 SR + RHT + 权重量化之后不变 |

### 4.2 GEMM 候选矩阵（`profile/fp4_gemm_matrix.json`；形状 = 真实训练形状）

测法：每个候选捕获进自己的 CUDA graph（×5 次嵌套），同一进程内按顺序交替（奇数轮倒序）重放，10 次重放/轮，3 轮取中位数。"FINAL 当前" = r17 MXFP8 原生布局 + r13 epilogue（proj/fc 前向：γ+fp32 残差 / GELU；fc2 dgrad：GELU'+db；wgrad fp32 输出）、qkv 前向 bf16 + bias、head 为 bf16（wgrad 取 torch/QuACK 中较快者）。FP4 候选的"可用"列：前向、fc2 dgrad 取带 epilogue 的 QuACK 版本（`nv4_quack_epi`），其余取 plain / fp32 输出的最快者。


#### vitl16: per (class, linear, kind) GEMM time (us, CUDA-graph replay, median of 3) and TFLOPS

| case | M x N x K | FINAL now | best NVFP4 (usable) | best MXFP4 | NVFP4 vs now | MXFP4 vs now | now = |
|---|---|---:|---:|---:|---:|---:|---|
| plain fc1 fwd | 7203x4096x1024 | 123 (493 TF) | 81 (743 TF) | 80 (756 TF) | 1.51x | 1.53x | mx8 epi |
| plain fc1 dgrad | 7203x4096x1024 | 95 (639 TF) | 48 (1251 TF) | 46 (1300 TF) | 1.96x | 2.03x | mx8 |
| plain fc1 wgrad | 7203x4096x1024 | 106 (569 TF) | 56 (1073 TF) | 54 (1122 TF) | 1.89x | 1.97x | mx8 native |
| plain fc2 fwd | 7203x1024x4096 | 146 (414 TF) | 75 (802 TF) | 75 (809 TF) | 1.94x | 1.95x | mx8 epi |
| plain fc2 dgrad | 7203x1024x4096 | 156 (387 TF) | 89 (680 TF) | 86 (701 TF) | 1.76x | 1.81x | mx8 |
| plain fc2 wgrad | 7203x1024x4096 | 106 (569 TF) | 56 (1079 TF) | 54 (1124 TF) | 1.90x | 1.97x | mx8 native |
| plain proj fwd | 7203x1024x1024 | 41 (369 TF) | 26 (583 TF) | 26 (582 TF) | 1.58x | 1.58x | mx8 epi |
| plain proj dgrad | 7203x1024x1024 | 26 (574 TF) | 16 (967 TF) | 14 (1050 TF) | 1.68x | 1.83x | mx8 |
| plain proj wgrad | 7203x1024x1024 | 47 (323 TF) | 21 (709 TF) | 20 (765 TF) | 2.20x | 2.37x | mx8 native |
| plain qkv fwd | 7203x3072x1024 | 198 (228 TF) | 39 (1175 TF) | 39 (1165 TF) | 5.14x | 5.10x | bf16 epi |
| plain qkv dgrad | 7203x3072x1024 | 71 (637 TF) | 37 (1229 TF) | 35 (1283 TF) | 1.93x | 2.02x | mx8 |
| plain qkv wgrad | 7203x3072x1024 | 104 (437 TF) | 52 (873 TF) | 52 (872 TF) | 2.00x | 2.00x | mx8 native |
| head 1 fwd | 7168x4096x1024 | 277 (217 TF) | 55 (1094 TF) | 53 (1131 TF) | 5.05x | 5.22x | bf16 (head) |
| head 1 dgrad | 7168x4096x1024 | 282 (213 TF) | 49 (1217 TF) | 47 (1267 TF) | 5.70x | 5.94x | bf16 (head) |
| head 1 wgrad | 7168x4096x1024 | 298 (202 TF) | 55 (1090 TF) | 53 (1140 TF) | 5.40x | 5.65x | bf16 (head) |
| head 2 fwd | 28672x4096x1024 | 1064 (226 TF) | 213 (1127 TF) | 215 (1119 TF) | 4.99x | 4.95x | bf16 (head) |
| head 2 dgrad | 28672x4096x1024 | 1055 (228 TF) | 216 (1112 TF) | 207 (1162 TF) | 4.88x | 5.10x | bf16 (head) |
| head 2 wgrad | 28672x4096x1024 | 1120 (215 TF) | 214 (1122 TF) | 207 (1164 TF) | 5.22x | 5.42x | bf16 (head) |

(mx4 的带 epilogue 版本也全部是直接实测（带 * 的才是推导值）；head 在 FINAL 里是 bf16，head 的 fwd/dgrad 没有 epilogue。)

Plain NVFP4 backend sums over the 30 cases of vitl16 (bf16 out; wgrad = bf16-out variants): quack 1.85 ms, cublaslt 1.99 ms, spark 2.21 ms

#### vitb16: per (class, linear, kind) GEMM time (us, CUDA-graph replay, median of 3) and TFLOPS

| case | M x N x K | FINAL now | best NVFP4 (usable) | best MXFP4 | NVFP4 vs now | MXFP4 vs now | now = |
|---|---|---:|---:|---:|---:|---:|---|
| plain fc1 fwd | 14406x3072x768 | 153 (444 TF) | 123 (553 TF) | 123 (554 TF) | 1.24x | 1.25x | mx8 epi |
| plain fc1 dgrad | 14406x3072x768 | 106 (638 TF) | 52 (1304 TF) | 50 (1349 TF) | 2.04x | 2.11x | mx8 |
| plain fc1 wgrad | 14406x3072x768 | 117 (580 TF) | 57 (1195 TF) | 55 (1234 TF) | 2.06x | 2.13x | mx8 native |
| plain fc2 fwd | 14406x768x3072 | 169 (401 TF) | 99 (687 TF) | 98 (691 TF) | 1.71x | 1.72x | mx8 epi |
| plain fc2 dgrad | 14406x768x3072 | 196 (346 TF) | 132 (515 TF) | 132 (517 TF) | 1.49x | 1.49x | mx8 |
| plain fc2 wgrad | 14406x768x3072 | 118 (576 TF) | 57 (1193 TF) | 53 (1284 TF) | 2.07x | 2.23x | mx8 native |
| plain proj fwd | 14406x768x768 | 75 (225 TF) | 68 (251 TF) | 65 (260 TF) | 1.12x | 1.15x | mx8 epi |
| plain proj dgrad | 14406x768x768 | 28 (598 TF) | 18 (940 TF) | 17 (1011 TF) | 1.57x | 1.69x | mx8 |
| plain proj wgrad | 14406x768x768 | 89 (191 TF) | 25 (690 TF) | 23 (726 TF) | 3.62x | 3.81x | mx8 native |
| plain qkv fwd | 14406x2304x768 | 250 (204 TF) | 46 (1097 TF) | 47 (1096 TF) | 5.38x | 5.38x | bf16 epi |
| plain qkv dgrad | 14406x2304x768 | 79 (645 TF) | 39 (1291 TF) | 38 (1350 TF) | 2.00x | 2.09x | mx8 |
| plain qkv wgrad | 14406x2304x768 | 96 (532 TF) | 49 (1044 TF) | 47 (1091 TF) | 1.96x | 2.05x | mx8 native |
| head 1 fwd | 14336x3072x768 | 307 (221 TF) | 68 (1000 TF) | 60 (1134 TF) | 4.53x | 5.14x | bf16 (head) |
| head 1 dgrad | 14336x3072x768 | 304 (223 TF) | 50 (1353 TF) | 49 (1391 TF) | 6.08x | 6.24x | bf16 (head) |
| head 1 wgrad | 14336x3072x768 | 323 (210 TF) | 56 (1213 TF) | 53 (1279 TF) | 5.79x | 6.10x | bf16 (head) |
| head 2 fwd | 57344x3072x768 | 1134 (239 TF) | 294 (922 TF) | 289 (936 TF) | 3.86x | 3.92x | bf16 (head) |
| head 2 dgrad | 57344x3072x768 | 1135 (238 TF) | 243 (1111 TF) | 236 (1145 TF) | 4.66x | 4.80x | bf16 (head) |
| head 2 wgrad | 57344x3072x768 | 1257 (215 TF) | 246 (1101 TF) | 236 (1146 TF) | 5.11x | 5.32x | bf16 (head) |

(mx4 的带 epilogue 版本也全部是直接实测（带 * 的才是推导值）；head 在 FINAL 里是 bf16，head 的 fwd/dgrad 没有 epilogue。)

Plain NVFP4 backend sums over the 30 cases of vitb16 (bf16 out; wgrad = bf16-out variants): quack 2.15 ms, cublaslt 2.27 ms, spark 2.94 ms

**后端对比（plain NVFP4，bf16 输出，30 个形状的和）**：ViT-L QuACK 1.85 ms / cuBLASLt 1.99 / Spark 2.21；ViT-B QuACK 2.15 / cuBLASLt 2.27 / Spark 2.94。QuTLASS（48 个不同形状，bf16 输出，同一进程内交替）：nv4 4.14 ms 对 QuACK 3.67 ms，mx4 4.07 对 3.48；只在部分大 K 的 wgrad / head 形状上快（最多约 12%），小形状慢 10–80%。QuACK 与 cuBLASLt 都不低于 Spark 的 80%（它们比 Spark 快），所以没有碰 CUTLASS 例 79。

要点：
- 带 fp32 残差流的 epilogue（proj / fc2 前向）是带宽受限（ViT-B proj 前向 68 µs ≈ 读写 110 MB），FP4 几乎不帮忙（1.12×）；fc1 前向（写 z 和 h 两份 bf16）和 fc2 dgrad（GELU'）的 epilogue 把 FP4 GEMM 从 53→82 µs、55→89 µs 拖慢。
- 小输出的 wgrad（proj：768/1024 × 768/1024 × 14464/7296）plain QuACK 慢（46 µs），split-K（`split_k=None`）后 21–25 µs，cuBLASLt 27 µs。
- mx4 plain 比 nv4 快 3–7%；带 epilogue 的差别在噪声内。

### 4.3 每步 GEMM 时间（µs/步 → ms/步，按次数加权：ViT-L 21 plain + 3 joint 块，ViT-B 9 + 3，head 每步 4 次 predict × ConvT1/ConvT2）

| ms/步 | 骨干 d（now→nv4） | 骨干 w | fc 前向 | qkv 前向（bf16→mx8→nv4） | head f（bf16→mx8→nv4） | head d | head w |
|---|---|---|---|---|---|---|---|
| vitl16 | 8.42→4.60 | 8.81→4.49 | 6.53→3.80 | 4.84→1.83→0.94 | 5.37→1.98→1.07 | 5.35→1.06 | 5.67→1.08 |
| vitb16 | 5.04→2.98 | 5.17→2.29 | 4.03→2.75 | 3.06→1.06→0.57 | 5.76→2.41→1.44 | 5.75→1.17 | 6.32→1.21 |

prefix-token 侧 GEMM（gather + bf16 GEMM fp32 输出 + 加进 FP4 的 wgrad 结果，4 个 Linear × 全部块，ms/步）：
- vitl16: 所有块只排除 5 个 prefix token：1.5396307182312012；普通块 5 个 + joint 块 105 个：1.8399283289909363
- vitb16: 所有块只排除 5 个 prefix token：0.635831041932106；普通块 5 个 + joint 块 105 个：0.9589209598302841

### 4.4 量化器开销（`profile/fp4_quant_cost.json`；HBM 拷贝实测 1528 GB/s）

L2 预热说明：输入 ≤ 约 50 MB 的小张量在回放中留在 96 MB L2 里，所以 mx8 的实测时间可以低于 HBM 下限。真实流水线里生产者 → 消费者同样有 L2 命中，所以本表不额外惩罚。

| 候选（µs） | ViT-L [7203,1024] | [7203,4096] | head dZ [28672,4096] |
|---|---:|---:|---:|
| r17 mx8 `quant_dual`（行+列，对照） | 7.7 | 61.9 | 313 |
| r17 mx8 行 | 5.5 | 14.4 | 226 |
| Spark nv4 行（给定 ts） | 9.7 | 30.9 | 183 |
| Spark nv4 行（设备 amax，2 次 launch） | 13.8 | 41.2 | 323 |
| Spark mx4 行（floor 配方） | 9.7 | 26.1 | 181 |
| QuTLASS fusedQuantizeMxAbsMax（行 + H32） | 11.0 | 35.0 | 178 |
| QuTLASS fusedQuantizeNvAbsMax（行 + H16） | 19.9 | 69.1 | 273 |
| QuACK `to_nvfp4_compiled`（行，无 scale 打包） | 21.4 | – | – |
| 转置副本：`x.t().contiguous()` | 71.8 | 269 | 1042 |
| 转置副本：Spark（转置+填充+量化） | 97.7 | 414 | 1525 |
| 转置副本：QuTLASS `backward_t_bf16`（转置+H32+mx4） | 59.4 | 233 | 916 |
| 一次读入双副本 nv4 的带宽下限（HBM） | 15.1 | 60.3（+amax 遍历 99.0） | 240（+394） |
| 同上 mx4 | 14.8 | 59.1 | 235 |

结论：行方向的 FP4 量化器已经在读带宽下限上（9.7 µs ≈ 读 14.7 MB 的时间）；**转置副本比下限慢 6–10×**，而且 r17 自己的双输出量化器（一次读入、转置写）才是正确的做法。所以 R21 必须写 FP4 版的双输出量化器。NVFP4 在量化之前需要整张量 amax：由生产者给出（`atomicMax`，免费）或单独遍历（多读 2 B/元素，下限 +65%）。

### 4.5 误差表（`profile/fp4_error_table*.json`）

捕获方式：R0 模块（无 patch）+ Fabric bf16-mixed autocast，1 个真实 batch（seed 0），`nn.Linear` 的前向 pre-hook / full-backward hook（块 0、11、22（joint）对 ViT-L；0、6、10（joint）对 ViT-B），head 的 `ConvTranspose2d` k2s2（按 GEMM `[M,C]×[C,4C]` 取，第 1 次和第 4 次 predict 调用）。X = bf16 输入，dY = bf16 grad-output，W = fp32 master。**偏差**：任务要求用 FINAL 栈抓，但 FINAL 的融合块没有逐 Linear 的模块边界，所以用 R0/bf16-mixed 的模块抓；X、dY 与 FINAL 的相差 bf16/MXFP8 的流水线噪声。每个 GEMM 对 fp32 参考（X W ᵀ、dY W、dYᵀ X，TF32 关闭）的相对 L2，方案用 fp4_ref 做（RN/SR/RHT/2D），算子乘积用反量化值在 fp32 里算。

骨干 12 层（4 Linear × 3 块）中位数（括号为最大值）：

| 种类 | mx8 | nv4 RN w2d | nv4 w1d | nv4+RHT | mx4 RN w2d | mx4 w1d | mx4+RHT |
|---|---|---|---|---|---|---|---|
| fwd（ViT-L） | 0.029（0.037） | 0.114（0.145） | 0.095（0.132） | 0.112（0.147） | 0.186（0.203） | 0.157（0.168） | 0.141（0.191） |
| fwd（ViT-B） | 0.027（0.038） | 0.111（0.150） | 0.089（0.138） | 0.117（0.149） | 0.171（0.202） | 0.147（0.174） | 0.157（0.194） |

| dgrad | mx8 | nv4 dY-RN w2d | nv4 dY-SR w2d | nv4 dY-SR w1d | nv4+RHT dY-RN | nv4+RHT dY-SR | mx4 dY-RN | mx4 dY-SR | mx4+RHT dY-SR |
|---|---|---|---|---|---|---|---|---|---|
| ViT-L | 0.034 | 0.135（0.154） | 0.162（0.185） | 0.146（0.171） | 0.142（0.186） | 0.174（0.227） | 0.199 | 0.228 | 0.212 |
| ViT-B | 0.036 | 0.146（0.155） | 0.176（0.186） | 0.156（0.172） | 0.148（0.177） | 0.178（0.219） | 0.214 | 0.253 | 0.222 |

| wgrad | mx8 | nv4 RN | nv4 SR | nv4+RHT RN | nv4+RHT SR | mx4 RN | mx4 SR | mx4+RHT RN | mx4+RHT SR |
|---|---|---|---|---|---|---|---|---|---|
| ViT-L 中位（最大） | 0.022（0.042） | 0.080（**0.604**） | 0.085（**3.04**） | 0.087（**9.09**） | 0.107（**13.0**） | 0.156（0.616） | 0.141（4.63） | 0.124（27.8） | 0.146（19.9） |
| ViT-B 中位（最大） | 0.014（0.031） | 0.080（0.193） | 0.077（0.204） | 0.071（0.224） | 0.083（0.251） | 0.148 | 0.138 | 0.122 | 0.140 |

head ConvT（4 个张量）：前向 mx8 0.036 / nv4 0.144 / mx4 0.164–0.171；dgrad mx8 0.036 / nv4 RN 0.144 / nv4 SR 0.175 / mx4 0.17–0.22；**wgrad mx8 0.002 / nv4 RN 0.013（最大 0.017–0.024）/ nv4+RHT RN 0.005–0.008 / mx4 0.02–0.026**（wgrad 在 7168–57344 个 token 上累加，量化噪声平均掉了）。

**逐项读法**
- FP4 前向误差 0.11（nv4）/ 0.17–0.18（mx4），是 MXFP8 的 4 倍 / 6 倍；dgrad 0.14 / 0.20。这正是 PLAN_v3 担心的"2–4 倍"，前向 FP4 过放宽门的概率低。
- 2D 权重 scale 比 1D 差：nv4 前向 0.114 vs 0.095，dgrad 0.164 vs 0.149（16×16 方块的 amax 更大）；换来 W 与 Wᵀ 一致（NVIDIA 配方）。单步误差表看不出一致性的好处，R20b 的 B/S 门才有意义。
- SR 的单次抽样误差比 RN 大约 20%（0.137→0.164）。SR 的好处（长期无偏）单步误差表看不到。
- RHT 对高斯式数据基本没用（前向 0.112 vs 0.114），对 mx4 有用（前向 0.184→0.141，dgrad 0.202→0.181），对 wgrad 在 head 上有用（0.013→0.007）、在有 outlier token 的层上有害。
- mx4 的误差是 nv4 的 1.3–1.9 倍。

**massive-activation 发现（ViT-L，`profile/debug_wgrad_block0.json`、`debug_prefix_tokens.json`）**
- block 0 的 fc1/fc2：X 最大值 772 / 1992（rms 0.92 / 9.0），dY 最大 1.5e-3 / 1.3e-2（rms 1.8e-6 / 1.5e-5）。最大值出现在 token 下标 ≡ 2 (mod 1029)，即每张图的 register token（还有 cls）。
- 根因：每张量 scale `ts` 被少数 outlier token 定死（nv4 的 `ts = amax/2688`），e4m3 block scale 的动态范围（2^-9 – 448）装不下 dY 的 token 间动态范围：block 0 fc2 的 dY 转置副本里 **24% 的 16 元素块 scale 下溢为 0（整块清零），55% 落在 subnormal**（<2^-6，精度低）；X 的转置副本有 78% 的块 scale 为 0（GELU 输出大量近 0 的值）。
- RHT 沿 token 轴恶化（block 0 fc2 relL2 0.60 → 9.1；同一 batch 的另一次捕获（反向的原子累加使 dY 的位不同）是 0.58 → 25.2，SR 档 1.57 / 3.04，所以 block 0 的 outlier 层误差对 dY 的位敏感，量级结论不变）：outlier token 的 dY 量化噪声被旋转扩散到同一个 16 token 组的所有位置，再和 X 的 outlier token 相乘（两个 outlier 不在同一个 token 上）。`debug_wgrad` 拆开：只量化 dY 就 9.1，只量化 X 只有 0.085。
- 把 cls + register（5 个/图；joint 块还有 100 个 query，共 105 个/图）token 排除出 FP4 token 规约，用小 bf16 GEMM 精确补上：block 0 fc2 nv4 0.604→0.042（RHT 9.09→0.045），fc1 0.259→0.055，block 22（joint）fc1/fc2/proj/qkv nv4 0.055/0.092/0.113/0.059 → 0.007/0.011/0.009/0.022；这些 prefix token 占 dW 范数的 29–93%。普通块的 prefix 是 5 个 token（0.5% FLOP），joint 块是 105 个（9.3% FLOP，bf16 侧 GEMM 约占 FP4 GEMM 的一半时间），所以 joint 块可以只排除 cls+reg（5 个）再看。
- 这个侧 GEMM 的实测成本（gather + bf16 GEMM fp32 输出 + 一次 `N×K` fp32 加进 FP4 的 wgrad 结果）：fc2 wgrad ViT-L 12 µs（不含加）/ 20 µs（含加），对 55 µs 的 FP4 wgrad；全模型每步：只排 5 个 token 时 ViT-L 1.54 ms / ViT-B 0.64 ms，普通块 5 个 + joint 块 105 个时 1.84 / 0.96 ms。加法遍历可以并进 FP4 GEMM 的 epilogue（读 C）。已计入 §4.6 之外，没有从节省里扣掉：S1 要再扣约 1.5–1.8 ms（L）/ 0.6–1.0 ms（B）。

### 4.6 每步节省估计（`profile/per_step_estimate.json`）

GEMM 项来自 §4.2 的实测，乘以每步次数；量化项见下面的两类来源：**"现成 kernel 不融合"是实测**（Spark 行 + 转置 + Spark 量化），**"融合量化器"是估计**（没有这个 kernel）：r17 mx8 双输出量化器的实测时间 × 字节比（nv4 双副本 3.125 B/元素对 mx8 4.06 B/元素；单独 amax 遍历 +2 B；mx4 3.06）× 1.25 的安全系数。权重量化（每步 W 和 Wᵀ 各一份，FINAL 里已经有 MXFP8 的）不单独估计（视为相等）。

| 项（ms/步，L / B） | ΔGEMM | Δ量化（融合，nv4，amax 由生产者给出） | 节省 | 占步时 | ≥3 ms 或 ≥3% |
|---|---|---|---|---|---|
| ① head dgrad+wgrad → nv4 | 8.88 / 9.69 | 1.64 / 2.66 | 7.24 / 7.04 | 7.1% / 8.1% | 满足 / 满足 |
| ③ 骨干 dgrad+wgrad → nv4 | 8.13 / 4.93 | 0.27 / 0.65 | 7.86 / 4.29 | 7.7% / 4.9% | 满足 / 满足 |
| ④ fc1/fc2 前向 → nv4 | 2.73 / 1.28 | -0.19 / -0.65 | 2.92 / 1.93 | 2.9% / 2.2% | **不满足** / **不满足** |
| ⑤a qkv 前向 → MXFP8 | 3.01 / 2.00 | 0.13 / 0.08 | 2.88 / 1.92 | 2.8% / 2.2% | **不满足** / **不满足** |
| ②a head 前向 → MXFP8 | 3.38 / 3.35 | 0.08 / 0.30 | 3.30 / 3.05 | 3.2% / 3.5% | 满足 / 满足 |

合计（节省为负 = 整步变慢）：

| 场景 | ViT-L S1（①+③） | ViT-L S2（再加 ④⑤a②a） | ViT-B S1 | ViT-B S2 |
|---|---|---|---|---|
| 现成 kernel 不融合（实测） | -31.64 ms（-31.1%） | -23.58（-23.2%） | -30.93 ms（-35.7%） | -25.14（-29.0%） |
| 融合，nv4，amax 由生产者给出 | 15.10 ms（14.8%） | 24.20（23.8%） | 11.32 ms（13.1%） | 18.23（21.0%） |
| 融合，nv4，单独 amax 遍历 | 11.28 ms（11.1%） | 20.38（20.0%） | 6.82 ms（7.9%） | 13.72（15.8%） |
| 融合，mx4 | 15.61 ms（15.3%） | 24.77（24.3%） | 11.69 ms（13.5%） | 18.62（21.5%） |

其他（只有 GEMM 项，没有量化估计）：

| 项 | ΔGEMM ms（L / B） | 速度倍数（L / B） |
|---|---|---|
| ①b head dgrad+wgrad → mx8（退路，对 bf16） | 6.61 / 7.04 | 2.50× / 2.40× |
| ②b head 前向 → nv4（stretch） | 4.29 / 4.32 | 5.00× / 3.99× |
| ⑤b qkv 前向 → nv4（stretch） | 3.90 / 2.49 | 5.16× / 5.36× |
| proj 前向 → nv4（不建议：带宽受限） | 0.36 / 0.08 | 1.58× / 1.09× |

（含按元素数缩放的未直接测量形状；标记 `any_quant_scaled_by_elements`。）对比 PLAN_v3 §2 的预估：保守 −12…−15（L）/ −11…−13（B），乐观 −18…−22.5 / −16…−19，实测 GEMM + 估计量化器落在同一量级（S1 15.1 / 11.3，S2 24.2 / 18.2），S2 偏高是因为 head 前向 MXFP8 和 qkv 前向 MXFP8 的 GEMM 节省比预估大。**这些都没有过任何精度门**。

## 5. 正确性

- fp4_ref 自测见 §4.1：与 Spark CUDA 量化器逐字节一致，与 QuACK 的差异有解释；GEMM 消费一致。
- 误差表是单 GEMM 层面的相对 L2，不是 B/S 门；它给出的信号：前向 FP4（nv4 0.11）和 dgrad（0.14）的误差高，wgrad（0.08，head 上 0.01）较低，block 0 的 massive activation 层必须特殊处理。
- 没有改任何参数 / state_dict / 训练状态，所以没有 export / resume 检查。
- 速度数字都是 CUDA graph 回放的算子级数字，不是整步加速；整步 FP4 的数字没有测（not measured）。

## 6. 静态分析与 roofline

不适用（ncu 不可用，R20a 没写 kernel）。参考点：FP4 指令峰值 2,029 TFLOPS（Spark bench_peak），QuACK 在我们形状上最高约 1.32 PFLOPS（mx4，ViT-B head ConvT1 dgrad，14336×768×3072，51 µs；nv4 最高 1.29 PFLOPS），约指令峰值的 65%；带 epilogue 或小输出时受带宽限制：fc1 前向 82 µs 对 plain 53 µs。HBM 拷贝 1528 GB/s。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

**对 R20b 的候选（建议顺序）**

1. `head.dw=nv4+sr`（主线；wgrad 可加 `rht`），退路 `head.dw=mx8`。head wgrad 误差小（0.011），dgrad 误差大（nv4 0.144，SR 0.175）——关注 dgrad 对 B 门的影响。
2. `all.w=nv4+rht+sr`，必须带 `nofirst`（块 0 的 massive activation）或"prefix token 侧 GEMM"的变体；再对比 `all.d=nv4+sr`（w2d 对 w1d）。
3. `qkv.f=mx8`、`head.f=mx8`（r17 真 kernel）。
4. `fc.f=nv4`（±rht）：stretch，单独不满足 ≥3 ms，ViT-B 速度也不满足，只有搭在③的 kernel 上顺带才值得。`head.f=nv4`、`qkv.f=nv4` 是 stretch（前向算子误差 0.11–0.14）。
5. 对照臂：mx4 对 nv4（最优 nv4 档上）、SR 对 RN、RHT 开对关、nolast / nojoint。

**格式 / 后端推荐**：nv4（mx4 GEMM 快 3–7%，量化免 amax，但误差大 1.3–1.9 倍；只有 nv4 需要单独 amax 遍历时 mx4 才在整步上明显赢：S1 差 4.3 ms（L）/ 4.9 ms（B），amax 由生产者给出时只差 0.5 / 0.4 ms）；后端 QuACK（FP4 + 全部 r13 epilogue 可用，nv4 的 `ts` 用 `impl/fp4_epi.py` 的设备标量参数）；小输出的 wgrad 用 `split_k=None` 或 cuBLASLt。

**对 R21/R22 的设计要求**：(1) 一次读入的双副本量化器（行副本 + 转置副本，token 轴补零，blocked scale，可选 RHT/SR），nv4 的 amax 由生产者给出；(2) wgrad 的 prefix token 处理（侧 GEMM 或 `nofirst`）；(3) nv4 的 block-scale 动态范围对 dY 不够（24% 块清零），可以让 R21 的量化器按行/块做 amax 缩放策略，或回退 mx4（范围更大：ue8m0）；(4) fp4_ref 的 SR 与 RHT 的 kernel 对拍用外部 `u` 张量。

**失败项**
- `F.scaled_mm` 的 MXFP4 在 sm_120 上不可用（NotImplementedError）。
- Spark：只有 bf16 输出；K 要 256 的倍数（wgrad 的 Mpad 7296 / 14464 需补到 7424 / 14592，矩阵里已补，对应 FLOP +1.8% / +0.9%）；第一次构建漏了 `fp8quant.cu`（`absmax_bits` 未定义），已在构建脚本里修复；我的绑定需要显式 variant。
- Spark 的设备 amax 路径给出的 ts 在 20 组随机数据里只有 12 组与 fp32 除法 `amax/(6·448)` 逐位相等，其余相差 ≤1 ulp（与 `amax·(1/2688)` 的乘法形式也只吻合 12/20）；R21 若采用要先对齐。
- QuACK `to_nvfp4` 与 fp4_ref 不逐位一致（见 §4.1，原因已解释）。
- 误差表用 R0/bf16-mixed 模块抓，不是 FINAL 栈（见 §4.5）。
- 矩阵跑了两遍（首版缺 mx4 epilogue 和 prefix 侧 GEMM，补齐后重跑；首版 json 在 tmp 的 `prev/`）：同形状的 FINAL-now 与 FP4 时间相差 ≤5%，汇总项相差 ≤0.5 ms；ViT-L head 的 json 来自首版（该文件不受补充项影响）。
- 没有 nsys（算子级）。

**契约变更请求（给协调者）**
1. 11.2：nv4 的 block scale 公式写成 `e4m3_RN(min((amax_blk/6)/ts, 448))`（运算顺序），量化值写成 `x/(e4m3(s_blk)·ts)` 的 fp32 除法，并写明"scale 为 0 的块数据全 0、scale 字节 0"；mx4 全 0 块 scale 字节 0。
2. 11.2：RHT 的符号向量是 16 个 ±1（`rht_signs()`，种子 20251003），先 FWHT 后乘符号。
3. 11.5 A 门：对 NVFP4 per-tensor scale，设备 amax 路径要和 `amax/(6·448)` 的 fp32 除法逐位对齐（Spark 的不对齐）。
4. 11.4：策略串加 `prefix` 选项（wgrad 的 cls+reg token 走 bf16 侧 GEMM）和 `amax=producer|pass`。
5. 11.6：④ / ⑤ 单独不满足"≥3 ms 或 ≥3%"，但作为③的增量几乎没有额外 kernel 成本，请协调者决定是否把"部分"定义为"一个 kernel 轮内能一起做的项"。

**下一个瓶颈**：FP4 之后 GEMM 本身退居次要，量化器（需要自写的双输出 FP4 量化器）、带 fp32 残差流的 epilogue（proj / fc2 前向，带宽受限）和 attention / LN / 逐元素（45 ms，不在本阶段范围）成为主要部分。


---

# Round20（R20b）：FP4 策略门扫描（半真实仿真：torch 量化 + 真实 QuACK FP4 GEMM），B 门 / S 门，负责：Sonnet，第三阶段

> 本章只覆盖 R20b，R20a 的内容（上面）没有改动。数字全部在 `infer-5090`（RTX 5090）上实测，持双锁，没测的写 not measured / null。**所有门都是 eager 下的仿真**：量化器是 torch 实现（`fp4_ref` 语义），GEMM 是真实的 QuACK FP4 kernel，所以门的结果能直接迁移到 R22（只需把 torch 量化器换成融合 kernel）；**这里没有任何整步速度数字**（仿真的 torch 量化器比融合 kernel 慢得多，整步时间没有意义）。节省来自 R20a 的 `per_step_estimate`（GEMM 实测 + 融合量化器估计），不是本轮实测。
> 状态：完成。计划内的 B 门（筛选 3 批 + 决赛 8 批）、S 门（ViT-L：B3 / H3 / B4；ViT-B：B3 / H3，另有 B4 / exact-amax 对照）、对照臂（8 批，B4 为底）、块测试都已跑完；没测的项见 §0 “没测”。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 做了什么 | `impl/fp4_policy.py`（策略串解析）、`impl/fp4_sim.py`（simulation 量化器：`fp4_ref` 语义 + 专用 SR 生成器 + 共享 RHT 符号 + `amax=exact|delayed`）、`impl/fp4_gemm.py`（真实 QuACK FP4 GEMM 的 fwd / dgrad / wgrad 封装，nv4 走 `fp4_epi.nv4_epilogues()`、mx4 走 r13 epilogue）、`impl/fp4_block.py`（r17 融合块的副本，按 GEMM 种类混合 bf16 / mx8 / nv4 / mx4）、`impl/fp4_head.py`（r14 head 的副本，ConvT f/d/w 分别选格式）、`impl/patch.py`（`BEST_STACK_V2 + ("round20_fp4_probe",)`，策略运行时切换 `ctx.meta["round20_fp4_probe"].set_policy(s)`）；`tests/` 里的块测试、`gate_b_multi.py`、`gate_s_multi.py`、`summarize_r20b.py` |
| 门结果（一句话） | vitl16（n=8）放宽门通过：FINAL/H1/H2/H3/B3/B4/B5；严格门通过：FINAL/H1/H2/H3/B4/B5（严格门只记录）；vitb16（n=8）放宽门通过：FINAL/H1/H2/H3/B3/B4/B5；严格门通过：FINAL/H1/H2（严格门只记录）。S 门：ViT-L（B3 / H3 / B4 × 4 场景）和 ViT-B（B3 / H3 × 4 场景，加上表中已有的 B4 / exact-amax 对照）全部通过，EMA 偏差对 R0 最大 0.74%、对 FINAL 最大 0.85%（门 5%），无 NaN / Inf，delayed amax 下无异常。仅记录项（筛选 n=3）：`fc.f=nv4` 在 6 个底上全部不过放宽门；`qkv.f=mx8` 在 5/6 个底上过、ViT-L 的 B3 底不过，信号不稳。 |
| 推荐 POLICY | ViT-L：`head.f=mx8,head.dw=nv4+sr,all.d=nv4+sr,all.w=nv4+sr+rht,nofirst,amax=delayed`（B4）；ViT-B：`head.f=mx8,head.dw=nv4+sr,all.d=nv4+sr,all.w=nv4+sr+rht,amax=delayed`（B3）；两个模型的仅 head 退路档：`head.f=mx8,head.dw=nv4+sr,amax=delayed`（H3；更保守的 `head.dw=mx8` 对 R0 的余弦与 FINAL 几乎相同）。按 R20a `per_step_estimate`（融合量化器、amax 由生产者给出，**不是实测**）：ViT-L B4 ≈ 18.07 ms（17.8%），H3 ≈ 10.54 ms（10.4%）；ViT-B B3 ≈ 14.37 ms（16.6%），H3 ≈ 10.09 ms（11.6%）。 |
| R21b（head）/ R22（骨干） | **R21b（head）：执行**（H3 = ①+②a 过 11.6 的三条：graph 下 5.1× / 2.7× 的 GEMM 速度、B+S 门、≥3 ms）。**R22（骨干）：执行**（③ 过三条；ViT-L 用 nofirst，ViT-B 不需要 prefix / nofirst）。④（fc 前向 nv4）、⑤a（qkv 前向 mx8）：不执行（R20a 的速度 / 节省规则不满足，B 门上 fc.f=nv4 也不过）。 |
| 没测 | 整步速度（graph 下与 FINAL 同场次）：R20b 只做精度门，没有真 kernel；graph_equiv / export / resume：R20b 不发布栈、不改参数；qkv.f=mx8 / fc.f=nv4 的 ≥8 批 B 门（只有筛选阶段 n=3 的记录）；ViT-L 的 amax=exact 对照 S 门；ViT-L S 门的 delayed 诊断计数（诊断代码在 ViT-L S 门跑完之后才加入）；S 门里的 mx4 / RN / w1d / norht 对照臂（它们只做了 B 门） |

## 1. 目标与假设

用放宽门（CONTRACTS 11.5）筛出哪些 GEMM 可以改成 FP4、用哪种配方，定下两个模型各自的 POLICY，并套用 11.6 的 go/no-go。

| 假设（R20a 之后） | 本轮要验证 |
|---|---|
| head 的 dgrad+wgrad 用 nv4+SR 能过放宽门（R20a：head wgrad 误差 0.013，dgrad 0.144–0.175） | H1 / H3 的 B 门（≥8 批）、S 门 |
| 骨干 dgrad+wgrad nv4（RHT+SR 用在 wgrad）能过放宽门；block 0 的 massive activation 要处理（prefix 或 nofirst） | B1–B5，B3 对 B4 对 B5 |
| delayed amax 在 50 步里不会让 nv4 饱和 | S 门（候选全部用 `amax=delayed`），加 `amax=exact` 对照与 cur/prev 比值诊断 |
| fc 前向 nv4（④）和 qkv 前向 mx8（⑤a）按 R20a 规则不进实现，但可以顺带筛选 | 仅记录 |

## 2. 改动（全部在 `train-opt/round20_fp4_probe/`；没有改 round0–19、common、PLAN、CONTRACTS，也没有改 R20a 已有的 `fp4_ref.py` / `fp4_epi.py` 等文件，R21a 需要的这两个文件语义不变）

| 文件 | 内容 |
|---|---|
| `impl/fp4_policy.py` | 策略串：`<lin>.<kinds>=<fmt>[+opts]`，lin ∈ qkv/proj/fc1/fc2/fc/attn/all/head，kinds ⊆ fdw，fmt ∈ bf16/mx8/nv4/mx4；条目选项 `sr rn rht w2d w1d`（作用于紧跟的条目），全局选项 `nofirst nolast nojoint prefix amax=exact|delayed`。FP4 的 d / w 条目默认 SR（梯度 dY），w2d；没写到的保持 FINAL（qkv f=bf16 d=mx8 w=mx8；proj/fc1/fc2 mx8；head bf16）。空串 = FINAL |
| `impl/fp4_sim.py` | `Sim`：专用 `torch.Generator`（SR）、`rht_signs` 符号向量（种子 20251003）、delayed-amax 表（每个调用点 `(prev_amax, inited)` 设备张量；`torch.where` 选择，无 host 同步）+ 诊断（超 2× 次数、cur/prev 最大比值、最差调用点）。`q_matrix / q_matrix_2d / q_row / q_col / q_weight`：`amax="exact"` 时与 `fp4_ref.quantize_matrix / quantize_matrix_2d / quantize / quantize_weight` 逐位一致（RN、给定 u 的 SR、RHT、2D/1D 权重；测试 b） |
| `impl/fp4_gemm.py` | 真实 QuACK FP4 GEMM：`fwd_plain/bias/res/gelu`、`dgrad`、`dgrad_dgelu`、`wgrad`（fp32 输出）；nv4 的 `ts = ts_a·ts_b` 以设备标量传给 epilogue |
| `impl/fp4_block.py` | r17 `fused_block_mx8.py` 的副本（复用 r13 的 `_LN1/Shadows/_Bundle/eligible`、r17 的 `mx8` / `gemm_mx8`）。`LayerQ`：每个 Linear 的 f/d/w 各自选格式；X 在前向只量化需要的副本（fwd 行副本、wgrad 转置副本），dY 只量化一次（dgrad 行副本、wgrad 转置副本）；权重每步在 block 0 量化（`Handle.begin_step`，同 r17 `Mx8Weights.refresh`）；`prefix`：wgrad 的 FP4 收缩只覆盖每张图的后 1024 个 patch token，prefix token（plain 块 5 个，joint 块 105 个）用 bf16 小 GEMM（fp32 输出）精确算后加回 |
| `impl/fp4_head.py` | r14 `head.py` 的副本：每个 ConvT 的 f/d/w 各自选 bf16（r14 的 QuACK / cuBLAS）/ mx8（r17）/ nv4 / mx4；d、w 支持 `+sr`，w 支持 `+rht`（沿像素维）。策略空串时与 r14 `fused` 逐位一致（测试 e） |
| `impl/patch.py` | `round20_fp4_probe`：覆盖 `ops.block` / `ops.predict`（旧的作回退），不改 Parameter 身份和 state_dict（apply_patches 会检查），`variant` = 策略串 |
| `tests/test_fp4_block.py` | a 策略解析；b `fp4_sim` 对 `fp4_ref` 逐位、delayed 语义、默认生成器未被碰；c 真实 QuACK FP4 GEMM 对反量化 fp64 乘积；d prefix 复现 R20a 的 block 0 数字；e 空策略对 r17 块 / r14 head；f 块级合理性 |
| `tests/gate_b_multi.py` | 从 r19 复制改写：一个进程里 R0、P1、FINAL 和所有候选共用同一次 build / 同一初始权重 / 同一批 / 同一 RngTape；candidate 在同一个 round20 ctx 里运行时切换策略；候选前后各跑一次 FINAL（`FINAL`、`FINAL_again`）确认运行间无漂移 |
| `tests/gate_s_multi.py` | 4 个场景 × 50 步 × 2 批；R0、FINAL 和候选同一进程；候选 `amax=delayed`；对 R0 和对 FINAL 的 EMA 偏差、NaN/Inf、grad-norm 比值、delayed 诊断 |
| `tests/cands.py`, `queue_*.sh`, `retry.sh`, `run_gateB.sh`, `summarize_r20b.py`, `show.py` | 候选表、队列（`retry.sh` 处理 `assert_gpu_exclusive` 误报）、汇总表 |
| `profile/` | `gateB_r20b_*_screen.json`、`*_finals.json`、`*_arms.json`、`gateS_r20b_*.json`、`test_fp4_block.json`、`tables_r20b.md` |

**与 PLAN 的差别**：PLAN_v3 说 nv4 / mx4 走“fake-quant 后的 bf16 GEMM”；按本任务和 6a 的要求改成“torch 量化 + **真实** QuACK FP4 GEMM”（R20a 已验证 fp4_ref 的操作数喂 QuACK 结果正确），所以累加顺序也和真 kernel 相同。量化噪声（RN / SR / RHT / delayed amax）是真的，只有“量化 kernel 的速度”是仿真的。

## 3. nsys 分析过程

**本轮没有 nsys，也没有整步速度测量**：R20b 是精度门（B / S），全部在 eager 下用 torch 量化器仿真，不能代表融合后的 kernel 时间，整步 graph 下的速度要等 R21/R22 的真 kernel。节省估计仍用 R20a 的 `per_step_estimate.json`（GEMM 实测 + 融合量化器估计）：

| 项 | 来源 | ViT-L | ViT-B |
|---|---|---:|---:|
| ① head dgrad+wgrad → nv4 | R20a §4.6（量化器 amax 由生产者给出） | 7.24 ms | 7.04 ms |
| ②a head 前向 → mx8 | 同上 | 3.30 ms | 3.05 ms |
| ③ 骨干 dgrad+wgrad → nv4 | 同上 | 7.86 ms | 4.29 ms |
| prefix 侧 GEMM 成本（plain 5 + joint 105 token） | R20a §4.3 实测 | 1.84 ms | 0.96 ms |
| nofirst 损失 ≈ ③ 的 1/24（L）/ 1/12（B） | 估计 | 0.33 ms | 0.36 ms |
| FINAL 步时（同机，PLAN §0） | | 101.8 ms | 86.6 ms |

另外，仿真过程中记录了每个候选一步 fwd+bwd 的 eager 时间（`gateB` json 的 `seconds`），只用于确认没有异常，**不是速度数据**。

**命令**（每个 python 进程从启动到退出持双锁，≤10 分钟；队列脚本见 `tests/queue_*.sh`）：

```bash
python tests/test_fp4_block.py --only a,b,c,d,e,f                  # -> profile/test_fp4_block.json
python tests/gate_b_multi.py --model vitl16 --cands H1 H2 H3 B1 B2 B3 B4 B5 B3+mx4 ... --seeds 0 1 2 --out profile/gateB_r20b_vitl16_screen.json   # 筛选，全部候选 + 对照臂 + 仅记录项
python tests/gate_b_multi.py --model vitl16 --cands H1 H2 H3 B3 B4 B5 --seeds 0..7 --per-param --out profile/gateB_r20b_vitl16_finals.json  # 决赛，≥8 批
python tests/gate_b_multi.py --model vitl16 --cands B4 B4+mx4 B4+rn B4+w1d B4+norht --seeds 0..7 --out profile/gateB_r20b_vitl16_arms.json   # 对照臂，≥8 批
python tests/gate_s_multi.py --model vitl16 --cands B3 H3 B4 --scenarios schedule compressed --out profile/gateS_r20b_vitl16_a.json            # S 门（两个进程各 2 个场景）
python tests/summarize_r20b.py --results                              # -> profile/tables_r20b.md, results.json["r20b"]
```

## 4. 实测结果

### 4.1 仿真块 / head 的单元测试（`tests/test_fp4_block.py`，`profile/test_fp4_block.json`）

| 测试 | 结果 |
|---|---|
| a 策略解析 | 通过（FINAL 默认值、别名、SR 默认、nofirst/nolast/nojoint 回到 FINAL、非法输入报错） |
| b `fp4_sim` 对 `fp4_ref` | `amax=exact`：nv4 / mx4 的 RN、RHT+SR（给定同一个 u）、转置副本 + RHT、2D / 1D 权重，数据字节与 scale 字节全部逐位一致；`delayed`：第一次调用 = exact，第二次 ts = 2 × 上一次 amax / 2688（逐位）；默认 CPU / CUDA 生成器状态在量化之后不变（RngTape 对齐不受影响） |
| c 真实 QuACK FP4 GEMM 分派（ViT-B plain 块形状 qkv / proj / fc1 / fc2，nv4 和 mx4） | fwd bias / res / gelu 对反量化 fp64 乘积相对 L2 = 1.66e-3（= bf16 输出舍入下限；fp32 输出的 `fwd_res` 的 out 为 0）；dgrad 1.66e-3（mx4 1.7e-3）；dgrad+GELU' 及 db 对参考 1.66e-3 / 0；wgrad（fp32 输出）与 prefix 版 wgrad 对反量化乘积 = 0（逐位）。门槛 ≤2e-3 |
| d prefix 复现 R20a block 0（ViT-L 缓存张量 `cap_vitl16.pt`） | nv4 wgrad 相对 L2：block0.fc1 0.2594 → prefix 后 0.0549；fc2 0.6045 → 0.0416；fc2+RHT 9.0914 → 0.0449；proj 0.1279 → 0.0948；qkv 0.0473 → 0.0496；block22.fc2 0.0919 → 0.0113。与 R20a `debug_prefix_tokens.json` 的数字在 1e-6 内一致（`fp4_block.LayerQ` 里的 prefix 实现 = R20a 的侧 GEMM 方案） |
| e 空策略（= FINAL）对 r17 块 / r14 head（ViT-B，plain 块 1 和 joint 块 11） | 块输出相对差 0.0（逐位），dx 相对差 1e-5（r17 / r13 dx 的原子累加不确定性，两次运行 r17 对 r17 也有）, 所有参数梯度余弦 ≥0.99999（`dW_cos_min`=1.0）；head：mask logits、dx、全部参数梯度、cls logits 与 r14 **逐位相同** |
| e head 策略对 r14（单次、合成输入，ViT-B） | `head.dw=nv4+sr`：mask logits 余弦 1.0000、dx 0.984、参数梯度余弦最小 0.9715（`upscale.0.conv1.weight`）；`head.f=mx8,head.dw=nv4+sr+rht`：0.9985 / 0.983 / 0.970；`head.dw=mx8`（退路）：1.0000 / 0.9993 / 0.9986 |
| f 块级合理性（对 R0 的默认 torch 块在 bf16 autocast 下的输出和梯度；输入 = 随机图经 `prepare_tokens` 得到的 token，**不含**真实训练里的 massive activation；ViT-B 块 0 / 1 / 11（joint），ViT-L 块 0 / 1 / 23（joint）；无 autocast 的 fp32 块在这里 SDPA 报 `No available kernel`，所以参考是 bf16 autocast 的 R0 块） | FINAL 块：out ≥0.9992，dx 0.9977–0.9988（ViT-L 块 0 为 0.9857），dW 最小 0.996–0.997（ViT-L 块 0 为 0.886，LN 参数）。`all.d=nv4+sr,all.w=nv4+sr+rht`：out 不变；**dx 0.970–0.982（ViT-B）、0.965–0.984（ViT-L）**；dW 最小 plain 块 0.907–0.975，joint 块 0.894（ViT-B）/ 0.961（ViT-L）；加 `prefix`：ViT-B joint 块 dW 最小 0.894 → 0.958，plain 块和 ViT-L 基本不变。`all.f=nv4`（前向）：out 0.942–0.9998，**dx 0.67–0.99**（ViT-B 块 0 为 0.67、块 1 为 0.74；ViT-L 块 1 为 0.83），dW 最小 0.07–0.92 → 前向 nv4 对单块反向的破坏远大于 dgrad / wgrad 的 nv4 |

### 4.2 筛选（B 门，同一进程、同一初始权重 / 批 / RngTape，n 见表）

候选表（`tests/cands.py`）：

| 编号 | 策略 |
|---|---|
| H1 | `head.dw=nv4+sr` |
| H2 | `head.dw=mx8`（退路） |
| H3 | `head.f=mx8,head.dw=nv4+sr` |
| B1 | H3 + `all.w=nv4+sr+rht` |
| B2 | B1 + `prefix` |
| B3 | H3 + `all.d=nv4+sr,all.w=nv4+sr+rht` |
| B4 | B3 + `nofirst` |
| B5 | B3 + `prefix` |
| 对照臂 | 以 B3 / B4 / B5 为底：`+mx4`（nv4→mx4）、`+rn`（SR→RN）、`+w1d`（权重 1D scale）、`+norht`（wgrad 不用 rht）；仅记录：`+fcF4`（`fc.f=nv4`）、`+qkvF8`（`qkv.f=mx8`） |

注意：**数据加载器在不同进程里对同一个 seed 给出的 batch 不完全相同**（R0 的 loss 在两次进程里不同：例如 ViT-B seed 0 为 30.475 / 30.098），所以只有“同一进程内”的比较是配对的；表里的行是各自进程内对 R0 / P1 的值，行与行之间 <0.001 的差别要当噪声。

#### vitl16：筛选（全部候选；n 见表，同一进程、同一初始权重 / 批 / RngTape）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 3 | 0.9950 / 0.9938 | 0.9977 / 0.9939 | 1.9e-03 | 通过(n<8) | 通过 | 1.0000 | 0.545 | 2.0 |
| H1 | 3 | 0.9932 / 0.9923 | 0.9962 / 0.9923 | 1.9e-03 | 通过(n<8) | 通过 | 0.9984 | 0.714 | 0.5 |
| H2 | 3 | 0.9949 / 0.9938 | 0.9977 / 0.9938 | 1.9e-03 | 通过(n<8) | 通过 | 0.9999 | 0.632 | 0.2 |
| H3 | 3 | 0.9930 / 0.9921 | 0.9959 / 0.9922 | 1.9e-03 | 通过(n<8) | 通过 | 0.9982 | 0.722 | 0.4 |
| B1 | 3 | 0.9906 / 0.9879 | 0.9932 / 0.9879 | 1.9e-03 | 通过(n<8) | 通过 | 0.9955 | -0.410 | 1.0 |
| B2 | 3 | 0.9930 / 0.9921 | 0.9959 / 0.9922 | 1.9e-03 | 通过(n<8) | 通过 | 0.9982 | 0.686 | 0.9 |
| B3 | 3 | 0.9899 / 0.9845 | 0.9926 / 0.9845 | 1.9e-03 | 通过(n<8) | 不通过 | 0.9947 | -0.023 | 1.7 |
| B4 | 3 | 0.9924 / 0.9914 | 0.9952 / 0.9914 | 1.9e-03 | 通过(n<8) | 通过 | 0.9974 | 0.214 | 1.2 |
| B5 | 3 | 0.9925 / 0.9914 | 0.9953 / 0.9915 | 1.9e-03 | 通过(n<8) | 通过 | 0.9975 | 0.320 | 1.2 |
| B3+mx4 | 3 | 0.9796 / 0.9783 | 0.9796 / 0.9783 | 1.9e-03 | **不通过**(n<8) | 不通过 | 0.9841 | -0.295 | 1.1 |
| B3+rn | 3 | 0.9904 / 0.9865 | 0.9933 / 0.9865 | 1.9e-03 | 通过(n<8) | 通过 | 0.9955 | 0.021 | 0.9 |
| B3+w1d | 3 | 0.9886 / 0.9774 | 0.9915 / 0.9774 | 1.9e-03 | 通过(n<8) | 不通过 | 0.9937 | -0.222 | 1.2 |
| B3+norht | 3 | 0.9924 / 0.9913 | 0.9951 / 0.9914 | 1.9e-03 | 通过(n<8) | 通过 | 0.9973 | 0.321 | 1.0 |
| B3+fcF4 | 3 | 0.9681 / 0.9552 | 0.9680 / 0.9551 | 2.7e-03 | **不通过**(n<8) | 不通过 | 0.9716 | -0.104 | 2.1 |
| B3+qkvF8 | 3 | 0.9753 / 0.9690 | 0.9755 / 0.9689 | 3.5e-03 | **不通过**(n<8) | 不通过 | 0.9802 | -0.509 | 1.3 |
| B4+mx4 | 3 | 0.9917 / 0.9905 | 0.9943 / 0.9905 | 1.9e-03 | 通过(n<8) | 通过 | 0.9965 | 0.029 | 1.1 |
| B4+rn | 3 | 0.9924 / 0.9915 | 0.9953 / 0.9916 | 1.9e-03 | 通过(n<8) | 通过 | 0.9975 | 0.477 | 0.8 |
| B4+w1d | 3 | 0.9926 / 0.9916 | 0.9954 / 0.9916 | 1.9e-03 | 通过(n<8) | 通过 | 0.9976 | 0.382 | 1.2 |
| B4+norht | 3 | 0.9925 / 0.9913 | 0.9952 / 0.9914 | 1.9e-03 | 通过(n<8) | 通过 | 0.9973 | 0.509 | 1.0 |
| B4+fcF4 | 3 | 0.9797 / 0.9401 | 0.9796 / 0.9399 | 4.0e-03 | **不通过**(n<8) | 不通过 | 0.9835 | 0.065 | 1.2 |
| B4+qkvF8 | 3 | 0.9886 / 0.9859 | 0.9888 / 0.9886 | 2.8e-03 | 通过(n<8) | 不通过 | 0.9939 | 0.224 | 1.2 |
| B5+mx4 | 3 | 0.9923 / 0.9907 | 0.9946 / 0.9908 | 1.9e-03 | 通过(n<8) | 通过 | 0.9968 | -0.277 | 1.7 |
| B5+rn | 3 | 0.9926 / 0.9915 | 0.9953 / 0.9916 | 1.9e-03 | 通过(n<8) | 通过 | 0.9975 | 0.494 | 0.9 |
| B5+w1d | 3 | 0.9927 / 0.9916 | 0.9954 / 0.9917 | 1.9e-03 | 通过(n<8) | 通过 | 0.9976 | 0.535 | 1.2 |
| B5+norht | 3 | 0.9925 / 0.9914 | 0.9953 / 0.9915 | 1.9e-03 | 通过(n<8) | 通过 | 0.9974 | 0.149 | 1.0 |
| B5+fcF4 | 3 | 0.9769 / 0.9653 | 0.9794 / 0.9652 | 3.7e-03 | **不通过**(n<8) | 不通过 | 0.9830 | -0.105 | 1.3 |
| B5+qkvF8 | 3 | 0.9918 / 0.9884 | 0.9920 / 0.9884 | 3.5e-03 | 通过(n<8) | 通过 | 0.9933 | -0.327 | 1.2 |
| FINAL_again | 3 | 0.9950 / 0.9938 | 0.9977 / 0.9939 | 1.9e-03 | 通过(n<8) | 通过 | 1.0000 | 0.556 | 0.2 |


#### vitb16：筛选（全部候选；n 见表，同一进程、同一初始权重 / 批 / RngTape）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 3 | 0.9958 / 0.9824 | 0.9957 / 0.9826 | 2.1e-03 | 通过(n<8) | 通过 | 1.0000 | 0.612 | 1.3 |
| H1 | 3 | 0.9944 / 0.9810 | 0.9942 / 0.9812 | 2.1e-03 | 通过(n<8) | 通过 | 0.9987 | 0.715 | 0.5 |
| H2 | 3 | 0.9957 / 0.9823 | 0.9957 / 0.9825 | 2.1e-03 | 通过(n<8) | 通过 | 0.9999 | 0.646 | 0.1 |
| H3 | 3 | 0.9919 / 0.9814 | 0.9921 / 0.9816 | 2.0e-03 | 通过(n<8) | 通过 | 0.9977 | 0.682 | 0.5 |
| B1 | 3 | 0.9914 / 0.9810 | 0.9916 / 0.9813 | 2.0e-03 | 通过(n<8) | 通过 | 0.9975 | 0.695 | 0.9 |
| B2 | 3 | 0.9919 / 0.9813 | 0.9921 / 0.9815 | 2.0e-03 | 通过(n<8) | 通过 | 0.9977 | 0.703 | 0.9 |
| B3 | 3 | 0.9886 / 0.9784 | 0.9887 / 0.9786 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9947 | 0.241 | 1.1 |
| B4 | 3 | 0.9886 / 0.9783 | 0.9888 / 0.9785 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9945 | 0.162 | 1.1 |
| B5 | 3 | 0.9890 / 0.9780 | 0.9891 / 0.9782 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9943 | 0.066 | 1.1 |
| B3+mx4 | 3 | 0.9835 / 0.9747 | 0.9837 / 0.9748 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9905 | -0.037 | 1.1 |
| B3+rn | 3 | 0.9891 / 0.9786 | 0.9892 / 0.9788 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9948 | 0.499 | 0.8 |
| B3+w1d | 3 | 0.9898 / 0.9790 | 0.9900 / 0.9792 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9955 | 0.464 | 1.1 |
| B3+norht | 3 | 0.9889 / 0.9782 | 0.9890 / 0.9784 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9945 | 0.291 | 1.0 |
| B3+fcF4 | 3 | 0.9630 / 0.9367 | 0.9645 / 0.9370 | 1.2e-02 | **不通过**(n<8) | 不通过 | 0.9659 | -0.179 | 1.2 |
| B3+qkvF8 | 3 | 0.9884 / 0.9754 | 0.9889 / 0.9755 | 4.5e-03 | 通过(n<8) | 不通过 | 0.9899 | 0.119 | 1.1 |
| B4+mx4 | 3 | 0.9843 / 0.9750 | 0.9845 / 0.9752 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9909 | -0.023 | 1.0 |
| B4+rn | 3 | 0.9890 / 0.9788 | 0.9892 / 0.9790 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9950 | 0.546 | 0.8 |
| B4+w1d | 3 | 0.9897 / 0.9790 | 0.9899 / 0.9792 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9956 | 0.468 | 1.1 |
| B4+norht | 3 | 0.9888 / 0.9785 | 0.9890 / 0.9787 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9947 | 0.147 | 1.0 |
| B4+fcF4 | 3 | 0.9627 / 0.9449 | 0.9638 / 0.9452 | 1.3e-02 | **不通过**(n<8) | 不通过 | 0.9664 | -0.179 | 1.1 |
| B4+qkvF8 | 3 | 0.9869 / 0.9749 | 0.9868 / 0.9751 | 4.9e-03 | 通过(n<8) | 不通过 | 0.9889 | -0.263 | 1.1 |
| B5+mx4 | 3 | 0.9849 / 0.9756 | 0.9852 / 0.9757 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9913 | 0.278 | 1.0 |
| B5+rn | 3 | 0.9894 / 0.9788 | 0.9896 / 0.9790 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9950 | 0.504 | 0.8 |
| B5+w1d | 3 | 0.9904 / 0.9792 | 0.9906 / 0.9794 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9958 | 0.466 | 1.1 |
| B5+norht | 3 | 0.9890 / 0.9782 | 0.9892 / 0.9784 | 2.0e-03 | 通过(n<8) | 不通过 | 0.9945 | 0.172 | 1.0 |
| B5+fcF4 | 3 | 0.9636 / 0.9368 | 0.9651 / 0.9371 | 1.2e-02 | **不通过**(n<8) | 不通过 | 0.9667 | 0.020 | 1.2 |
| B5+qkvF8 | 3 | 0.9892 / 0.9757 | 0.9897 / 0.9758 | 4.5e-03 | 通过(n<8) | 不通过 | 0.9903 | 0.224 | 1.2 |
| FINAL_again | 3 | 0.9958 / 0.9824 | 0.9957 / 0.9826 | 2.1e-03 | 通过(n<8) | 通过 | 1.0000 | 0.616 | 0.2 |


### 4.3 决赛（B 门，≥8 批）

放宽门（对 R0 和对 P1 分别）：全局梯度余弦中位数 ≥0.98、最小 ≥0.97、loss_rel ≤2e-2；严格门只记录。ViT-L 的各候选 8 批来自 3 个进程（seed 0 / 2、seed 1、seed 3–7），每个种子内部所有候选同一批；ViT-B 的决赛重新在**一个进程**里跑了 seed 0–7（旧的混合进程版本保留为 `gateB_r20b_vitb16_finals_mixedproc.json`）。

#### vitl16：决赛 B 门（≥8 批）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 8 | 0.9962 / 0.9938 | 0.9970 / 0.9939 | 2.5e-03 | 通过 | 通过 | 1.0000 | -0.636 | 1.1 |
| H1 | 8 | 0.9946 / 0.9923 | 0.9955 / 0.9923 | 2.5e-03 | 通过 | 通过 | 0.9985 | -0.708 | 0.4 |
| H2 | 8 | 0.9962 / 0.9938 | 0.9970 / 0.9938 | 2.5e-03 | 通过 | 通过 | 0.9999 | -0.608 | 0.2 |
| H3 | 8 | 0.9938 / 0.9921 | 0.9945 / 0.9922 | 2.1e-03 | 通过 | 通过 | 0.9979 | -0.573 | 0.4 |
| B3 | 8 | 0.9897 / 0.9845 | 0.9893 / 0.9845 | 2.1e-03 | 通过 | 不通过 | 0.9934 | -0.681 | 1.4 |
| B4 | 8 | 0.9926 / 0.9892 | 0.9930 / 0.9898 | 2.1e-03 | 通过 | 通过 | 0.9969 | -0.750 | 1.2 |
| B5 | 8 | 0.9926 / 0.9904 | 0.9930 / 0.9910 | 2.1e-03 | 通过 | 通过 | 0.9969 | -0.758 | 1.2 |


#### vitb16：决赛 B 门（≥8 批）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 8 | 0.9949 / 0.9824 | 0.9948 / 0.9903 | 2.3e-03 | 通过 | 通过 | 1.0000 | 0.492 | 0.4 |
| H1 | 8 | 0.9936 / 0.9815 | 0.9936 / 0.9892 | 2.3e-03 | 通过 | 通过 | 0.9987 | 0.453 | 0.5 |
| H2 | 8 | 0.9948 / 0.9824 | 0.9947 / 0.9902 | 2.3e-03 | 通过 | 通过 | 0.9999 | 0.471 | 0.1 |
| H3 | 8 | 0.9925 / 0.9874 | 0.9927 / 0.9780 | 2.2e-03 | 通过 | 不通过 | 0.9969 | 0.449 | 0.5 |
| B3 | 8 | 0.9884 / 0.9839 | 0.9892 / 0.9746 | 2.2e-03 | 通过 | 不通过 | 0.9930 | -0.554 | 1.2 |
| B4 | 8 | 0.9884 / 0.9840 | 0.9893 / 0.9747 | 2.2e-03 | 通过 | 不通过 | 0.9931 | -0.518 | 1.1 |
| B5 | 8 | 0.9886 / 0.9841 | 0.9896 / 0.9748 | 2.2e-03 | 通过 | 不通过 | 0.9931 | -0.323 | 1.2 |


### 4.4 对照臂（以 B4 为底，≥8 批，同一进程配对）

#### vitl16：对照臂（以 B4 为底，≥8 批，同一进程配对）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 8 | 0.9966 / 0.9911 | 0.9964 / 0.9905 | 3.3e-03 | 通过 | 通过 | 1.0000 | 0.730 | 0.4 |
| B4 | 8 | 0.9932 / 0.9817 | 0.9936 / 0.9811 | 3.1e-03 | 通过 | 通过 | 0.9964 | 0.348 | 1.2 |
| B4+mx4 | 8 | 0.9921 / 0.9796 | 0.9925 / 0.9796 | 3.1e-03 | 通过 | 不通过 | 0.9952 | -0.289 | 1.1 |
| B4+rn | 8 | 0.9933 / 0.9818 | 0.9937 / 0.9812 | 3.1e-03 | 通过 | 通过 | 0.9965 | 0.295 | 0.9 |
| B4+w1d | 8 | 0.9934 / 0.9819 | 0.9938 / 0.9813 | 3.1e-03 | 通过 | 通过 | 0.9966 | 0.416 | 1.2 |
| B4+norht | 8 | 0.9932 / 0.9816 | 0.9936 / 0.9811 | 3.1e-03 | 通过 | 通过 | 0.9964 | 0.216 | 1.0 |
| FINAL_again | 8 | 0.9966 / 0.9911 | 0.9964 / 0.9905 | 3.3e-03 | 通过 | 通过 | 1.0000 | 0.698 | 0.2 |


#### vitb16：对照臂（以 B4 为底，≥8 批，同一进程配对）

| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| FINAL | 8 | 0.9960 / 0.9853 | 0.9962 / 0.9860 | 2.6e-03 | 通过 | 通过 | 1.0000 | 0.157 | 0.4 |
| B4 | 8 | 0.9919 / 0.9802 | 0.9916 / 0.9805 | 2.7e-03 | 通过 | 通过 | 0.9939 | -0.590 | 1.1 |
| B4+mx4 | 8 | 0.9874 / 0.9758 | 0.9870 / 0.9762 | 2.7e-03 | 通过 | 不通过 | 0.9897 | -0.309 | 1.0 |
| B4+rn | 8 | 0.9923 / 0.9803 | 0.9920 / 0.9806 | 2.7e-03 | 通过 | 通过 | 0.9942 | -0.436 | 0.8 |
| B4+w1d | 8 | 0.9928 / 0.9811 | 0.9925 / 0.9815 | 2.7e-03 | 通过 | 通过 | 0.9946 | -0.172 | 1.1 |
| B4+norht | 8 | 0.9921 / 0.9800 | 0.9918 / 0.9803 | 2.7e-03 | 通过 | 通过 | 0.9940 | -0.220 | 1.0 |
| FINAL_again | 8 | 0.9960 / 0.9854 | 0.9962 / 0.9861 | 2.6e-03 | 通过 | 通过 | 1.0000 | 0.197 | 0.1 |


筛选阶段（3 批）对 B3 / B4 / B5 的所有对照臂结果见 §4.2 的两张筛选表（每行的底是名字前缀）。

### 4.5 S 门（4 场景 × 50 步 × 2 批，全部 eager，候选 `amax=delayed`；ViT-B 表里的 B3x / B4x 是 `amax=exact` 对照）

R0、FINAL、候选在同一进程里（ViT-L：schedule / compressed 一个进程、fixed_mid / fixed_late 一个进程；ViT-B：B3 / H3 两个进程，B4 / B3x / B4x 另外两个进程；**不同进程的批不同**，所以 B3 / H3 与 B3x / B4x 之间不是配对比较，B4 对 B4x 才是）；EMA(0.9) 损失偏差对 R0 和对 FINAL 都要 ≤5%，不能有 NaN / Inf；grad-norm 比值记录（门要求 0.1–10）。“检查点”= 每 10 步，“全 50 步”= 逐步取最大。

#### vitl16：S 门（4 场景 × 50 步 × 2 批，全部 eager，amax=delayed）

| 场景 | 候选 | EMA 偏差 vs R0（检查点 / 全 50 步） | EMA 偏差 vs FINAL（检查点 / 全 50 步） | 有限 | grad-norm 比值(对 R0) 范围 | delayed amax：超 2× 次数 / 延迟调用数，最大 cur/prev | 通过 |
|---|---|---|---|---|---|---|---|
| schedule | B3 | 0.0032 / 0.0032 | 0.0028 / 0.0028 | True | 1.001–1.068 | - | 通过 |
| schedule | H3 | 0.0029 / 0.0029 | 0.0023 / 0.0023 | True | 0.998–1.008 | - | 通过 |
| schedule | B4 | 0.0029 / 0.0029 | 0.0023 / 0.0023 | True | 0.999–1.009 | - | 通过 |
| schedule | (FINAL 对 R0 参考) | 0.0007 (全 50 步) | - | True | - | - | - |
| compressed | B3 | 0.0017 / 0.0018 | 0.0019 / 0.0019 | True | 1.008–1.064 | - | 通过 |
| compressed | H3 | 0.0014 / 0.0016 | 0.0013 / 0.0013 | True | 0.992–1.002 | - | 通过 |
| compressed | B4 | 0.0014 / 0.0017 | 0.0013 / 0.0013 | True | 0.996–1.051 | - | 通过 |
| compressed | (FINAL 对 R0 参考) | 0.0010 (全 50 步) | - | True | - | - | - |
| fixed_mid | B3 | 0.0021 / 0.0022 | 0.0020 / 0.0020 | True | 0.998–1.069 | - | 通过 |
| fixed_mid | H3 | 0.0021 / 0.0022 | 0.0012 / 0.0013 | True | 0.997–1.008 | - | 通过 |
| fixed_mid | B4 | 0.0019 / 0.0021 | 0.0011 / 0.0011 | True | 0.991–1.004 | - | 通过 |
| fixed_mid | (FINAL 对 R0 参考) | 0.0010 (全 50 步) | - | True | - | - | - |
| fixed_late | B3 | 0.0029 / 0.0029 | 0.0021 / 0.0021 | True | 0.998–1.105 | - | 通过 |
| fixed_late | H3 | 0.0027 / 0.0027 | 0.0014 / 0.0014 | True | 0.997–1.011 | - | 通过 |
| fixed_late | B4 | 0.0027 / 0.0027 | 0.0014 / 0.0015 | True | 0.996–1.009 | - | 通过 |
| fixed_late | (FINAL 对 R0 参考) | 0.0014 (全 50 步) | - | True | - | - | - |


#### vitb16：S 门（4 场景 × 50 步 × 2 批，全部 eager，amax=delayed）

| 场景 | 候选 | EMA 偏差 vs R0（检查点 / 全 50 步） | EMA 偏差 vs FINAL（检查点 / 全 50 步） | 有限 | grad-norm 比值(对 R0) 范围 | delayed amax：超 2× 次数 / 延迟调用数，最大 cur/prev | 通过 |
|---|---|---|---|---|---|---|---|
| schedule | B3 | 0.0066 / 0.0066 | 0.0072 / 0.0072 | True | 0.998–1.028 | - | 通过 |
| schedule | H3 | 0.0065 / 0.0065 | 0.0072 / 0.0072 | True | 1.000–1.022 | - | 通过 |
| schedule | B4 | 0.0061 / 0.0061 | 0.0065 / 0.0065 | True | 0.991–1.043 | 74 / 10192，4.32 | 通过 |
| schedule | B3x | 0.0062 / 0.0062 | 0.0066 / 0.0066 | True | 0.993–1.043 | n/a | 通过 |
| schedule | B4x | 0.0062 / 0.0062 | 0.0066 / 0.0066 | True | 0.994–1.044 | n/a | 通过 |
| schedule | (FINAL 对 R0 参考) | 0.0008 (全 50 步) | - | True | - | - | - |
| compressed | B3 | 0.0063 / 0.0063 | 0.0068 / 0.0068 | True | 0.994–1.048 | - | 通过 |
| compressed | H3 | 0.0063 / 0.0063 | 0.0068 / 0.0068 | True | 0.996–1.040 | - | 通过 |
| compressed | B4 | 0.0074 / 0.0074 | 0.0069 / 0.0069 | True | 0.999–1.018 | 218 / 10192，10.15 | 通过 |
| compressed | B3x | 0.0074 / 0.0074 | 0.0069 / 0.0069 | True | 0.994–1.018 | n/a | 通过 |
| compressed | B4x | 0.0074 / 0.0074 | 0.0069 / 0.0069 | True | 0.999–1.019 | n/a | 通过 |
| compressed | (FINAL 对 R0 参考) | 0.0011 (全 50 步) | - | True | - | - | - |
| fixed_mid | B3 | 0.0058 / 0.0058 | 0.0058 / 0.0058 | True | 0.999–1.025 | - | 通过 |
| fixed_mid | H3 | 0.0056 / 0.0056 | 0.0057 / 0.0057 | True | 0.999–1.026 | - | 通过 |
| fixed_mid | B4 | 0.0059 / 0.0059 | 0.0079 / 0.0079 | True | 0.997–1.031 | 82 / 10192，4.40 | 通过 |
| fixed_mid | B3x | 0.0060 / 0.0060 | 0.0079 / 0.0079 | True | 0.996–1.030 | n/a | 通过 |
| fixed_mid | B4x | 0.0060 / 0.0060 | 0.0079 / 0.0079 | True | 0.997–1.030 | n/a | 通过 |
| fixed_mid | (FINAL 对 R0 参考) | 0.0021 (全 50 步) | - | True | - | - | - |
| fixed_late | B3 | 0.0062 / 0.0062 | 0.0060 / 0.0060 | True | 1.000–1.029 | - | 通过 |
| fixed_late | H3 | 0.0061 / 0.0061 | 0.0059 / 0.0059 | True | 1.000–1.023 | - | 通过 |
| fixed_late | B4 | 0.0064 / 0.0064 | 0.0082 / 0.0082 | True | 0.971–1.020 | 78 / 10192，4.66 | 通过 |
| fixed_late | B3x | 0.0066 / 0.0066 | 0.0085 / 0.0085 | True | 0.970–1.018 | n/a | 通过 |
| fixed_late | B4x | 0.0064 / 0.0064 | 0.0083 / 0.0083 | True | 0.972–1.021 | n/a | 通过 |
| fixed_late | (FINAL 对 R0 参考) | 0.0019 (全 50 步) | - | True | - | - | - |


**读数**：(1) 所有候选所有场景都过：EMA 偏差最大值（对 R0 / 对 FINAL）：ViT-L 0.32% / 0.28%，ViT-B 0.74% / 0.85%（门 5%），无 NaN / Inf，grad-norm 比值 0.97–1.11。(2) **偏差几乎全部来自 head 前向 mx8（H3 与 B3 / B4 的偏差相同）**：ViT-B H3 0.65% 对 B3 0.66%，ViT-L H3 0.29% 对 B3 0.32% / B4 0.29%；骨干 nv4 dgrad+wgrad 在 50 步里没有给 EMA 增加可见的偏差。FINAL 自己对 R0 的偏差只有 0.02%–0.14%。(3) **B3 的 grad-norm 比值在 ViT-L 上会离开 1**：schedule 步 40 为 1.068、compressed 步 50 为 1.064、fixed_mid 1.069、fixed_late 1.105；B4（nofirst）同场景 ≤1.009（compressed 步 50 一次 1.051），H3 ≤1.011——block 0 的 massive activation 在 B3 里确实把梯度范数推高了 5–10%，nofirst 去掉了大部分。ViT-B 没有这个现象（B3 最大 1.048）。

## 5. 正确性

- fp4_sim 对 fp4_ref 逐位一致，空策略对 r17 块 / r14 head 的一致性、prefix 对 R20a 的复现见 §4.1。
- 本轮没有改任何 Parameter / state_dict / 训练状态（`apply_patches` 在应用 patch 前后检查了 state_dict 键与形状），所以没有 export / resume；graph_equiv 不适用（仿真量化器是 eager 的 torch 算子，且 R20b 不发布栈）；这些检查属于 R22 / R23。
- B 门里 FINAL 在候选前后各跑一次（`FINAL` / `FINAL_again`），余弦对 FINAL 恒为 1.0000（逐位的 GEMM 和 r17 dx 的原子累加不确定性 ≤1e-5），所以候选间的差别不是运行间漂移。
- 逐参数最小余弦在 FINAL 自己身上就有很低甚至为负的值（例如 ViT-L 的 FINAL 对 R0 逐参数最小 −0.64）：这是梯度极小的参数（`qkv.bias` 的 k 部分被掩码、LN 偏置等）的噪声，对全局梯度没有贡献；按 Amendment 4 / 6 只记录不作门。

## 6. 静态分析与 roofline

不适用（没有写 kernel）。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

**门结果**见 §4.3（决赛）和 §4.5（S 门）；筛选全表见 §4.2。

**1. 推荐 POLICY（分模型）**

| 模型 | 骨干 + head（最激进的推荐） | 仅 head 退路 | 估计节省（R20a，非实测） |
|---|---|---|---|
| ViT-L | `head.f=mx8,head.dw=nv4+sr,all.d=nv4+sr,all.w=nv4+sr+rht,nofirst,amax=delayed` | `head.f=mx8,head.dw=nv4+sr,amax=delayed` | B4 18.07 ms（17.8%）；H3 10.54 ms（10.4%）；（B3 18.4，B5 16.56） |
| ViT-B | `head.f=mx8,head.dw=nv4+sr,all.d=nv4+sr,all.w=nv4+sr+rht,amax=delayed` | `head.f=mx8,head.dw=nv4+sr,amax=delayed` | B3 14.37 ms（16.6%）；H3 10.09 ms（11.6%）；（B4 14.02，B5 13.41） |

**ViT-B 的余量**：最差的一批（seed 5，FINAL 自己对 R0 也只有 0.9824）上 B3 / B4 / B5 对 P1 的余弦为 0.9746（放宽门最小 ≥0.97，余量约 0.005）；中位数 0.9884（放宽门 0.98）。ViT-B 的骨干 FP4 在最差批上离放宽门线不远，R22 的真 kernel 门要盯这一项。**批次噪声**：数据加载器跨进程不确定，同一个模型三次独立抽到的 8 批上 ViT-B B4 的对 R0 中位数分别为 旧混合进程 0.9905，决赛单进程 0.9884，对照臂单进程 0.9919（差到 0.0035），所以 ViT-B 的严格门（中位数 ≥0.99）通不通过取决于抽到的批，不能用它区分 B3 / B4 / B5；放宽门在三次抽批里都通过。

选择规则（11.6）：SR 和 RN 都过 → 选 SR；mx4 的判断见第 3 条。其中 ViT-L 的推荐是 B4 而不是字面上“最激进的通过者”B3：**B3 在两个模型上都过放宽门**，但 ViT-L 上 B3 的中位数明显低于 B4 / B5（vitl16：B3 / B4 / B5 中位数 0.9897 / 0.9926 / 0.9926；vitb16：B3 / B4 / B5 中位数 0.9884 / 0.9884 / 0.9886；ViT-L 上 B3 过不了严格门），S 门里 grad-norm 比值到 1.06–1.11；B4 多花约 0.33 ms（≈0.3%）就严格门通过、grad-norm 回到 1.00。如果协调者要按字面取最激进者，就是 `head.f=mx8,head.dw=nv4+sr,all.d=nv4+sr,all.w=nv4+sr+rht,amax=delayed`（两个模型都通过放宽门和 S 门）。

**2. prefix 与 nofirst 的取舍**

| | 精度（B 门，≥8 批，对 R0 中位 / 最小） | 成本（R20a 估计，ms/步 L / B） | 备注 |
|---|---|---|---|
| B3（都不用） | ViT-L 0.9897 / 0.9845；ViT-B 0.9884 / 0.9839 | 0 | ViT-L 严格门不过；S 门 grad-norm 比值最高 1.105 |
| B4（nofirst） | ViT-L 0.9926 / 0.9892；ViT-B 0.9884 / 0.9840 | 0.33 / 0.36（③ 的 1/24、1/12 的 GEMM 节省没了；block 0 回到 FINAL 的 mx8） | ViT-L：与 B5 同精度，S 门 grad-norm 正常 |
| B5（prefix） | ViT-L 0.9926 / 0.9904；ViT-B 0.9886 / 0.9841 | 1.84 / 0.96（侧 GEMM 实测，加法并进 epilogue 之前） | 精度和 B4 相同，成本是 nofirst 的 5.6 倍（L）/ 2.7 倍（B） |

- 在全局梯度余弦上 **prefix 和 nofirst 基本无法区分**（vitl16：B3 / B4 / B5 中位数 0.9897 / 0.9926 / 0.9926；vitb16：B3 / B4 / B5 中位数 0.9884 / 0.9884 / 0.9886），nofirst 的成本只有 prefix 的约 1/6（L）/ 约 1/3（B），所以取舍是：**ViT-L 用 nofirst（B4）；ViT-B 两者都不需要（B3）**。
- prefix 的优点只在块级单步测试里可见：block 0 / joint 块的 wgrad 误差明显下降（见 §4.1 d、f），但这一点没有转化为全局梯度的可见差别；prefix 还会把 block 0 以外的所有块的 5 / 105 个 token 都走 bf16 侧 GEMM，需要额外 kernel。如果之后的 R22 验证里 block 0 仍是问题，nofirst 是可以立即切换的退路，prefix 不必实现。
- 局限：B 门只看全局梯度余弦；逐参数最小余弦在 FINAL 上也低（噪声），不能用来区分。

**3. 格式与对照臂**（§4.4 的 ≥8 批 B4 底对照臂 + 下面的 3 批筛选明细）：vitl16（对 R0 中位/最小）：B4 0.9932/0.9817（放宽门通过）；mx4 0.9921/0.9796（放宽门通过）；RN 0.9933/0.9818（放宽门通过）；w1d 0.9934/0.9819（放宽门通过）；不用 rht 0.9932/0.9816（放宽门通过）；vitb16（对 R0 中位/最小）：B4 0.9919/0.9802（放宽门通过）；mx4 0.9874/0.9758（放宽门通过）；RN 0.9923/0.9803（放宽门通过）；w1d 0.9928/0.9811（放宽门通过）；不用 rht 0.9921/0.9800（放宽门通过）。
  - vitl16 筛选 n=3，底 B3（对 R0 中位/最小）：本身 0.9899/0.9845；mx4 0.9796/0.9783(不过)；RN 0.9904/0.9865；w1d 0.9886/0.9774；不用 rht 0.9924/0.9913；fc.f=nv4 0.9681/0.9552(不过)；qkv.f=mx8 0.9753/0.9690(不过)
  - vitl16 筛选 n=3，底 B4（对 R0 中位/最小）：本身 0.9924/0.9914；mx4 0.9917/0.9905；RN 0.9924/0.9915；w1d 0.9926/0.9916；不用 rht 0.9925/0.9913；fc.f=nv4 0.9797/0.9401(不过)；qkv.f=mx8 0.9886/0.9859
  - vitl16 筛选 n=3，底 B5（对 R0 中位/最小）：本身 0.9925/0.9914；mx4 0.9923/0.9907；RN 0.9926/0.9915；w1d 0.9927/0.9916；不用 rht 0.9925/0.9914；fc.f=nv4 0.9769/0.9653(不过)；qkv.f=mx8 0.9918/0.9884
  - vitb16 筛选 n=3，底 B3（对 R0 中位/最小）：本身 0.9886/0.9784；mx4 0.9835/0.9747；RN 0.9891/0.9786；w1d 0.9898/0.9790；不用 rht 0.9889/0.9782；fc.f=nv4 0.9630/0.9367(不过)；qkv.f=mx8 0.9884/0.9754
  - vitb16 筛选 n=3，底 B4（对 R0 中位/最小）：本身 0.9886/0.9783；mx4 0.9843/0.9750；RN 0.9890/0.9788；w1d 0.9897/0.9790；不用 rht 0.9888/0.9785；fc.f=nv4 0.9627/0.9449(不过)；qkv.f=mx8 0.9869/0.9749
  - vitb16 筛选 n=3，底 B5（对 R0 中位/最小）：本身 0.9890/0.9780；mx4 0.9849/0.9756；RN 0.9894/0.9788；w1d 0.9904/0.9792；不用 rht 0.9890/0.9782；fc.f=nv4 0.9636/0.9368(不过)；qkv.f=mx8 0.9892/0.9757
- **nv4 对 mx4**：ViT-B 上 mx4 低 0.004–0.005（仍过放宽门）；ViT-L 上 mx4 在 B3 底（block 0 用 FP4）**不过放宽门**（0.9796 < 0.98），在 nofirst / prefix 底只低 0.0001–0.0011（n=8 的 B4 底：0.9932 → 0.9921，仍过放宽门）。mx4 的损失集中在 block 0 的 massive activation 上。按 R20a 估计 mx4 整步只快 0.5 / 0.4 ms（amax 由生产者给出时；delayed amax 下 nv4 没有额外的 amax 遍历），低于 2% 发布线。11.6 的字面规则是“mx4 也过门并且整步更快才改用 mx4”：**按字面规则 mx4 有资格（B4 底 n=8 两个模型都过放宽门：ViT-L 0.9921、ViT-B 0.9874），但收益 ≈0.5%、精度更低，建议保持 nv4**，由协调者定。
- **SR 对 RN**：两者都过；单步 B 门里 RN 反而略高 0.0000–0.0006（SR 的噪声是无偏的，单步余弦看不出它的长期好处；S 门只测了 SR）→ 按 11.6 选 SR。
- **w2d 对 w1d**：差在噪声内（ViT-B w1d +0.0009–0.0014（n=8 的 B4 底 +0.0009）；ViT-L B3 底 −0.0013，B4 / B5 底 +0.0002）→ 保持 w2d（W 与 Wᵀ 数值一致，量化器一次出两份）。
- **wgrad 的 RHT**：ViT-L B3 底去掉 rht 反而高 0.0025（0.9899 → 0.9924，RHT 在 block 0 的 outlier token 上有害，R20a 已见），B4 / B5 底 ±0.0001；ViT-B ±0.0003。B 门里 **RHT 没有可见收益**（§4.4 的 ≥8 批 B4 底对照臂：去掉 rht 的差别 ≤0.0002），有差别的地方是它在有害 → R21 的量化器可以先不做 RHT（更简单）；不带 rht 的等价 POLICY 见 results.json 的 `POLICY_no_rht_variant`。
- **仅记录项**：`fc.f=nv4` 在 6 个底上全部**不过**放宽门（对 R0 中位 0.963–0.980，最小 0.937–0.965，ViT-B 的 loss_rel 到 1.2e-2）；`qkv.f=mx8` 在 5/6 个底上过（ViT-B 三个底 0.987–0.989；ViT-L 的 B3 底 0.9753 不过，B4 / B5 底 0.989 / 0.992），只有 n=3，信号不稳；两项都按 R20a 规则不发布。

**4. delayed amax**（候选 B4 / B3 的 delayed 与 B3x / B4x 的 exact 的 S 门结果见 §4.5；逐行诊断在 `results.json["r20b"]["delayed_amax_diag"]`）
- **delayed 对 exact（ViT-B，对 R0 的 EMA 偏差，检查点最大）**：schedule B4 delayed 0.0061 / B4x exact 0.0062；compressed B4 delayed 0.0074 / B4x exact 0.0074；fixed_mid B4 delayed 0.0059 / B4x exact 0.0060；fixed_late B4 delayed 0.0064 / B4x exact 0.0064。
- **headroom 诊断（ViT-B，B4，4 个场景合计，每个场景 50 步，208 个调用点）**：延迟调用 40768 次，当前 amax 超过 2× 上一次 amax（即该张量里大于 6·448·ts 的元素被饱和裁剪）的有 452 次（1.11%；按场景 0.7%–2.1%），cur/prev 最大 10.15。比值 >2 的调用点按类型（dr / dc = dY 的行 / 转置副本，xr / xc = X 的行 / 转置副本，W* = 权重），只统计保存了全部 >2 调用点的场景 fixed_mid（41 个）、fixed_late（38 个）：dr 47，dc 32。**这些场景里超过 2× 的全部是梯度 dY 的副本，X 和权重一次也没有超过。**
- 结论：在 S 门的 50 步里 **delayed 与 exact 没有可见差别**（EMA 偏差相同到 ±0.0001，无 NaN / Inf，grad-norm 比值正常），所以 delayed amax 在这个门上是安全的，B 门（单步）本来就等同于 exact。
- **但 2× headroom 对梯度 dY 不够**：1.11% 的延迟调用当前 amax 超过 2× 上一次，最大 10×，这些张量里最大的几个元素会被饱和裁剪；S 门（2 批 × 50 步）没有显示这个裁剪的后果，但它不能代表长训练里 lr 大步更新或梯度突发的情况。**建议 R21 / R22**：dY 的 amax 由生产者（r13 的 LN-bwd / gamma_prep / head 的 bwd 等 kernel）用 atomicMax 给出 exact 值（这些生产者是我们自己的 kernel，可以改），X 与权重用 delayed（权重变化慢；X 的生产者里 flash attention 的输出 `o` 不能改，只能 delayed），并在量化 kernel 里输出 cur/prev 比值作为运行时饱和监控。
- **局限**：S 门只有 2 个批、50 步；诊断没有包含 ViT-L（诊断代码在 ViT-L S 门之后才加入）；没有对 `amax=exact` 与 `delayed` 做 B 门以外的长轨迹比较。


**5. 失败项**
- 无 autocast 的 fp32 默认块 SDPA 报 `No available kernel`（`RuntimeError`），所以块级合理性测试（§4.1 f）的参考改成 R0 默认块在 bf16 autocast 下的结果；第一版测试曾用 `all.fdw=bf16` 的块作参考，已被这版覆盖。
- 一次启动时策略串里的 `fc.f=nv4`、`qkv.f=mx8` 作为候选名被当成“名字=策略”解析，报 `unknown policy token 'nv4'`，已把候选名改成 `fcF4 / qkvF8` 并重跑（旧 json 已删除重建，没有留下污染的数据）。
- ViT-L 筛选的第二次启动在 seed 1 OOM：`Handle.ctx` 引用 ctx，局部变量 `handle` 在下一个 seed 之前还活着，钉住了 ViT-L 约 6 GB 的模型和优化器状态；已在 seed 循环结尾 `handle.ctx = None; del handle`，重跑通过。
- 三次 `assert_gpu_exclusive`：另一个代理的 GPU 进程在初始化 CUDA 期间启动，我的进程被中止；已加 `retry.sh`（等待并重试）。
- 数据加载器在不同进程里对同一个 seed 给出的 batch 不完全一致（R0 loss 不同），所以旧的混合进程 ViT-B finals 里同一个 seed 的不同候选不是同一批；已用单进程重跑 ViT-B finals，旧文件保留为 `gateB_r20b_vitb16_finals_mixedproc.json`。

**6. 需要协调者决定**
1. ViT-L 取 B4（nofirst，推荐）还是字面上的最激进者 B3；
2. mx4 是否因为“过门且更快”而改用（我建议不）；
3. 是否要为 R22 保留 `nofirst` 作为运行时选项（成本 0.3 ms，不需要额外 kernel），以及是否实现 prefix（我建议不）；
4. wgrad 的 RHT 是否取消（对照臂显示没有可见收益，R21 的量化器可以先不做 RHT 以降低复杂度）；
5. R21b 的 head 策略取 H3（含 `head.f=mx8`）还是 H1（仅 dgrad+wgrad nv4）：H3 多 3.3 / 3.05 ms，B 门严格通过，S 门通过；
6. ④ / ⑤a 的“破例”：本轮的 B 门数据显示 fc.f=nv4 不过放宽门（不建议破例）；qkv.f=mx8 在 ViT-L 的 B3 底上不过、其余底上过，信号不稳（n≤3），不建议。

**7. 下一瓶颈**：骨干 nv4 的 amax 要由生产者给出（delayed amax 已经在仿真里证明不需要额外遍历）、双输出 FP4 量化器的带宽（R21a）；block 0 的 massive activation（ViT-L）要靠 nofirst 或 prefix 绕开；整步里 attention / LN / 逐元素（45 ms）仍不在范围内。

**契约变更请求**：(1) 11.4 / 6a 12.3 的策略串里，选项修饰“紧跟的条目”：`all.w=nv4+sr+rht` 的 `sr`、`rht` 作用于该条目，`prefix`、`nofirst`、`amax=` 是全局选项——已按此实现，请 6b 明确写入；(2) `amax=delayed` 的诊断量（cur/prev 最大比值、超 2× 次数）建议在 R21 的量化 kernel 里也输出，用于运行时发现饱和；(3) 11.5 B 门：数据加载器跨进程不确定，建议明确“批次数按（进程、seed）计，比较只在同一进程内配对”（Amendment 4 已有类似说明）。

