# Round 10：`round10_lowp_probe`（GEMM 侧低精度实测 + 仿真 + go/no-go）

> 状态：**进行中（2026-10-04，被会话额度打断过一次，已续做）**。本文件由 `NOTES_template.md` + `tables.py` 生成（`python build_notes.py`），表格数字直接来自 json / jsonl。
> 范围：骨干 / head 的 GEMM。attention、head 小 kernel、prep/post 归 round11，不碰。
> 数据文件：`results.json`（峰值、nsys、仿真检查、门 A 测试）、`results_gemm*.json`（GEMM 速度矩阵）、`results_quant.json`（量化器开销）、`sweeps/*.jsonl`（精度仿真）、`profile/`。
> 一切计时：CUDA graph 回放，候选交替，3 轮取中位数，持双锁（infer-opt + train-opt）。没测的数一律写 not measured。

## 0. 续做指南（被打断后看这里）

已完成（下文有数）：峰值 §1、r5_v1 nsys 基线 §2、cuBLAS fp16-acc §3.1、`mg_f16acc` 原型 §3.2、低精度 GEMM 候选 §3.3、量化器开销 §3.4、`mg_i8` 原型 §3.5、quant_ref §4.1、仿真框架 §4.2、fp16 溢出扫描 §4.3、精度扫描结果 §4.4。
未完成：见各节末的"待做"，以及 §5 决策、§6 给 R12 的建议。重跑方式：`bash run_locked.sh LOG bash batchX.sh ...`（`batchB*.sh` GEMM 矩阵、`batchC*.sh` 原型 tile、`batchI1.sh` int8 原型 tile、`batchS3.sh MODEL SECONDS GRID...` 精度扫描，全部可续做：已完成的条目会被跳过）；`python build_notes.py` 由 `NOTES_template.md` 重新生成本文件里的表。

脚本一览（都在本目录；`deploy.sh` 把本地 scratchpad 的文件 scp 到这里）：

| 文件 | 作用 |
|---|---|
| `run_locked.sh LOG cmd...` | 双锁（infer-opt → train-opt）下运行命令，输出到 LOG，末行 `EXIT n` |
| `envsetup.sh` | 批处理脚本的环境（PATH、PYTHONPATH 含 train-opt、QUACK_CACHE_DIR 与扩展构建目录都在 infer-opt-tmp/round10_lowp_probe） |
| `peak_extra.cu`, `peaks.sh` | 指令速率（bf16/fp16 f32acc、fp16 f16acc、int8、plain e4m3）+ Spark `bench_peak.cu` 原样编译 |
| `nsys_run.sh`, `nsys_groups.py`, `nsys_cat.py` | r5_v1 整帧 graph 的 nsys（`bench9.py --profile r5_v1`）与每类 kernel 的每帧时间 |
| `impl/bench10.py` | GEMM 计时框架：每个候选 R 份独立权重（R×权重字节 ≥ ~192 MB > L2），一张 graph 里依次调用，激活 L2 热、权重冷；按 graph 时间 / R 报告 |
| `gemm_matrix.py` | 速度矩阵入口（`--cands`、`--Ms`、`--extra` 切到 patch-embed / ConvT1 / ConvT2 / 掩码 einsum 形状） |
| `impl/cands_base.py` | r5_v1（带 epilogue）、cuBLAS bf16 / fp16 / fp16-acc |
| `impl/mg16_core.cu`, `mg16b.py`, `mg16l.cu`, `mg16.py` | **`mg_f16acc` 原型**（R5 `mg_core.cu` 的拷贝，见 §3.2） |
| `tune16.py` | 原型的 tile 扫描（gen/gen2/compile/stage1/stage2） |
| `impl/cands_mx8.py` | cuBLASLt MXFP8、QuACK mx8（r17，含 epilogue）、Spark fp8gemm_mx v1/v2、`torch._int_mm` |
| `impl/cands_fp4.py` | cuBLASLt / QuACK（r20，含 epilogue）/ Spark / QuTLASS 的 NVFP4、MXFP4 |
| `impl/cands_epi.py`, `impl/epi_kernels.cu` | 给"没有 epilogue 的候选"补价：独立的 RoPE+布局 / GELU / gamma+残差 / int32 反量化 kernel（带宽型，未调优） |
| `impl/spark_ext.py` | Spark fp8gemm_mx / fp8_quantize / fp4gemm / fp4quant 的 torch 绑定（就地编译，build 在 tmp） |
| `quant_matrix.py` | 激活量化器开销（R6 `ln_quant`/`quant_rows`/`gelu_quant`、r17 `quant_dual`、r24 `fp4q2`、Spark） |
| `impl/quant_ref.py` | **量化参照（R12 门 A 的唯一逐位参照）**：MX8(RCEIL)、NV4、MX4、INT8、FP16 预缩放、块 Hadamard |
| `tests/test_quant_ref.py`, `tests/test_mg16.py` | 门 A：量化器对 r17 CUDA kernel / r20 fp4_ref 逐位一致；原型 GEMM 误差 |
| `impl/emu.py` | **精度仿真**：r5_v1 的逐层复刻，每个 GEMM 可选 bf16 / fp16(原型 kernel) / mx8 / int8 / nv4 / mx4，技术：SmoothQuant、块 Hadamard、前缀行留 bf16、GPTQ |
| `impl/gptq.py` | 块缩放格式的 GPTQ（Hessian 来自 16 个校准帧） |
| `emu_eval.py` | refs / sweep / check：8 帧门 M + 全片门 H（去掉校准帧）+ logit MAE |
| `make_cfgs.py` | 生成扫描配置 `cfgs/*.json`（阶段 A–F） |
| `scan_overflow.py` | fp16 溢出扫描 |

## 1. 峰值（deliverable 1）

方法同 Spark `bench_peak`：寄存器操作数、每 warp 8 条独立累加链、每 SM 4 block × 8 warp、记录循环内 SM 时钟（约 2.84 GHz，GPU 空闲时测，功率 ~480 W）。`peak_extra.cu` 补 Spark 没有的 fp16 / int8 路径；Spark `bench_peak.cu` 原样编译。原始输出 `profile/peaks.log`。

| 路径（`mma.sync`） | 实测 TFLOPS | 对 bf16-f32acc |
|---|---:|---:|
| bf16 输入 / fp32 累加（m16n8k16；现状） | 246.8（复测 247.1；Spark 247.1） | 1.00× |
| fp16 输入 / fp32 累加 | 246.9 | 1.00× |
| **fp16 输入 / fp16 累加** | **492.6** | **1.99×** |
| int8 → s32（m16n8k32） | 983.2 | 3.98× |
| fp8 e4m3 plain / fp32 累加 | 492.8（Spark 493.0） | 2.00× |
| fp8 e4m3 plain / fp16 累加 | 983.5（Spark） | 3.98× |
| **mxf8f6f4 block-scaled MXFP8**（单位 scale） | **981.3**（Spark） | 3.97× |
| nvf4 `kind::mxf4nvf4`（scale_vec::4X） | 1964.4（Spark） | 7.95× |
| mxf4（scale_vec::2X ue8m0） | 1964.6（Spark） | 7.95× |
| tf32（参考） | 123.3 | |
| fp32 FMA（参考） | 118.2 | |

结论：GeForce 限速的是"fp32 累加的 f16/bf16 MMA"（以及 plain fp8 f32acc）；fp16 累加满速（1.99×），block-scaled MXFP8 / FP4 满速。PLAN 的 P1 规则"本机测得的指令速率 ≥ 1.7× bf16"：**1.99×，满足**。

## 2. r5_v1 的 nsys 基线（deliverable 2）

`nsys profile`（`--cuda-graph-trace=node`，`bench9.py --profile r5_v1`：整帧 graph，126 帧 1 遍，去掉首尾各几帧后取每帧均值）；分类脚本 `nsys_cat.py`（GEMM 按每层 `qkv, proj, fc1, fc2` 的出现顺序区分种类，后 3 层为 joint）。原始 CSV 在 `profile/*_cuda_gpu_kern_sum.csv`，分类 JSON 在 `profile/*_categories.json`。**这替换 PLAN_v2 Context 里的估计表。**

| 部分（每帧 GPU kernel 时间） | ViT-L ms | 占比 | ViT-B ms | 占比 |
|---|---:|---:|---:|---:|
| 骨干 GEMM（qkv+proj+fc1+fc2） | **6.809** | 69.9% | **2.094** | 55.3% |
| attention（flash） | 1.450 | 14.9% | 0.725 | 19.1% |
| LayerNorm（add_ln + 末尾 LN） | 0.233 | 2.4% | 0.095 | 2.5% |
| head GEMM（ConvT×2 + 掩码 einsum + 小 GEMM） | 0.605 | 6.2% | 0.309 | 8.2% |
| head `dw_ln` | 0.275 | 2.8% | 0.189 | 5.0% |
| patch-embed 卷积 | 0.132 | 1.4% | 0.097 | 2.6% |
| prep + post | 0.169 | 1.7% | 0.162 | 4.3% |
| 其余（cat/copy/elementwise） | 0.071 | 0.7% | 0.116 | 3.1% |
| **合计（kernel 时间）** | **9.744** | | **3.787** | |
| 首尾 kernel 跨度 | 9.845 | | 3.841 | |

GEMM 细分（µs/帧；joint = 最后 3 层，M=2258）：

| | ViT-L normal | ViT-L joint | ViT-B normal | ViT-B joint |
|---|---:|---:|---:|---:|
| qkv（TileLang `main_kernel`） | 1492.6 | 230.3 | 388.5 | 129.5 |
| proj（TileLang） | 613.1 | 88.5 | 160.8 | 55.0 |
| fc1（`gemm_kernel` MG） | 1837.3 | 285.7 | 482.2 | 169.1 |
| fc2（`gemm_kernel` MG，L normal 为 split-K=2） | 1963.5 | 297.6 | 528.8 | 180.0 |
| head ConvT1 / ConvT2 GEMM（cutlass bf16） | 103 / 351 | | 见 categories.json（合计 195.8） | |
| 掩码 einsum（cutlass 64x256，M=100×2） | 104.0 | | 79.2 | |
| patch-embed `implicit_convolve_sgemm` | 132.0 | | 96.9 | |

要点：PLAN 估计的"骨干 GEMM ~6.6 / ~2.0 ms"基本正确（6.81 / 2.09）；head GEMM 和 dw_ln 合计 0.88 / 0.50 ms，与估计 0.90 / 0.55 一致；LN 0.23 / 0.095；prep+post 实测 0.17 / 0.16（估计 0.14 / 0.13）。

## 3. GEMM 速度矩阵（deliverable 3，进行中）

### 3.1 cuBLAS fp16 + `allow_fp16_accumulation=True`（零代码）与 r5_v1

`F.linear`，M ∈ {2058, 2258}，µs（TFLOPS）。cuBLAS 三列**不含任何 epilogue**；r5_v1 列**含**其融合 epilogue（RoPE+布局 / GELU / gamma+fp32 残差）。来自 `results_gemm.json`（`profile/gemm_base1.log`）。该次运行开始时 `other_gpu_procs()` 为空。

| 形状 | r5_v1（含 epi） | cuBLAS bf16 | cuBLAS fp16（f32acc） | **cuBLAS fp16-acc** | fp16-acc 对 r5_v1 |
|---|---:|---:|---:|---:|---:|
| L qkv 2058 | 70.7 (183) | 69.9 | 69.5 | 45.5 (285) | 1.55× |
| L qkv 2258 | 76.0 (187) | 79.8 | 79.2 | 47.0 (302) | 1.62× |
| L proj 2058 | 29.2 (148) | 26.9 | 27.0 | 16.0 (271) | 1.83× |
| L proj 2258 | 29.6 (160) | 27.1 | 27.1 | 16.4 (289) | 1.80× |
| L fc1 2058 | 86.3 (200) | 102.6 | 102.8 | 63.8 (271) | 1.35× |
| L fc1 2258 | 94.1 (201) | 103.1 | 103.1 | 64.7 (293) | 1.45× |
| L fc2 2058 | 93.5 (185) | 105.6 | 105.8 | 53.9 (320) | 1.73× |
| L fc2 2258 | 99.6 (190) | 105.6 | 105.8 | 55.1 (344) | 1.81× |
| B qkv 2058 | 43.4 (168) | 41.7 | 41.7 | 26.7 (273) | 1.63× |
| B qkv 2258 | 44.2 (181) | 42.3 | 42.1 | 27.2 (294) | 1.62× |
| B proj 2058 | 17.9 (136) | 20.7 | 20.7 | 12.2 (200) | 1.47× |
| B proj 2258 | 18.1 (147) | 20.8 | 20.8 | 12.3 (217) | 1.47× |
| B fc1 2058 | 52.4 (185) | 61.7 | 61.6 | 36.1 (269) | 1.45× |
| B fc1 2258 | 55.0 (194) | 61.9 | 61.9 | 37.2 (287) | 1.48× |
| B fc2 2058 | 56.5 (172) | 61.0 | 60.8 | 39.8 (244) | 1.42× |
| B fc2 2258 | 58.5 (182) | 61.3 | 61.3 | 39.9 (267) | 1.47× |

读法：cuBLAS fp16-acc 在不含 epilogue 的情况下比 r5_v1 快 1.35–1.83×（244–344 TFLOPS，超过 bf16 指令峰值 247）；cuBLAS fp16/bf16 的 fp32 累加版与 r5_v1 同量级（r5_v1 靠手写 tile 和融合 epilogue 持平或略胜）。**fp16 累加是一条不需要量化的 GEMM 路径，上限 ≈ 1.4–1.8×。**

### 3.2 `mg_f16acc` 原型（R5 `mg_core.cu` 的拷贝，改 MMA）

- 源：`impl/mg16_core.cu` = `round5_gemm_sm120/impl/mg_core.cu`（只读，未改）的拷贝，文件头有说明。改动：A/W 为 fp16；MMA 换成 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`；fp16 累加器每 `PKT` 个 k-tile（默认 1，即每 BK=32/64 个 k）加进 fp32 寄存器并清零；Params 增加 `oscale`（2 的幂预缩放的还原，所有 epilogue 里 `acc*oscale`）；EPI 0/2 可输出 fp16；新增 EPI 4（纯 GEMM，bf16/fp16 输出，供 head ConvT、patch-embed、einsum 形状）。R5 的 RoPE+布局 / GELU / gamma+fp32 残差 epilogue 与 split-K 原样保留。`mg16b.py` / `mg16l.cu` 是 `mgb.py` / `mgl.cu` 的拷贝（独立构建目录与扩展名）。
- 正确性（门 A，`tests/test_mg16.py`，ViT-L / M=2058 / 随机数据含大通道，对**自身 fp16 操作数的 fp64 乘积**）：rel L2 qkv 3.3e-4、proj 1.2e-4、fc1 3.4e-4、fc2 1.7e-4，cos ≥ 0.99999994（含 fp16 输出舍入）。对**fp32 原始数据**的误差：fp16-acc 4.4e-4 / 3.9e-4 / 4.6e-4 / 3.9e-4，r5 bf16 kernel 2.9e-3 / 2.3e-3 / 2.9e-3 / 2.3e-3 → **比值 0.15–0.17 ≤ 1.5**（P1 门 A 满足；fp16 操作数比 bf16 多 3 位尾数）。ViT-B 同样（`results.json::mg16_test`）。
- tile 扫描（`tune16.py`）：随机池（bm 64–256、bn 64–256、bk 32/64、st 2–4、warp 布局、ORDER/UNR/FRAGALL/STG、split-K、PKT 1/2）约 620–880 个可编译配置/种类，编译 ~10 min（CPU），timing 用 `bench10`。**第 1 轮（单次计时，`profile/tune16_stage1b.log`；该轮开始时有另一进程在 GPU 上，数字偏保守，第 2 轮会交替复测）** 每个形状的最优，M=2058：

| 形状 | 原型最优 µs（TFLOPS） | r5_v1（§3.1） | 对 r5_v1 | cuBLAS fp16-acc |
|---|---:|---:|---:|---:|
| L qkv | 59.6 (217) | 70.7 | 1.19× | 45.5 |
| L proj | 22.7 (190) | 29.2 | 1.29× | 16.0 |
| L fc1 | 69.8 (247) | 86.3 | 1.24× | 63.8 |
| L fc2 | 69.1 (250) | 93.5 | 1.35× | 53.9 |
| B qkv | 36.9 (197) | 43.4 | 1.18× | 26.7 |
| B proj | 15.0 (162) | 17.9 | 1.19× | 12.2 |
| B fc1 | 41.5 (234) | 52.4 | 1.26× | 36.1 |
| B fc2 | 41.5 (234) | 56.5 | 1.36× | 39.8 |

  （这些最优里若干用了 `pkt=2`，即 fp16 累加跨 64 个 k；其精度要单独验证。）

**第 2 轮（stage2：第 1 轮最优的若干 tile + 同轮 r5_v1 / cuBLAS，交替 3 轮取中位数；`tune16.py --phase stage2`）** —— 全部 tile：

{{mg_stage2:mg16}}

只允许 `pkt=1` 的版本（每个 k-tile 提升到 fp32，fp16 累加长度 = BK）：

{{mg_stage2:mg16_pkt1}}

精度：fp16 档在 126 帧上的门 M/H 见 §4.4（用的就是这些 tile；`pkt=2` 的配置也过了门）。


### 3.2b `mg_i8` 原型（int8 W8A8，s32 累加，融合反量化 + R5 epilogue）

- 源：`impl/mg8_core.cu` = `mg16_core.cu` 的改写（见文件头）：`mma.sync.m16n8k32.s32.s8.s8.s32`，A/W int8，BK 以 bf16 等价元素计（BK=32 → 一个 k-tile 64 个 int8），cp.async/ldmatrix/swizzle 与 16 位 kernel 字节相同；epilogue 里 `acc * sx[row] * sw[col]`（sx = 逐 token 激活 scale，sw = 逐输出通道权重 scale）再接 R5 的 bias(+GELU) / bias+RoPE+布局 / gamma+fp32 残差 / 纯输出；无 split-K。
- 正确性（`tests/test_mg8.py`，对"同一批 int8 操作数和 scale 的精确 fp64 反量化乘积 + epilogue"）：proj / fc2 rel L2 3e-8（epilogue 的 fp32 算术）；qkv / fc1 1.66e-3 —— 这两项是输出 bf16 舍入（qkv/fc1 的输出是 bf16，proj/fc2 是 fp32 残差），cos ≥ 0.9999986。int32 累加是精确的，所以 kernel 本身没有额外误差。
- tile 扫描（同 `tune16.py` 的流程：`tune8.py`，stage1 单次 + stage2 同轮交替 3 轮对 r5_v1 / cuBLAS）：

{{mg_stage2:mg8}}

### 3.3 低精度 GEMM 候选（M = 2058 / 2258，µs（TFLOPS））

`results_gemm_lowp1.json`（`batchB.sh`；每项含 `foreign_gpu_procs` 记录）。**列的含义**：`r5_v1` 含融合 epilogue；`cublas_fp16acc`、`cublaslt_mx8`（cuBLASLt `torch._scaled_mm` MXFP8，bf16 输出）、`spark_mx8_v1/v2`（Spark `fp8gemm_mx`，v2 = TMA + warp specialization，M 补齐到 128，不满足形状要求时 n/a）、`int8_mm`（`torch._int_mm`，int32 输出）**都不含 epilogue、不含量化**；`epi_i32` / `epi_bf16` 是单独一遍带宽型 epilogue kernel 的实测价格（qkv：bias+RoPE+布局；fc1：bias+GELU；proj/fc2：bias+gamma+fp32 残差；int32 版带行列 scale 反量化；未调优，只是量级）。

{{gemm_lowp1}}

读法：cuBLASLt MXFP8 与 `_int_mm` 的 kernel 本体在 380–590 TFLOPS，比 r5_v1 快 2.3–3×、比 cuBLAS fp16-acc 快 1.5–2×，但**都不含 epilogue**（RoPE+布局、GELU、gamma+fp32 残差要另付 `epi_*` 一遍，qkv/fc1 约 12–20 µs，proj/fc2 约 4–6 µs，见上表和 §3.5）；Spark 的 fp8 MX kernel 只有 150–440 TFLOPS，明显不如 cuBLASLt（v2 对 M=2058 等形状不支持，n/a 表示形状不满足 M%128 补齐后的要求或 N/K 对齐要求）。`epi_*` 是 16 次连续调用 / 图的平均（单次调用的版本包含 graph launch 开销，约 +1.3 µs，在 `results_gemm_epi_singlecall.json`）。

### 3.3b QuACK MXFP8（r17，**含 epilogue**）与 FP4 候选（µs（TFLOPS），均不含量化）

QuACK mx8 / quack_nv4 / quack_mx4 含 train-opt r13/r20 的融合 epilogue（qkv 只有 bias，没有 RoPE+布局，需要另付 epi 一遍；fc1 = bias+GELU（还会多写一份 pre-activation z）；proj/fc2 = bias+gamma+fp32 残差（还会多写一份 y））。cuBLASLt 的 MXFP4 在 GeForce 上不支持（`MXFP4 scaling only supported in CUDA for B200/B300`），所以 mx4 只有 QuTLASS / Spark / QuACK 三个来源。

{{gemm_quack_fp4}}

读法：FP4 kernel 本体在这些小 M 形状上只有 400–990 TFLOPS（指令峰值 1964），对 MXFP8 的优势是 1.2–2×（qkv/fc1），proj 因 N=K=d 太小反而与 MXFP8 持平；QuACK FP4 带 epilogue 的 qkv/fc1 最快；Spark FP4 v2（TMA）形状限制多、不稳定。

### 3.3c head ConvT1/ConvT2 GEMM、掩码 einsum、patch-embed GEMM（M=2048 / 2048 / 8192 / 100 per crop；µs）

单个 crop 的形状（head 的 ConvT 是 `torch.mm`，每帧 2 个 crop 合在 M 里：ConvT1 M=2048、ConvT2 M=8192、einsum 是 B=2 个 [100,C]×[C,16384] 的 bmm、patch-embed 是 [2048,768]×[768,C]）。`mg16_tuned` 用的是同形状骨干 GEMM 的 tile（未单独调）。

{{gemm_extra}}

### 3.3d 2 帧合批模式（M = 4116 / 4516）

tile 沿用 M=2058/2258 的调参结果（没有为合批重调；`r5_v1` 的 tile 表在最近邻 M 上取），所以这里的 r5_v1 / mg16 / mg8 数字对合批是偏保守的下界。

{{gemm_b2}}

### 3.4 量化器开销（M = 2058 / 2258；µs / 次，16 次连续调用的平均）

位点：`ln` = LN 之后喂 qkv / fc1（K=d）；`o` = attention 输出喂 proj（K=d）；`gelu` = GELU 输出喂 fc2（K=ff）。`ln_add_r5` 是 R5 现有的 `add_ln`（LN 本身，用于对照：把量化并进 LN 后的增量是量化行的"额外成本"）；`ln_quant_r6` 是 R6 的 LN+MX 量化融合 kernel（读 fp32 残差）；`int8_rows_r10` 是我写的逐 token int8 量化（`epi_kernels.cu`，bf16→int8+scale）；`quant_dual_r17`（MX8 行版，r17）、`spark_fp8mx`（Spark MX8）、`fp4q2_*`（r24 FP4）、`spark_nv4/mx4`（Spark）为独立量化 kernel。

{{quant_costs}}

融合进生产者后的成本估计（**假设值，没有实测**）：LN 之后的量化并进 `add_ln`（本来就要读写整行，增量只有 amax + cvt + 写 1B/元素 + scale，约 1 µs）；GELU 之后的量化并进 fc1 epilogue（需要整行 amax：用 tile 内局部 amax，MX 的 1×32 块天然是 tile 内局部的，所以对 MX8/MX4/NV4 可行，约 1 µs；逐 token int8 需要整行 amax，fc1 的输出行跨多个 N-tile，做不到纯 epilogue，必须另一遍）；attention 输出 / GELU 输出之后的量化若不能融合则用上表的独立 kernel 实测值（"pess"）。per-frame 汇总里 central 用 qkv/fc1 = 1 µs、proj = 2 µs、fc2 = 3 µs，pess 用上表实测的独立 kernel。

### 3.5 逐帧 GEMM 账（由实测矩阵算出，`speed_model.py` / `analyze_gemm.py`）

每帧 = (L−3) × t(M=2058) + 3 × t(M=2258)，L = 24 / 12。"档位 / kernel" 的含义：`r5_v1` 为基线；`fp16` = `mg_f16acc` 原型（带融合 epilogue，tuned16.json 最优 tile）；`fp16_pkt1` = 只允许每个 k-tile 提升（pkt=1）的 tile；`fp16_cublas` = cuBLAS fp16-acc + 单独 epilogue 一遍；`int8` = `mg_i8` 原型（融合反量化+epilogue）；`int8_cublas` = `_int_mm` + 单独反量化/epilogue 一遍；`mx8` / `nv4` / `mx4` = 各候选里（cuBLASLt、QuTLASS、Spark、QuACK；除 QuACK 的 qkv 外含单独 epilogue 价格，QuACK 的 fc1/proj/fc2 用其融合 epilogue）最快者。"合计（无量化）" 里 nv4/mx4/mx8 的 qkv 含单独 epilogue 一遍。

{{perframe}}

要点：
- `fp16`（P1）在 **ViT-L 省约 1.2–1.35 ms/帧（对 6.74 的 GEMM 时间约 −20%），ViT-B 省约 0.39–0.40 ms/帧（−19%）**，不需要量化，精度见 §4.4（fp16 全部通过门 M+H）。
- `int8` / `mx8` 的 GEMM 本体再快一倍，扣掉量化成本后 ViT-L 约省 3.4–3.5 ms、ViT-B 约省 0.9–1.0 ms（central；pess 下略少），**前提是精度能过**——见 §4.4。
- FP4 的 GEMM 账面上最快（ViT-L 省 4.3 ms），但 §4.4 里 nv4/mx4 的骨干 GEMM 全部过不了门，只有 head 能用。

## 4. 精度仿真（deliverable 4，进行中）

### 4.1 `impl/quant_ref.py`（R12 门 A 的参照）——逐位一致

`tests/test_quant_ref.py`（`results.json::quant_ref_test`）：3 个形状 × 4 种数据（正态 / 离群通道 / 宽动态范围 / 含全零块）。
- MXFP8：e4m3 字节与 UE8M0 scale 字节对 train-opt r17 `quant_dual`（CUDA kernel，行拷贝）**逐位一致**；scale 规则 RCEIL：`byte = clamp(e + (frac≠0), 255)`，`e,frac` 取 `fl32(amax/448)`，`v = x·2^-(byte-127)` → `cvt.rn.satfinite.e4m3`；全零块 scale 字节 0、数据 0。
- NVFP4 / MXFP4：packed e2m1 码、blocked scale 字节、per-tensor ts 对 r20 `fp4_ref.quantize_matrix` **逐位一致**（nv4：e4m3 scale per 16 + fp32 ts=amax/(6·448)；mx4：ue8m0 per 32 RCEIL）。
- INT8（对称，按 token / 按输出通道，scale=amax/127，round-half-even，钳制 ±127）、FP16（`x·2^-p` 的 RN）、块 Hadamard（自然序 FWHT ×1/√b，b=16/32/64）：纯参照，无对应 kernel。
- 附带的量化误差水平（相对 L2，正态数据，K=1024）：mx8 2.7e-2、nv4 9.5e-2、mx4 1.15e-1、int8（per-token）7.9e-3；离群通道数据 int8 退化到 5e-2–1e-1，nv4/mx8 基本不变。

### 4.2 仿真框架

`impl/emu.py::EmuNet` 复刻 `make_model_fn_r5(..., "r5_v1")`（同一批 R5 kernel：LN kernel、qkv+RoPE、flash SDPA、proj/fc2 gamma+fp32 残差、fc1 GELU、R7 融合 head），每个 (层, 种类) 一个 `Spec(fmt, sq, had, pfx, gptq, wfmt, pa, pw)`。低精度格式：X、W 做"量化→反量化成 bf16"后走**同一个 R5 kernel/epilogue**（MX8/NV4/MX4 的值在 bf16 里精确可表示；nv4 的 per-tensor scale 与 int8 的 scale 在 bf16 里舍入，误差远小于量化误差，未单独处理）；fp16 档走真的 `mg_f16acc` kernel。SmoothQuant：qkv/fc1 折进 LN 的 γ/β 和 W，proj 折进 W_v 行 + W_proj 列（fc2 输入经 GELU，不能折）。前缀行 = 每个 crop 的前 5 行（joint 层前 105 行），留 bf16 路径时同时算两遍再按行合并。

- **检查**：P0 仿真（全 bf16）对 `make_model_fn_r5` 输出 `mask_logits` 与 `class_logits` 逐位相同（ViT-B，max_abs 0.0）。
- 参考：`emu_eval.py refs` 生成 126 帧 fp32 eager 参考：原始路径（PIL prep + 原 post）的 mask / conf（用于门 M/H；其中 8 个评测帧与已缓存的 `ref/` **完全一致**，conf 差 0.0）+ fast 路径的 (2,512,675) fp32 logits（用于 logit MAE / cos）。文件在 tmp：`ref126_{vitl16,vitb16}.pt`。
- 评测帧 `EVAL_IDX = round(i·125/7)` = 0,18,36,54,71,89,107,125；校准帧 `CAL_IDX`：16 帧均匀取自非评测帧（emu_eval.py 内计算）；**门 H = 其余 110 帧**（含 8 个评测帧），判据同门 M（pooled IoU ≥ 0.95，每帧 conf 相对误差 ≤ 5%），另报 min-frame IoU、conf>3% 的帧数、logit MAE。
- P0 基线（ViT-B，仿真）：门 M IoU 0.9943 / conf 0.21%；门 H（110 帧）IoU 0.9931 / min-frame IoU 0.980 / conf 最大 0.43% / conf>3% 帧数 0 / logit MAE 1.45e-5。（ViT-L 的 P0 检查与基线 **not measured yet**。）

### 4.3 fp16 溢出扫描（`scan_overflow.py`，4 个校准帧，P0 真实激活，`profile/overflow_scan_*.json`）

对每层每个 GEMM 算：每 16/32/64-k 块的 max|部分和|、绝对和上界（max Σ|x||w|，是该块内任何 fp16 运行部分和的上界）、max|输出|、前缀行 / 非前缀行分开，以及让上界 ≤ 2^14 所需的 2 的幂预缩放 p。

- **ViT-L 96 个 GEMM、ViT-B 48 个 GEMM，无一溢出**：32-k 和 64-k 块的绝对和上界全部 < 65504，所需预缩放 **p = 0（100% 的 GEMM）**，前缀行也不例外（"只有前缀行是违规者"的 GEMM 数为 0）。
- 最大值（块长 32）：ViT-L 绝对和上界最大 2.07e3（L0.fc2，前缀行）、部分和最大 2.06e3；输入 |x| 最大 1.98e3（L0.fc2 前缀行）；输出 |out| 最大 5.18e4（L0.fc2 前缀行，累加在 fp32，不受 fp16 限制）。ViT-B 绝对和上界最大 421（前缀行），输入 |x| 最大 1.13e3（前缀行）。非前缀行：绝对和上界最大 28.1（ViT-L）/ 38.0（ViT-B），|x| 最大 45 / 68；64-k 块的上界最大 2.79e3（ViT-L 前缀行）/ 421（ViT-B 前缀行）。
- 含义：massive activation（残差 ~1.5e5）在 LN 之后已经被归一，fc1 输出 / GELU 输出 / attention 输出都远小于 fp16 上限 → **P1 不需要预缩放，不需要排除前缀行**（`oscale` 与前缀处理保留为开关，目前不用）。PLAN 的 P1 溢出条款（"≥ 90% 的 GEMM 能找到安全预缩放"）：100%，满足。注：只扫了 4 个校准帧；其余帧的最大值在 §4.4 的整片精度运行中间接检验（出现 inf/NaN 会直接让门失败）。

### 4.4 精度扫描：门 M / 门 H（8 帧筛选 → 126 帧）

扫描脚本 `emu_eval.py sweep`（配置由 `make_cfgs.py` 生成，结果 `sweeps/*.jsonl`，每行一个配置，含每帧 IoU / conf 误差 / MAE）。一个配置的 126 帧完整评测在 ViT-B 上约 1.3 s、ViT-L 约 2.3 s（仿真是 eager，不做 CUDA graph）。**ViT-L 的 conf 指标是混沌的**（R6 的结论在这里也成立：同一精度档位下把前缀行留 bf16 反而可能把某帧 conf 从 2% 推到 30%），因此下面的"通过"是在这一段视频上的实测，选择有"挑配置"的偏差；判断优先看 logit MAE（平滑指标），换视频前需要重新验证。

P0（r5_v1，仿真逐位等价）的基线：ViT-B 门 M IoU 0.9943 / conf 0.21%，门 H（110 帧）IoU 0.9931 / min-frame 0.980 / conf 最大 0.43% / MAE 1.45e-5；ViT-L 门 M IoU 0.9946 / conf 0.13%，门 H IoU 0.9948 / min-frame 0.985 / conf 最大 1.04% / MAE 6.31e-6。

**fp16 档（P1，真实 `mg_f16acc` kernel，当前 tuned16.json 的 tile，含 pkt=2 的配置；无预缩放，前缀行不特殊处理，另有"前缀行留 bf16"版）**：全部通过（见下表）。fp16 档的 logit MAE 与 P0 相同量级（ViT-B 1.4e-5 vs 1.45e-5；ViT-L 6.1–6.4e-6 vs 6.31e-6），说明误差预算基本没被动用。

ViT-B：
{{sweep_F:vitb16}}

ViT-L：
{{sweep_F:vitl16}}

**阶段 A（单个种类、全部层、RN，8 帧筛选，仅门 M 与 logit MAE）**：

ViT-B：
{{sweep_A:vitb16}}

ViT-L：
{{sweep_A:vitl16}}

（待做：阶段 G1–G6 的汇总、组合 finalist、决策。）
