# infer-opt 第二阶段（v2）：Blackwell 低精度 + 自研算子 + 大模型推理优化思路（round10–14）

> 位置：`/root/lightly-train/infer-opt/`（批准后本文件另存为服务器上的 `infer-opt/PLAN_v2.md`，旧 `PLAN.md` 不动）
> 起点：R9 推荐栈（`r5_v1` 融合 block + R7 融合前后处理/head + R9 整帧 CUDA Graph + R8 三流流水线）
> 日期：2026-10-04

## Context

**为什么做**：在 R0–R9 已有 6.3× / 13.0× 的基础上继续提速。重点是用 Blackwell sm_120 的新张量核路径（FP16 累加、MXFP8、NVFP4/MXFP4 block-scaled MMA）写自己的高性能 CUDA 算子，并把大模型推理里成熟的做法（W8A8/W4A4 PTQ、量化融合进生产者、FP8/INT8 attention、整图/持久化调度、合批）用到这个视频分割模型上。

**用户决定（2026-10-04）**
1. 精度分两档：
   - 默认 **safe** 档必须过严格门：8 帧正式门 M + 126 帧全片 H；
   - 可选 **fast** 档只要求门 M，H 只记录。
2. 多帧合批吞吐模式、单次宽图推理两项都做，都作为可选开关，**默认关**。
3. NVDEC 解码库是否 pip 安装，**到 R14 时再问用户**；在那之前只做 CPU 侧的解码改进。

**现状（R9，同轮配对，整段视频稳态）**

| 模型 | 原脚本 | R9 | 加速 | 单帧延迟 | 8 帧门（IoU / conf） | 126 帧 conf 最大 |
|---|---|---|---|---|---|---|
| ViT-L/16 | 15.1 FPS | 94.7 FPS | 6.3× | 10.24 ms | 0.9946 / 0.13% | 2.98% |
| ViT-B/16 | 16.5 FPS | 214.1 FPS | 13.0× | 4.00 ms | 0.9943 / 0.21% | 0.49% |

**时间在哪（每帧 GPU，2 个 crop；来自 R9 nsys 和 R5 调参表，R10 会对 r5_v1 重新 nsys）**

| 部分 | ViT-L ms | 占比 | ViT-B ms | 占比 | 现在的实现 |
|---|---:|---:|---:|---:|---|
| 骨干 GEMM（每层 qkv/proj/fc1/fc2） | ~6.6 | 68% | ~2.0 | 51% | R5 手写 mma.sync / TileLang，bf16 输入 + fp32 累加，152–207 TFLOPS |
| attention | 1.42 | 15% | 0.71 | 18% | torch flash，59 µs/层，约 147 TFLOPS |
| LayerNorm | 0.22 | 2% | 0.09 | 2% | R3 `add_ln` |
| head（ConvT GEMM + dw_ln + einsum） | ~0.90 | 9% | ~0.55 | 14% | cuBLAS GEMM + R7 `dw_ln` |
| patch-embed 卷积 + token 拼接/拷贝/尾 LN | ~0.18 | 2% | ~0.14 | 4% | cuDNN implicit conv，只有约 27 TFLOPS |
| prep + post | 0.14 | 1% | 0.13 | 3% | R7 |
| **GPU 合计** | **~9.5–9.7** | | **~3.6–3.9** | | 分项之和 / 按 R9 nsys 跨度换算 |
| 实际每帧（吞吐的倒数） | 10.56 | host 差约 0.9 | 4.67 | host 差约 0.8 | Python 每帧开销、解码线程 |

**R6 的低精度为什么没进主线**
1. 只接在 `lightly` 路径上：cuBLAS `_scaled_mm` 没有 RoPE/GELU/残差 epilogue，和 r5 融合 block 叠不上。
2. 朴素 PTQ：没有平滑、旋转、GPTQ。ViT-L 只敢量化 6/96 个 GEMM。
3. NVFP4 只做了假量化。

R6 同时证明：精度瓶颈在任务指标本身。head 只有约 2000 像素，IoU/conf 对扰动敏感；ViT-L 第 7 帧的 conf 是"混沌"的，所以必须看 126 帧。

**5090（sm_120）的关键事实**（来源：Spark `bench_peak`、gn-kernels、RTX Blackwell 白皮书、train-opt 实测；R10 在本机复测）

| 张量核路径 | 指令峰值 TFLOPS | 2048³ 量级实测 | 说明 |
|---|---:|---:|---|
| BF16/FP16 输入 + FP32 累加（现状） | 259 | 168–207 | GeForce 限半速 |
| **FP16 输入 + FP16 累加** | ~517（规格 419） | Triton 279，约 1.66× bf16 | 不限速；不需要量化 |
| INT8，s32 累加 | ~1000（规格 838） | 466–583 | 不限速 |
| **MXFP8 block-scaled，FP32 累加** | 1014 | 462–703 | 不限速；scale 全为 1 时就是满速 FP8 |
| **NVFP4 / MXFP4** | 2029 | 762–1369 | 不限速 |
| FP6（e2m3/e3m2） | 与 FP8 同速 | — | 没有额外收益，不做 |

其他 sm_120 特性：
- 没有 wgmma / tcgen05 / TMEM，张量核只能用 `mma.sync`。
- 有 TMA + mbarrier、cluster≤8 + DSMEM（无 multicast）、PDL（`griddepcontrol`）。
- `cvt` 可直接转 e2m1x2 / e4m3x2 / ue8m0x2。
- FP4 的两个操作数都必须 K-major。推理前向的 GEMM 天然都是 TN，所以没有 train-opt 那种转置副本问题。

**可复用资产（全部只读引用，不改原文件）**

| 来源 | 内容 |
|---|---|
| infer-opt | `common/model_factory.py`（8 帧门 `mf.gate`）、`round9_integration/bench9.py`（整视频配对测法）、`round5_gemm_sm120/impl/mg_core.cu` + `mgb.py`（手写 GEMM：cubin-per-config、可 capture 的 launcher、split-K、RoPE/GELU/残差 epilogue）、R6 `mx_ext.cu` / `sens.py` / `greedy.py` / `heldout_real.py`、R7 prep/post/head kernel |
| train-opt | r17 MXFP8 双输出量化器 + QuACK block-scaled GEMM 包装（`gemm_mx8.py`）；r20 `fp4_ref.py`（nv4/mx4 参考，含 RHT/SR）和 QuACK / cuBLASLt / Spark / QuTLASS 的 FP4 候选包装；r21/r24 融合 FP4 量化器；r12 手写 TMA attention（尾块精确） |
| third_party（`/root/autodl-tmp/train-opt-tmp/third_party`） | Spark（MIT：fp8/fp4 GEMM v2，TMA + warp specialization；fp8 attention）、SageAttention2/3（sm_120a）、QuTLASS（Hadamard + FP4 量化）、gn-kernels |
| 环境 | torch 2.12（`allow_fp16_accumulation`、NVFP4/MXFP8 `_scaled_mm`）、QuACK 0.6.5、TileLang 0.1.14、Triton 3.7.1、CUTLASS 4.2；不需要新装包（解码库除外，见 R14） |

**目标（估计，R10 用实测替换）**

| 模型 | R9 | safe 档 | fast 档 |
|---|---:|---:|---:|
| ViT-L | 94.7 FPS | 150–170 FPS | 190–230 FPS |
| ViT-B | 214 FPS | 330–400 FPS | 450–520 FPS |

ViT-B 超过约 285 FPS 的部分需要先解决解码瓶颈（R14）。

---

## 0. 规则（沿用 train-opt 第三阶段的教训）

- **只新增目录**：只新建 `round10_*` 到 `round14_*`；R0–R9 只读 import。
- **共享文件只追加**：README、`common/summarize.py` 的 PRIMARY、`common/` 新文件。改之前先备份到 `/root/autodl-tmp/infer-opt-tmp/*.bak_pN`。
- **先仿真、后写 kernel**：先用 fake-quant 在真实权重、真实帧上跑门，定下哪些 GEMM 能降精度，只给过门的写 kernel。go/no-go 规则事先写死（§3 R10）。
- **不设只写 kernel 的轮**：每轮交付一个可插拔的 `model_fn` 变体（R9 接口）。它要和当前最优栈在同一场次做 A-B-A-B 整视频配对；在它适用的模型上 FPS ≥ +2% 才发布。
- **逐层叠加**：R12、R13 都叠在 r11 第一部分（bf16 等价小项）之上，R14 不用重新集成这些小项。
- **门要跑全才算完成**：门 A/M/H/G 全部跑完（§2.3）。
- **GPU 纪律**：所有 GPU 进程（计时、nsys、精度）都持双锁，即 infer-opt 和 train-opt 的 `gpu.lock`，顺序与 train-opt `common/bench.py::gpu_lock` 相同，新写 `common/lock2.py`。
  - 单个任务 ≤10 分钟，nohup + 日志。
  - 计时只用 CUDA graph 回放，候选同进程交替，3 轮取中位数。
- **防中断**：NOTES.md 和 results.json 边做边写，被会话额度打断后可以续做。
- **环境**：不 pip install。Python 用 `/root/miniconda3/bin/python`；编译用 `nvcc -arch=sm_120a`；build 目录放在 `/root/autodl-tmp/infer-opt-tmp/`。

## 1. 大模型推理优化思路 → 本项目怎么用

| 大模型推理做法 | 在这里的对应 | 轮次 |
|---|---|---|
| FP16 累加 + 分块提升到 FP32（消费级卡的 fp16 accumulation；DeepSeek-V3 FP8 GEMM 的 promotion；SageAttention 的 fp16+fp32 累加） | GEMM 和 attention 改成 FP16 输入 + FP16 累加，每个 k-tile 把部分和加回 FP32 寄存器。约 2× 指令速率，不需要量化，误差应 ≈ bf16 | R10 / R12 / R13 |
| W8A8 FP8（TensorRT-LLM / vLLM 的 per-block MX 或 per-channel） | MXFP8 骨干 + head GEMM，权重离线量化，激活在线 1×32 block 量化 | R10 / R12 |
| W4A4 NVFP4 + 块 Hadamard + GPTQ（QuaRot / SpinQuant / MR-GPTQ，QuTLASS） | NVFP4/MXFP4 用在 MLP 等不敏感的 GEMM 上，作为 stretch | R10 / R12 |
| SmoothQuant 离群值迁移 | LN→qkv、LN→fc1 的通道平滑，折进 LN 的 γ/β 和 W | R10 / R12 |
| 少量校准数据的 GPTQ / AWQ | 用与评测帧不重叠的 16 帧校准权重舍入 | R10 / R12 |
| 敏感层保留高精度（首尾层等） | 按（层, GEMM 种类）的敏感度做混合精度策略；前缀 token（cls + 4 register，joint 层再加 100 query）单独处理 | R10 |
| 量化融合进生产者（fused RMSNorm+quant、SwiGLU+quant、attention 输出量化） | LN+量化、fc1 的 GELU+量化 epilogue、attention 输出量化；整条链上不再有独立的量化 kernel | R12 / R13 |
| FlashAttention-3 FP8 / SageAttention2/3 | INT8 QK + FP8 PV（f16 累加）或 FP4 attention，D=64，RoPE、量化都融合 | R11 / R13 |
| CUDA Graph 整步、PDL、stream-K、持久化 megakernel（FlashInfer / Mirage MPK） | 整帧 graph（已有）；更快的 kernel 下重新测 PDL；N=1024 的 GEMM 用 split-K/stream-K；megakernel 只作 stretch | R12 / R14 |
| 连续合批 / prefill 合批 | 多帧合批的吞吐模式，或双流并发（用户已同意；可选开关，默认关） | R14 |
| 去掉冗余计算 | 单次宽图推理：两个重叠 crop 约有 34% 的重复 token（会改变输出；用户已同意；可选开关，默认关，必须过门） | R14 |
| 硬件解码直通 | NVDEC 解码直接落到显存，NV12→RGB 并进 prep kernel（装不装包到 R14 时再问用户） | R14 |

不做（理由）：
- 2:4 稀疏：需要微调恢复精度。
- token 剪枝/合并：改变分割输出。
- 跨帧 token 复用（KV cache 类比）：全局 attention 下无法精确复用，近似复用是另一个大项目。
- W4A8：sm_120 上与 FP8 同速，没有计算收益。
- 投机解码：不适用。

## 2. 精度方案

### 2.1 档位（每个模型、每种 GEMM 可以不同）

| 档 | GEMM 输入 / 累加 | 预期速度（对 bf16，2048 量级） | 风险 |
|---|---|---|---|
| P0 | bf16 / fp32（现状） | 1× | — |
| **P1** | FP16 / FP16，每个 k-tile 提升到 fp32 寄存器 | 约 1.6× | 溢出：massive activation 在 cls/register 行，层 0 残差约 1.5e5。处理：每层 2 的幂预缩放；前缀行静态 2 的幂缩放；仍溢出的 GEMM 回到 P0 |
| **P2** | MXFP8 E4M3 + UE8M0 1×32（RCEIL），fp32 累加；对照 INT8 + SmoothQuant | 约 2.5×（含融合量化） | R6：ViT-L 覆盖小 |
| P3 | NVFP4（E2M1 + E4M3 1×16 + 每张量 fp32 scale，静态校准）/ MXFP4，配块 Hadamard + GPTQ | 约 4× | 大概率只有 ViT-B 的部分 MLP 能过门 |

attention 的档位：
- A0：flash bf16（现状）。
- A1：FP16 累加 FA2。
- A2：Sage 式 INT8 QK（K 减均值平滑）+ FP8 PV，f16 累加 + f32 缓冲。
- A3：Sage3 FP4。

head 的 ConvT GEMM 与骨干同档。

### 2.2 保持不变的数值约定（R0/R3 结论）
- 残差流 fp32。
- LN 统计量、RoPE、softmax、后处理都用 fp32。
- LayerScale γ 在 fp32 epilogue 里乘。

### 2.3 门
- **A（算子）**：
  - 量化器对 `round10_lowp_probe/impl/quant_ref.py` 逐位一致。
  - 低精度 GEMM 对"反量化操作数的 fp32 乘积"，相对 L2 ≤ 2e-3（bf16 输出舍入量级）。
  - P1 GEMM 对 fp32 的误差 ≤ bf16 GEMM 误差 × 1.5。
- **M（8 帧正式门，`mf.gate`）**：对 fp32 eager，pooled head IoU ≥ 0.95，每帧 conf 相对误差 ≤ 5%。
- **H（126 帧全片）**：pooled IoU ≥ 0.95，每帧 conf ≤ 5%。另外报告最差单帧 IoU、conf > 3% 的帧数、logit MAE。统计时去掉校准帧。
- **G**：graph 回放与 eager 逐位相同。
- 档位与门的对应（用户已定）：
  - safe 档（默认）要求 M + H；
  - fast 档（可选）只要求 M，H 只记录；
  - 两个模型各自定策略，例如 ViT-L 的 safe 档可能只有 P1。
- 可选开关也要过门：多帧合批过 M + H + G；单次宽图推理至少过 M，H 也要报告。
- 筛选用的平滑指标：logit MAE、mask-logit cos（R6 已证明 IoU/conf 是离散的、混沌的）。

## 3. 各轮

### R10 `round10_lowp_probe`（Agent A，与 R11 并行；约 3–4 h）：GEMM 侧的实测 + 仿真 + go/no-go

1. **峰值**：在 tmp 里编译 Spark `bench_peak.cu`，复测 bf16-f32acc、fp16-f16acc、fp16-f32acc、int8、fp8 plain、mxf8f6f4、nvf4/mxf4 的指令速率。另外零代码先测 cuBLAS fp16 + `allow_fp16_accumulation=True` 在我们形状上的速度。
2. **r5_v1 的 nsys 基线**：用 `common/profile_nsys.sh` + `nsys_analyze.py` 出每类 kernel 的每帧时间，替换上面 Context 里的估计表。
3. **GEMM 速度矩阵**：CUDA graph 回放、交替运行、3 轮中位数、持双锁。
   - 形状：
     - ViT-L/B × qkv/proj/fc1/fc2 × M ∈ {2058, 2258}，另加 {4116, 4516} 供合批模式；
     - head ConvT1/ConvT2 GEMM、mask einsum；
     - patch-embed GEMM [2048, 768] × [768, C]。
   - 候选：
     - r5_v1（bf16，带 epilogue，当前值）；
     - cuBLAS fp16-acc；
     - **`mg_f16acc` 原型**：把 R5 的 `mg_core.cu` 复制到本轮，MMA 换成 `mma.sync.m16n8k16.f16.f16.f16.f16`，每个 k-tile 提升到 fp32，epilogue 不变；
     - MXFP8：cuBLASLt `_scaled_mm`、QuACK mx8 + epilogue（只读 import train-opt r17 的 `gemm_mx8.py`）、Spark fp8gemm v2（MX 模式，M 补齐到 128）；
     - NVFP4/MXFP4：cuBLASLt、QuACK（r20 `fp4_cands.py` / `fp4_epi.py`）、Spark fp4gemm v2、QuTLASS；
     - INT8：`torch._int_mm`。
   - 量化器开销：
     - LN+量化：R6 `ln_quant`（MX），r17 `quant_dual` 行版，r24 `fp4q2` 行版；
     - GELU+量化；
     - 由此估计融合进生产者后的成本。
4. **精度仿真（核心）**：
   - 方法：在 r5_v1 的 `fused_block` 上把每个 GEMM 的 X/W 做"量化→反量化成 bf16"，再走 R5 原来的 epilogue kernel。MX8/NV4 的值在 bf16 里能精确表示，所以只差累加顺序。P1 用 fp16 舍入的输入，加上原型 kernel 实跑。
   - 溢出扫描：每层每 64-k 块部分和的最大值，以及每层需要的 2 的幂预缩放。
   - 扫描空间：
     - 格式：fp16 / mx8 / int8+SQ / nv4 / mx4；
     - GEMM 种类：qkv / proj / fc1 / fc2 / head；
     - 层集合：all / nofirst / nojoint / 逐层贪心；
     - 技术：无 / SmoothQuant α / 16-32 点块 Hadamard / GPTQ / 前缀行留 bf16。
   - 流程：8 帧 logit MAE 粗筛 → 8 帧门 → 126 帧决赛。结构参考 R6 的 `sens.py` / `greedy.py` / `heldout_real.py`，只读。
   - 新写 `impl/quant_ref.py`，从 train-opt r20 `fp4_ref.py` 改编，作为 R12 门 A 的唯一逐位参照。
   - 校准帧：16 帧，与 8 个评测帧不重叠。
5. **决策表 → `profile/decision.json`**：
   - 内容：每个模型、每种 GEMM 的档位；实测 GEMM Δ − 估计量化开销 = 每帧节省；精度证据。
   - go 规则（事先写死）：
     - 档位项：每帧节省 ≥ 0.15 ms（L 约 1.5%，B 约 4%），且过对应档的门（safe：M+H；fast：M）。
     - P1 额外要求：本机测得的指令速率 ≥ 1.7× bf16；原型在我们形状上（含 epilogue）≥ 1.3× r5_v1；溢出扫描能给 ≥ 90% 的 GEMM 找到安全的预缩放。

### R11 `round11_attn_tier0`（Agent B，与 R10 并行；约 3–4 h）：不改精度的小项 + attention 候选矩阵

**第一部分：bf16 等价的小项**，做成 `r11` model_fn，叠在 r5_v1 上，直接过门、配对、发布。

| 项 | 做法 | 预计省（L / B，ms/帧） |
|---|---|---|
| patch-embed | 改成 GEMM，去掉 cuDNN implicit conv 和 cat/copy。A 操作数有两种取法，按实测二选一：(a) 直接从 [B,3,512,512] 图上按 16×16 patch 取，即隐式 im2col，每个 patch 的一行是 16 个连续像素，可以 16 B 向量加载；(b) 让新 prep kernel 直接写 patch-major 的 [B·1024, 768]。epilogue 直接写 fp32 残差流 x0，cls/register 行也在这里拼进去 | 0.13 / 0.10 |
| head `dw_ln` | 2D tile + smem halo 复用、2 blocks/SM、bf16x2 向量化，目标 2×（R7 记录了 2.5× 的 halo 冗余） | 0.13 / 0.09 |
| `post_a` | 用 ex2.approx/rcp.approx 算 sigmoid，fp16x2 做点积。可选开关：跳过 class 概率 < ε 的 query（误差有界，单独过门） | 0.05 / 0.05 |
| 尾部 | `bb.norm` 和小拷贝并进一个 kernel | 0.02 / 0.02 |

**第二部分：attention 候选矩阵**（速度 + 精度，每个候选单独替换 r5_v1 的 attention 后过门 M/H）

- 候选：
  - flash（现状）；
  - R4 TileLang；
  - SageAttention2 sm_120a：INT8 QK（per-warp / per-thread）+ FP8 PV，"fp32+fp16" 累加；
  - SageAttention2 的 INT8 QK + FP16 PV f16 累加；
  - SageAttention3 FP4；
  - Spark fp8 attention（先查是否支持 D=64）；
  - **FP16 累加 FA2 原型**：基于 train-opt r12 的 `attn_fwd.cu`（手写 TMA + mbarrier，尾块精确）或 R4 TileLang 改。
- 形状：B=2（另加 4）× H=16/12 × N=1029/1129 × D=64。
- 编译：Sage 复制到 tmp 里 `build_ext --inplace`，或用 `cpp_extension.load`，不 pip install。
- 交付：`r11` model_fn、`profile/attn_decision.json`。

### R12 `round12_gemm_lowp`（Agent A；约 4–6 h）：按 R10 决策做生产版低精度 GEMM

1. **量化激活格式 + 设备端 helper**（`impl/qfmt.cuh`）：MX8 / NV4 / FP16（+ 2 的幂预缩放）。scale 布局由本轮 GEMM 的加载方式决定（若用 QuACK/cuBLAS 则用它们的 128×4 blocked 布局）。R13 的 attention 输出也 include 这个 helper，按同一格式写。
2. **生产者融合**：
   - LN+量化：含 SmoothQuant 折叠后的 γ'/β'，可选块 Hadamard；
   - fc1 epilogue 做 GELU+量化，直接产出 fc2 的输入；
   - attention 输出量化：R13 提供，之前用一个独立的量化 kernel 过渡；
   - patch-embed 输出。
3. **GEMM kernel**：
   - 把 R5 的 `mg_core.cu` 复制到本轮（不改 R5），增加三种 MMA：fp16-acc（带提升）、`kind::mxf8f6f4.block_scale`、`kind::mxf4nvf4.block_scale`。
   - k-loop 的 fragment/scale 写法参考 Spark 的 `fp8gemm_tile.cuh` / `fp4gemm_tile.cuh`（MIT，注明出处）。
   - epilogue 复用 R5：RoPE + [3,B,N,H,64] 布局、GELU、γ + fp32 残差、split-K。
   - 如果 R10 显示 TMA + warp specialization（Spark v2 / QuACK）在我们形状上明显更快，主循环改用这种结构。
   - 每个形状调 tile / split-K / stream-K；在新 kernel 上重新测 PDL；可选 L2 持久化窗口（残差和激活常驻 L2）。
4. **离线权重准备**：`prepare_block_weights_lowp()` 一次完成 MX8/NV4 量化、GPTQ / SmoothQuant / Hadamard 折叠（按 R10 决策），并缓存。
5. **head**：ConvT1 / ConvT2 GEMM 用对应档位。
6. **集成**：
   - 交付 `r12_<policy>` model_fn，两个模型各自一套策略；
   - 门 A/M/H/G 全跑；
   - 与 r5_v1、r11 同场 A-B-A-B 整视频配对。

### R13 `round13_attn_lowp`（Agent B；约 3–4 h）：按 R11 决策做生产版 attention

- D=64，N=1029/1129，尾块精确。
- 直接读 R12 qkv epilogue 输出的 [3,B,N,H,64]（fp16/bf16，RoPE 已做）。
- 需要的 Q/K/V 量化放在 attention prologue 里做，或者按 §4 约定由 qkv epilogue 直接写 8 位 + scale。
- K 平滑所需的均值：在 qkv epilogue 里原子累加，或用一个小 kernel。
- 输出直接写成 proj 的输入格式（fp16，或 MX8 + scale，用 R12 的 `qfmt.cuh`）。
- 门全跑；叠在 r12 上做配对 bench。

### R14 `round14_final`（Agent，Sonnet 5.5 xhigh；约 3 h；我做最终审核和汇报）：集成 + 系统层 + 发布

1. **集成**：r11 + r12 + r13 叠成 `r14_safe` / `r14_fast`。
2. **系统层**：
   - 解码：先做 CPU 侧改进并实测吞吐（线程数、YUV 直出 + GPU 转色、多实例解码）。
     - 如果解码仍是瓶颈，**这时再问用户**是否 pip 安装 NVDEC 解码库（PyNvVideoCodec 或 torchcodec），装之前先 dry-run。
     - 接上 NVDEC 后，NV12→RGB 并进 prep kernel，同时省掉每帧 8 MB 的 H2D。
   - host 每帧开销：减少 Python 调用、合并 D2H、用事件查询代替同步；必要时写一个 C++ 小扩展。
   - 吞吐模式（可选，默认关）：`--frame-batch 2`（每张 graph 处理 4 个 crop，tile 用 R12 在 M=4116 调好的配置）和双流并发两种实现，选优；报告 FPS 与单帧延迟的取舍。
   - 单次宽图推理（可选，默认关）：
     - 512×675 的整图补到 688，一次前向；mask/post 改成单块，不再 untile；
     - 必须过门 M，H 也要报告；
     - NOTES 里写明：这一项改变的是计算方式，不只是数值。
3. **发布**：
   - 同场 A-B-A-B 对比：原脚本 / R9 r5_v1 / r14 各档，两个模型，126 帧 × 3 遍；
   - 门 M/H/G；nsys；
   - 新的 `round14_final/video_infer.py`：
     - `--precision bf16|safe|fast`，默认 safe；
     - `--frame-batch 1|2`，默认 1；
     - `--single-pass`，默认关；
     - 不支持的形状（竖图、其他 crop 数）照 R9 的做法自动回退，并打印实际用的实现；
   - README 追加一节；`summarize.py` 的 PRIMARY 追加（先备份）。

## 4. 接口约定（写进 PLAN_v2.md，R12 和 R13 都按它做）

- **model_fn**：沿用 R9：`model_fn(static bf16 [B,3,512,512]) -> (mask_logits_lr, class_logits)`，可 capture，无 host 同步。
  - 优先保持这个接口（patch-embed 用隐式 im2col）。只有 R11 实测 patch-major prep 明显更快时，才加新接口 `model_fn_patches(patches [B·1024,768])`，并在新 round 里写一个 R9 pipeline 的子类。
- **qkv → attention**：物理布局 [3,B,N,H,64]，RoPE 已应用，dtype ∈ {bf16, fp16}。8 位格式由 R13 定义，R12 的 epilogue 留开关。
- **attention → proj**：[B·N, H·64]，dtype / scale 用 R12 `qfmt.cuh` 定义的格式。
- **前缀 token**：普通层前 5 行（cls + 4 register），joint 层前 105 行（100 query + cls + 4 register）。P1/P2 的缩放和排除都按这个行号。
- **量化参照**：只有 `round10_lowp_probe/impl/quant_ref.py` 一份；门 A 逐位比较。
- **锁**：`common/lock2.py::gpu_lock2()`。先拿 infer 锁，再拿 train 锁。
  - 持锁期间同时设 `INFER_OPT_GPU_LOCK_HELD=1` 和 `TRAIN_OPT_GPU_LOCK_HELD=1`。
  - 原因：旧的 `bench.time_fn` 等函数内部会再调 `gpu_lock()`；同一进程用新 fd 再 flock 会自锁。

## 5. 执行安排

| 阶段 | Agent A（infer-opt-worker，Sonnet 5.5 xhigh） | Agent B（同上） | 我（Opus 5.5） |
|---|---|---|---|
| 0 | — | — | 写 `PLAN_v2.md`、`common/lock2.py`；在 README 追加"第二阶段进行中" |
| 1 | R10 GEMM 探针 + 仿真 + 决策 | R11 小项 + attention 矩阵 | 审核两份决策表，按 §3 的规则 go/no-go，汇报 |
| 2 | R12 生产版 GEMM | R13 生产版 attention | 盯接口约定、仲裁冲突 |
| 3 | R14 集成 + 系统 + 发布 | — | 解码若仍是瓶颈，先问用户是否装 NVDEC 库；审核 R14，写最终汇报 |

- 用户要求（2026-10-04）：任务由 Sonnet 5.5 xhigh 子代理完成，R14 也交给子代理；我只做协调、审核和汇报。
- 最多 2 个 agent 并行。每个 agent 用新上下文、紧凑 prompt，在自己的 round 目录里工作；prompt 里写明 §0 规则。
- 被会话额度打断后，先在服务器上 `ps` + `ls -lt` 查进度，再用 SendMessage 让 agent "继续，不要重做"。

## 6. 风险与退路

| 风险 | 退路 |
|---|---|
| ViT-L 的 P2/P3 过不了门（R6 的经验） | P1 不量化，误差 ≈ bf16，仍有约 1.6× 的 GEMM 收益；P2 只用在 ViT-B 或少数 GEMM 上 |
| P1 在个别层溢出 | 每层 / 前缀行做 2 的幂预缩放；仍不行的 GEMM 留 P0 |
| 小 M（2058）下 FP8/FP4 GEMM 效率低 | tile / split-K / stream-K 调参；整体收益 < 2% 就不发布 |
| Sage 在 CUDA 13 / torch 2.12 下编译失败 | 用 Spark fp8 attention（MIT，sm_120 已验证）或 train-opt r12 手写 attention 改 |
| GPU 与 train-opt 共用 | 双锁；单任务 ≤10 分钟 |
| 只有一段测试视频，有过拟合风险 | 校准帧与评测帧分开；126 帧统计去掉校准帧；在 NOTES 里写明"换视频前需要重新验证" |

## 7. 验证（每轮都做，R14 汇总）

```bash
cd /root/lightly-train/infer-opt
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
# 门 A（各轮 tests/），例：
python round12_gemm_lowp/tests/test_gemm.py && python round12_gemm_lowp/tests/test_quant.py
# 门 M + H + G + 整视频配对（各轮 bench 脚本按 bench9.py 改编，持双锁）：
python round14_final/bench14.py --model vitl16 --variants r5_v1,r14_safe,r14_fast --loops 3
python round14_final/bench14.py --model vitb16 --variants r5_v1,r14_safe,r14_fast --loops 3
# 可选开关（默认关），单独对照：
python round14_final/bench14.py --model vitl16 --variants r14_safe,r14_safe+fb2,r14_safe+single --loops 3
# nsys：
common/profile_nsys.sh round14_final/profile/vitl16_safe round14_final/bench14.py --model vitl16 --profile r14_safe
# 汇总：
python common/summarize.py
```

最终汇报给出同场的表：原脚本、R9、R14 各档 × 两个模型。列包括 FPS（inflight 2）、单帧延迟 p50/p90、model-only graph p50、门 M/H 的数值、nsys 的 kernel 数和 GPU 忙碌率。

## 8. 预计每帧 GPU 时间（ms，估计；R10/R11 后用实测更新）

"现在"一列是分项估计之和，R10 的 nsys 会给实测值。

| 部分 | ViT-L 现在 | ViT-L safe | ViT-L fast | ViT-B 现在 | ViT-B safe | ViT-B fast |
|---|---:|---:|---:|---:|---:|---:|
| 骨干 GEMM | 6.6 | 4.1（P1） | 3.0（P2 为主） | 2.0 | 1.25 | 0.85 |
| attention | 1.42 | 0.85（A1） | 0.5（A2） | 0.71 | 0.43 | 0.27 |
| LN（+量化） | 0.22 | 0.22 | 0.25 | 0.09 | 0.09 | 0.10 |
| head | 0.90 | 0.65 | 0.50 | 0.55 | 0.40 | 0.33 |
| patch-embed + 尾部 | 0.18 | 0.05 | 0.05 | 0.14 | 0.04 | 0.04 |
| prep + post | 0.14 | 0.10 | 0.10 | 0.13 | 0.09 | 0.09 |
| **合计** | **9.5** | **~6.0** | **~4.4** | **3.6** | **~2.3** | **~1.7** |

若 P3（NVFP4）在 MLP 上过门，ViT-L fast 还能再省约 0.8 ms。
