# train-opt 第三阶段（v3）计划：MXFP4 / NVFP4 训练加速（round20–23，在 FINAL 之上迭代，旧 round 不覆盖）

## Context

**现状**

FINAL = `common.config.BEST_STACK_V2`（P2R14R15），在 round16 的 `StepGraphRunner` 下运行（同一台机器，三相加权）：

| 模型 | img/s | mid 步时 | 对 R0 |
|---|---:|---:|---:|
| ViT-L B=7 | 68.87 | 101.8 ms | 4.22× |
| ViT-B B=14 | 161.94 | 86.6 ms | 4.29× |

- GPU 忙碌率 0.98，每步 0 次同步，**剩下的只有 kernel 时间**。
- r19 的 roofline 结论：
  - 不改精度，最多还能再省约 11 ms；
  - 大头只能靠降精度：head 的 ConvT GEMM（bf16，已贴着约 230 TFLOPS 的限速上限）和骨干的 MXFP8 GEMM。

**用户决定（2026-10-03）**

1. 在 FINAL 基础上用 MXFP4 / NVFP4 继续提速。
2. 吸取上一阶段的经验，**少一个没用的 round**。
3. **不覆盖之前的 round**，在它们之上继续迭代。
4. 验证沿用 few-batch（1–2 个真实 batch，跑几次），不做长训练。
5. **FP4 用放宽后的精度门**：全局梯度余弦中位数 ≥0.98、最小 ≥0.97，再加 S 门。
6. 子代理一次只跑一个：R20 用 Sonnet，R21/R22 用 Opus。
7. 不加附加项：权重量化并进 AdamW、round18 val/ckpt、FP8 attention 都不做。

**目标**（预估，以 R20 实测为准）

| 模型 | 保守（head + 骨干反向 FP4） | 乐观（再加上放宽门后的前向项） |
|---|---|---|
| ViT-L | 78–81 img/s（+13–18%，4.8–5.0× R0） | 83–88 img/s（+21–28%，5.1–5.4× R0） |
| ViT-B | 185–190 img/s（+14–17%，4.9–5.0× R0） | 198–207 img/s（+22–28%，5.2–5.5× R0） |

## 0. 第二阶段的教训 → 本阶段规则（"少一个没用的 round"）

| 第二阶段发生的事 | 本阶段规则 |
|---|---|
| r12 的融合 attention：kernel 写完了，但没集成、没有整步数据，整轮白做 | **不设只写 kernel 的轮次**。每一轮都要交付能叠加到栈上的 patch，并在 StepGraphRunner 下与 FINAL 同场次对比；整步提升不到 2% 的不发布 |
| r9 可行性轮：只证明了 "nvfp4 supported/finite"，没有进主线 | 可行性压缩成 R20 的一次决策，go/no-go **事先写死**（§5）。R20 写的仿真代码同时是 R21/R22 gate A 的逐位参照，也是 R22 的块骨架，不是一次性代码 |
| r14 `mx8_bwd`、r17 的部分策略：实现了，但门没跑全，最后不能发布 | 门算在同一个任务的预算里：B（两个模型各 ≥8 批）、S（4 个场景）、graph_equiv、导出/resume 全部跑完才算完成 |
| r17 在 eager 下被 host 卡住，GPU 侧的收益看不出来 | 整步速度只在 graph 下测；算子级只用 CUDA graph 回放测 |
| MXFP8 已经在门的边缘（ViT-B 中位数 0.9945），FP4 单个 GEMM 的误差又是 MXFP8 的 2–4 倍 | **先仿真、后写 kernel**：先用 fake-quant 跑真实的 B/S 门，定下哪些 GEMM 能上 FP4，只给过门的 GEMM 写 kernel；两个模型可以用不同策略 |
| S 门用了跨进程缓存的曲线 | R0、P1、FINAL 和候选在同一进程、同一批数据上跑 |
| 3 个并行代理连续撞会话额度；`assert_gpu_exclusive` 误报 | 一次只跑一个子代理，新上下文、prompt 紧凑，每个任务 ≤1.5–2 h；NOTES 和 results 边做边写；GPU 任务排队、持双锁 |
| r15 的 checkpoint bug 到 r19 才被发现 | 凡是动到参数或状态的轮，都要真跑一次 `torch.save`、export 和 resume |
| 不同时段测量有 ±2–3% 的漂移 | 倍数只在同一场次的测量之间比较 |

## 1. 事实依据（已核对本地源码和 third_party 文档）

### 硬件

数据来自 Spark `bench_peak` 在 5090 上的实测、gn-kernels，以及 r17/r19。

| 指令 / GEMM | 指令峰值（TFLOPS） | 持续 / 实际 GEMM |
|---|---:|---|
| BF16 HMMA，fp32 累加 | 259 | 约 230–240 |
| MXFP8 `m16n8k32 kind::mxf8f6f4.block_scale` | 1,014 | 约 745（600 W 下）；我们的形状上实测 314–638 |
| **FP4 `m16n8k64 kind::mxf4nvf4.block_scale`**（NVFP4 和 MXFP4 速度相同） | **2,029** | 约 1,530（600 W 下）；大形状上 Spark v2 / cuBLASLt 达到 1,220–1,430，gn-kernels NVFP4 1,306（同机 MXFP8 为 660） |

- 本机功耗上限 575 W，我们的形状属于中等（M 约 7k–57k，N 和 K 在 768–4096），FP4 的实际速度要在 R20 实测。
- fc1 前向、fc2 dgrad 的 epilogue 要读写 118 MB，会受显存带宽限制。

### 精度

Spark `fp4_accuracy`，4096³，A、B 都量化后 GEMM 的相对 Frobenius 误差：

| 输入 | NVFP4 | MXFP4 | MXFP8 |
|---|---:|---:|---:|
| 高斯 | 13.4% | 16.2% | 4.2% |
| 8 个异常通道（×50 到 ×2000） | 10–12% | 17.5–18.6% | 4.9–5.1% |

- NVFP4 约为 MXFP8 的 2–3 倍，MXFP4 约为 4 倍；NVFP4 比 MXFP4 低 1.2–1.8 倍。
- 已有观察：
  - r17：前向量化（qkv 前向）会让全局梯度余弦下降 0.002–0.007；反向量化几乎不影响。
  - r14：head 前向用 MXFP8 时，mask logits 的相对误差为 5–6%。

### 软件（都已在本地，不需要装新包）

- **QuACK 0.6.5**（Apache-2.0）
  - SM120 FP4 GEMM：`MmaMXF4Op` / `MmaMXF4NVF4Op`。
  - 格式描述：`NVFP4` 是 e2m1 + e4m3 vec16 + 每张量 scale；`MXFP4` 是 ue8m0 vec32。
  - 每张量 scale 以设备端 alpha 的形式折进 GEMM，不会触发 host 同步。
  - 另外还有：fp4 的 quantize-out epilogue（SFD，行/列方向）、Hadamard kernel、e2m1 随机舍入的软件实现（sm_120 没有硬件 `cvt.rs`）、Philox。
  - **限制：sm_120 上 4 bit 操作数只能是 K-major。** 所以 dgrad 需要物理转置的 Wᵀ，wgrad 需要沿 token 打包的 dYᵀ 和 Xᵀ。r17 用 `ldmatrix.trans.b8` 读 MN-major 操作数的做法，FP4 用不了。
  - QuACK 自带的 FP4 量化器是纯 torch、只支持 FLOOR，只能作参照。
- **torch 2.12**：`float4_e2m1fn_x2`、`F.scaled_mm`（cuBLASLt NVFP4 在 sm_120 可用，只支持 TN）。
- **CUTLASS 4.2**：`examples/79_blackwell_geforce_gemm/79a,79b`、`sm120_blockscaled_mma_tma.hpp`。
- **third_party**：
  - Spark（MIT）：`fp4gemm.cu`、`fp4gemm_tma.cu`（v2 达到 cuBLASLt 的 96%）、`fp4quant.cu`（只有行方向）；
  - gn-kernels：`sm120_mm_nvfp4.py`。

### 文献配方（作设计依据）

| 来源 | 要点 |
|---|---|
| NVIDIA NVFP4 预训练 | 全部 GEMM 用 NVFP4；wgrad 输入做 16 点 RHT；权重用 2D 16×16 scale，保证前向和反向一致；梯度随机舍入；少数层保留高精度 |
| Tseng 等，MXFP4 | 只有反向用 MXFP4，配 RHT + SR，基本无损 |
| Quartet / QuTLASS | 在 RTX 5090 上用 MXFP4 + Hadamard 做全量训练 |
| FP4 All the Way | NVFP4 优于 MXFP4 |
| TetraJet | ViT 用 MXFP4 训练时，主要问题是权重振荡 |

## 2. 时间在哪，能省多少（预估，R20 会用实测替换）

FINAL graph，mid 相 kernel 时间（r19 §3.3，ms/步，ViT-L / ViT-B）：

| 部分 | ViT-L | ViT-B | 本阶段候选 | 预估节省（L / B） |
|---|---:|---:|---|---|
| head GEMM（bf16，在限速上限；ConvT 每步 3.6 / 4.1 TFLOP） | 19.0 | 21.6 | ① **ConvT dgrad+wgrad → NVFP4**（退路 MXFP8）；② 放宽门之后：**ConvT 前向 → MXFP8**（r14 已有 `mx8_fwd`），前向 NVFP4 作 stretch | ① −7…−8 / −8…−9；② −2.5…−3 / −3…−3.5 |
| 骨干 MXFP8 GEMM | 24.4 | 14.5 | ③ **dgrad+wgrad → NVFP4**（RHT+SR）；④ 放宽门之后：**fc1/fc2 前向 → NVFP4** | ③ −5…−7 / −3…−4；④ −1…−1.5 / −0.5…−1 |
| 骨干 bf16 qkv 前向 | 4.8 | 2.7 | ⑤ 放宽门之后：**重新试 qkv 前向 → MXFP8**（r17 的 `all` 档，严格门下最小余弦曾到 0.96–0.985） | −2.5…−3 / −1.5 |
| MXFP8 量化 | 5.1 | 4.1 | 改成 FP4 量化：一次读入、产出全部副本；NVFP4 的 amax 尽量并进生产者 | 已计入上面各行 |
| attention、r14 融合、LN、逐元素、loss、conv、优化器 | 45.4 | 42.3 | 不在本阶段范围内；FP4 attention 没有成熟的训练配方 | 0 |
| **合计** | 98.6 | 85.1 | 保守 = ①+③，乐观 = 全部 | 保守 −12…−15 / −11…−13；乐观 −18…−22.5 / −16…−19 |

步时换算：

| 模型 | FINAL | 保守 | 乐观 |
|---|---|---|---|
| ViT-L | 101.8 ms | 87–90 ms（78–81 img/s） | 79–84 ms（83–88 img/s） |
| ViT-B | 86.6 ms | 74–76 ms（185–190 img/s） | 68–71 ms（198–207 img/s） |

## 3. 精度方案与策略空间

**格式**
- 默认 **NVFP4**：
  - e2m1 两个一字节，K-major；
  - 1×16 的 e4m3 block scale，存成 128×4 blocked `(rm, rk, 32, 4, 4)`，与 QuACK、cuBLASLt 一致；
  - fp32 每张量 scale = amax/(6·448)，放在设备上，作为 GEMM 的 alpha。
- 对照 **MXFP4**：ue8m0 vec32，scale 取 RCEIL 2^ceil(log2(amax/6))，不截断；不需要扫 amax，量化更便宜。
- R20 实测后定格式：只有 MXFP4 也过门、而且整步更快，才选 MXFP4。

**三个 GEMM 的量化方式**（Y = X Wᵀ，全部 TN）

| GEMM | A 操作数 | B 操作数 | 舍入 | RHT |
|---|---|---|---|---|
| fwd | X 行副本（沿 K） | W（沿 K） | RN | 默认关；R20 试开（沿 K，两个操作数用同一个符号向量） |
| dgrad | dY 行副本（沿 N） | **Wᵀ 物理副本**（沿 N） | dY 用 SR，W 用 RN | 默认关；R20 试开 |
| wgrad | **dYᵀ**（沿 token，物理转置） | **Xᵀ**（沿 token） | dY 用 SR，X 用 RN | **开**：沿 token 做 16 点 FWHT，乘固定的随机符号，两个操作数用同一个符号向量 |

- token 维填充到 128 的倍数，填充行为 0（沿用 r17 的做法）。
- 权重每步在 block 0 从 fp32 master 量化一次，同时出 W 和 Wᵀ 两份。默认用 2D 16×16 scale，这样两份的数值完全一致。

**随机数契约**
- SR 和 RHT 的符号**不得使用 torch 默认生成器**。RngTape 在 replay 时会调用 `draw()` 来对齐默认流，如果别处也用默认生成器，B 门的逐元素比较就会错位。
- 仿真中用专门的 `torch.Generator`。
- kernel 中用 Philox：种子是设备上的 int64 计数器，在 graph 内每步加 1；每个调用点有固定的 offset。

**策略串**（R20 定义，R21/R22 沿用）
- 形式：`<线性层>.<种类>=<格式>[+选项]`，没写到的保持 FINAL 的设置。
  - 线性层：qkv / proj / fc1 / fc2 / fc / all / head。
  - 种类：f / d / w。
  - 格式：bf16 / mx8 / nv4 / mx4。
  - 选项：rht、sr、w2d、nofirst、nolast、nojoint。

**候选**（全部在 R20b 用放宽门筛选）

| 部分 | 候选 |
|---|---|
| head | `head.dw=nv4`（主线）、`head.dw=mx8`（退路，r14 真 kernel）、`head.f=mx8`（r14 真 kernel）、`head.f=nv4`（stretch） |
| 骨干 | `all.w=nv4` → 加 `all.d=nv4` → 加 `fc.f=nv4`（±rht）→ 加 `qkv.f=mx8`（r17 真 kernel）；`qkv.f=nv4` 作 stretch。proj 前向是带宽受限，不试 |
| 对照 | 最优档上 mx4 对 nv4、SR 对 RN、RHT 开对关、nojoint / nolast 排除 |

## 4. 轮次与任务（一次一个子代理，顺序执行）

### Step 0（我，约 30 分钟，不用 GPU）

1. 新建 `train-opt/PLAN_v3.md`，作为本计划在服务器上的版本。**PLAN_v2.md 不动。**
2. 在 `CONTRACTS.md` 末尾**追加** Amendment 6，内容包括：
   - `FP4Tensor` 格式：`q`、`sf`、`ts`（每张量 scale）、`qt`/`sft`（转置副本）、`rows`、`rpad`；
   - §3 的 RHT 约定和随机数契约；
   - §5 的放宽门；
   - §6 的不覆盖规则。
3. 更新本地 `.claude/agents/train-opt-worker.md`：
   - 改成第三阶段：round 20–23，计划文件 PLAN_v3；
   - 写入 FP4 事实：2,029 TFLOPS；sm_120 上 4 bit 只能 K-major；误差约为 MXFP8 的 2–4 倍；
   - 写入规则：旧 round 只读。
   - 默认模型仍是 sonnet；R21/R22 在调用时传 `model: opus`。
4. （可选，只 clone 不安装）把 QuTLASS（Apache-2.0）clone 到 `third_party/`，作为"融合 Hadamard + 量化"的参考设计，记入 `SOURCES.txt`。
5. 写入 memory：
   - "不覆盖旧 round"；
   - FP4 放宽门；
   - 子代理的模型分配；
   - FP4 硬件事实。

### R20 `round20_fp4_probe`：先决定，再动手（Sonnet，2 个任务）

**R20a 速度矩阵 + 误差表（≤1.5 h）**

- **GEMM 候选矩阵**
  - 测法：CUDA graph 回放，同一进程内候选交替运行，3 轮取中位数。
  - 形状：
    - 两个模型骨干的 12 个 GEMM，plain 和 joint 的 M 各一套；wgrad 的 K=Mpad；
    - head ConvT1/2 的 fwd/dgrad/wgrad：ViT-L 为 [7168,1024]×[1024,4096] 和 [28672,…]，ViT-B 为 [14336,768]×[768,3072] 和 [57344,…]。
  - 候选：
    - QuACK NVFP4/MXFP4：普通版本，以及带 r13 epilogue 的版本（bias、γ+残差、GELU、GELU'），wgrad 输出 fp32；
    - `F.scaled_mm`（cuBLASLt）；
    - Spark `fp4gemm` v2（编成扩展）；
    - 对照组：r17 的 MXFP8 和 bf16 最优实现。
  - CUTLASS ex79：只有当 QuACK 和 cuBLASLt 都不到 Spark 的 80% 时才去改。
  - 输出：`profile/fp4_gemm_matrix.json`，以及按 §3 候选算出的每步节省。
- **真实张量误差表**
  - 数据来源：在 FINAL eager 步上挂 hook，抓一个真实 batch 的 X、W、dY。
  - 范围：ViT-L 的块 0、11、22（joint），ViT-B 的块 0、6、10，以及 head ConvT。
  - 逐个 GEMM 算对 fp32 的相对 L2 误差，组合包括：mx8、nv4 RN、nv4+RHT、mx4、mx4+RHT、梯度 SR 对 RN、权重 1D 对 2D。
  - 输出：`profile/fp4_error_table.json`。
- **仿真参考** `impl/fp4_ref.py`：纯 torch 实现 nv4/mx4 的量化和反量化，包括 RHT、SR、w2d、填充，scale 用 blocked 布局。它是 R21/R22 gate A 的逐位参照。

**R20b 仿真门扫描（≤1.5 h）**

- **patch `round20_fp4_probe`**（变体就是 §3 的策略串）
  - 把 r17 `fused_block_mx8` 的结构**复制**到 `round20/impl/fp4_block.py`，改成按 GEMM 种类分派：
    - mx8 和 bf16 走 r17/r13 的真 kernel；
    - nv4/mx4 走"fake-quant 后的 bf16 GEMM，fp32 累加"，epilogue 与 r13 相同。
  - e2m1×e4m3 在 bf16 中能精确表示，每张量 scale 在 fp32 epilogue 里乘，所以仿真结果等于真实 FP4 GEMM，只差累加顺序。
  - head：在 round20 里**新写** `_ConvT` 子类覆盖 `ops.predict`，r14 的文件不动；按 f/d/w 分派到 bf16、mx8（r17 的量化器和 GEMM）或 nv4（仿真）。
  - 策略可以在运行时切换：每个进程只建一次模型，避免重复构建和 JIT。
- **扫描**：每个模型一个进程，R0、P1、FINAL 和所有候选在同一进程、同一批数据上跑。
  1. 筛选：3 批，跑全部候选。
  2. 决赛：每个模型取前 2–3 档，跑 ≥8 批 B 门，再跑 S 门的 4 个场景（schedule、mid、late、compressed），对 R0 和对 FINAL 各算一次。
- **输出**
  - 每个模型的 `POLICY`、预计节省，以及 R21/R22 是否执行，写进 NOTES §7 和 `results.json["decision"]`。
  - 同时记录每档是否满足严格的 Amendment 5 门，方便看清风险档位。
  - 之后我审核，并在 CONTRACTS 追加 6b 定稿。

### R21 `round21_fp4_quant_head`：FP4 量化器、GEMM 封装、head（Opus；R20 通过后做；≤2 h，可拆成 2 个任务）

- **量化器** `impl/fp4_kernels.cu`（sm_120a）
  - 输入：bf16 或 fp32 的 `[R,C]`。一次读入，产出任意组合：
    - 行副本：沿 C 打包，K-major；
    - 转置副本 `[C, Rpad/2]`：沿 token 打包；
    - scale 直接写成 blocked 布局；
    - 可选：沿量化轴做 FWHT-16 + 符号，Philox SR；
    - 填充行写 0；
    - 需要时顺带输出 mx8 行副本（head 前向是 mx8、反向是 nv4 时用）。
  - NVFP4 的 amax 实测三种做法：
    1. 单独做一遍预扫描；
    2. 持久化 kernel 加 grid 同步，第二遍从 L2 读；
    3. 生产者用 atomicMax 写出（例如 r14 的 `bwd2`）。
  - 权重量化器：2D 16×16，同时出 W 和 Wᵀ。
- **GEMM 封装** `impl/gemm_fp4.py`：后端用 R20a 选出的；TN 的 fwd、dgrad、wgrad；wgrad 输出 fp32；每张量 scale 作设备端 alpha。
- **head 集成**：patch `round21_fp4_quant_head` 覆盖 `ops.predict`。
  - 通过 import 复用 r14 `_Upscale` 的结构和融合 kernel，不改 r14 的文件。
  - ConvT 的 f/d/w 按 R20 定下的 head 策略执行（bf16 / mx8 / nv4）。
- **门**
  - A：
    - 量化器 RN 与 `fp4_ref` 逐位一致；SR 在用同一个 Philox 时逐位一致，另做无偏性统计；
    - 尾块 M 覆盖 7203、7903、14406、15806、1029、33，数据覆盖尖峰、全 0、大值、极小值；
    - GEMM 对反量化后 fp32 结果的相对 L2 ≤2e-3；
    - head 对 R20 仿真的余弦 ≥0.9999。
  - B：两个模型各 ≥8 批。
  - S：4 个场景。
  - 另外：graph_equiv；导出和 resume；graph 下与 FINAL 同场次的整步 bench；nsys。
- **发布**：在 `common/config.py` 里新增 `P3H_STACK`。

### R22 `round22_fp4_backbone`：骨干（Opus；只有 R20 判定骨干有策略过门才做；≤2 h）

- 把 R20 `fp4_block.py` 里的仿真 GEMM 换成 R21 的真 kernel：
  - **激活 X**：前向时量化一次。按策略出前向用的行副本（mx8 或 nv4）；再出转置副本（加 RHT）留给 wgrad，取代原来 bf16 或 MXFP8 的列副本。
  - **dY**：量化一次。行副本（SR）给 dgrad，转置副本（RHT+SR）给 wgrad。
  - **qkv 前向**：如果 R20 判定 `qkv.f=mx8` 过门，就复用 r17 的 MXFP8 前向路径。
- QuACK 带 FP4 操作数的 epilogue 如果编不过或者偏慢，退回"普通 FP4 GEMM + r13 逐元素 kernel"，并实测这条退路。
- **门**：同 R21。A 门要求块对 R20 仿真块的余弦 ≥0.99999。
- **发布**：新增 `P3_STACK`。

### R23 `round23_fp4_final`（我，≤1 h）

1. 按模型组合出 `FINAL3`，两个模型的策略可以不同。
2. 在这个组合上重跑全部门：B ≥8 批、S 4 个场景、graph_equiv、导出、resume、`load_model`。
3. 同场次 bench R0、FINAL、FINAL3，两个模型都测；跑 e2e 200 步；出 nsys 类别表和 roofline。
4. 新写 `round23_fp4_final/run_final.py`，r19 的保持原样。
5. `common/config.py` 新增 `BEST_STACK_V3*`，V2 不动。
6. README 追加"第三阶段"一节，把瀑布表接长。
7. 更新 memory。

## 5. 门槛与事先写死的 go/no-go

- **B（FP4 放宽门，用户 2026-10-03 决定）**
  - 两个模型各 ≥8 个真实 batch，在同一进程里分别对 R0 和对 P1 计算。
  - 全局梯度余弦中位数 ≥0.98、最小 ≥0.97，loss_rel ≤2e-2。
  - 另外**只记录**：是否满足严格门（中位数 ≥0.99、最小 ≥0.98）、逐参数余弦、对 FINAL 的余弦。
  - 放宽门适用于本阶段所有降精度候选，包括重新试的 qkv/head 前向 MXFP8。
- **S**：50 步 × 2 batch，4 个场景，对 R0 和对 FINAL 的 EMA 偏差都要 ≤5%，不能有 NaN/Inf；grad-norm 比值记录下来。
- **A**：kernel 与 `fp4_ref` 逐位一致（RN）；GEMM 相对 L2 ≤2e-3；块和 head 对仿真的余弦 ≥0.9999。
- **速度**：整步只在 StepGraphRunner 下、与 FINAL 同场次比较，提升 ≥2% 才发布；算子级用 graph 回放测。
- **go/no-go**（R20 结束时套用，事后不改）：某一部分改用 FP4 需要同时满足三条：
  1. graph 回放下，比现状快：骨干 ≥1.5 倍于 MXFP8，head ≥3 倍于 bf16；
  2. 仿真策略过了 B 门和 S 门；
  3. 预计整步省 ≥3 ms 或 ≥3%。

  不满足的部分保持现状，**对应的 kernel 轮直接跳过**。
- **SR 还是 RN**：两者都过门就选 SR（长期无偏）；只有一个过门就用它。

## 6. 不覆盖旧 round 的规则与文件清单

- **新目录**（各含 `NOTES.md`、`results.json`、`run.py`、`impl/`、`tests/`、`profile/`）：
  - `train-opt/round20_fp4_probe/`
  - `round21_fp4_quant_head/`
  - `round22_fp4_backbone/`
  - `round23_fp4_final/`
  - 临时文件放 `/root/autodl-tmp/train-opt-tmp/round2x_*/`。
- **round0–19 只读**。可以 import 复用：
  - r13：`gemm_ops`、`block_kernels`、`fused_block`
  - r17：`mx8`、`gemm_mx8`、`fused_block_mx8`
  - r14：`head`、`head_cuda`
  - r15：`flat`
  - r16：`graph_step`
  - r19：`tests/gate_b_multi.py`、`check_export.py`、`run.py` 的 preset 机制

  需要改行为时，在新 round 里用子类或覆盖实现。旧测试的输出按 r19 `ro_run.py` 的方式重定向到新 round 的 `profile/`。
- **共享文件只追加**，由我改，改前备份到 `train-opt-tmp/*.bak_p6`：
  - `common/config.py`：只新增 `P3H_*`、`P3_*`、`BEST_STACK_V3*`；
  - `CONTRACTS.md`：Amendment 6 和 6b；
  - `README.md`：追加"第三阶段"一节；
  - `PLAN_v3.md`：新文件。

## 7. 执行方式

- 子代理用 `train-opt-worker`，一次只跑一个，每个任务都是新上下文、prompt 紧凑，预算 ≤1.5–2 h。
  - R20a、R20b 用 Sonnet；R21、R22 用 Opus（Agent 调用时传 `model: "opus"`）。
- 子代理只写自己的 round 目录；所有 GPU 进程持双锁、排队执行。
- 顺序：Step 0 → R20a → R20b → 我审核并定稿 6b → R21（1–2 个任务）→ R22（视决定）→ R23。
  - 预计 5–7 个子代理任务，墙钟时间约 8–10 h。
- 不在本阶段范围内：权重量化并进 AdamW、round18 的 val/ckpt、FP8/FP4 attention、round12 的集成。

## 8. Verification

**每轮**
- pytest 通过，gate A 的 JSON 写入 `profile/`。
- B、S、graph_equiv、导出的 JSON 写入 `profile/`。
- graph 下与 FINAL 同场次的三相加权 bench，两个模型都测，给出对 FINAL 和对 R0 的倍数。
- nsys 前后的 kernel 类别表。

**最终**
- FINAL3 对 R0 和对 FINAL 的倍数。
- e2e 200 步，稳态 ≥ GPU-only 的 97%。
- 导出结果能被原版 `load_model` 加载，并能在原版路径上 resume。
- 瀑布表 R0 → … → FINAL → FINAL3。
- 每个模型的最终精度档位：是否满足放宽门，是否满足严格门。

## 9. 风险与回退

| 风险 | 回退 |
|---|---|
| 放宽门下仍有 FP4 档过不了（例如 ViT-B 骨干） | 那一部分保持 MXFP8，两个模型分别定策略；只做 head 也有约 +8–10% |
| head 的 NVFP4 过不了 | head 的 dgrad/wgrad 用 MXFP8（r14 已验证的路径），约 +5–7% |
| QuACK FP4 + epilogue 编不过或者慢 | 普通 FP4 GEMM（Spark 或 cuBLASLt）+ r13 逐元素 kernel；最后才考虑 CUTLASS ex79 EVT |
| NVFP4 的 amax 扫描太贵 | amax 并进生产者 kernel，或持久化 kernel 第二遍从 L2 读，或在过门的前提下改用 MXFP4+RHT |
| 放宽门带来长训练风险（只有 few-batch 证据） | 每档都记录是否满足严格门；最终栈同时保留 FINAL（V2）和"严格档"预设，可随时切换 |
| 容器重启或额度中断 | 结果边做边落盘，每个任务可以续跑 |
