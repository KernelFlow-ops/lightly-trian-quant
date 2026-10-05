# train-opt 第二阶段计划（v3）：调研驱动的 sm_120 极限训练加速（BF16 → MXFP8，最优开源实现 + 自研 CUDA + 全步 Graph，3 个 Sonnet 子代理）

## Context

- **项目**：`/root/lightly-train/train-opt`（infer-5090，RTX 5090 sm_120，170 SM，99 KB smem/block，96 MB L2）。目标是 DINOv3+EoMT 语义分割的**训练速度最大化**，ViT-L/16 为主，ViT-B/16 为辅。
- **现状**：round0–10 是 FP32 语义下的优化。

| 模型 | R0 | P0（r1+r3+r4+r5+r7） | 倍数 |
|---|---|---|---|
| ViT-L B=7 | 16.4 img/s（425 ms） | 18.3 img/s（382 ms） | 1.12× |
| ViT-B B=14 | 37.8 img/s | 45.2 img/s | 1.20× |

- **根因**：脚本配置了 `bf16-mixed`，但 Lightly 的 `training_step` 不走 autocast，实际是 FP32/TF32 训练。ViT-L 的时间主要花在：
  - FP32 attention：反向 86 ms，前向 25 ms；
  - sm80 TF32 GEMM：约 140 ms；
  - head 的 cuDNN 卷积：约 50 ms；
  - 逐元素算子：60–70 ms，每步约 6500 个 kernel。
- **用户决策**：
  - 精度：BF16 主线，加 Blackwell MXFP8。
  - 验证：**只用 1–2 个真实 batch 测几次，不做长训练**。
  - 实现：**全面调研并整合现有高效算子工程，结合 Blackwell 新特性改进**。
  - 分工：**用 3 个 Sonnet 5.5 xhigh 子代理**，可以在原方案上创新。
  - 安装：**允许 pip 安装 CuTe DSL 相关包**，前提是先 dry-run 确认不改动现有包。
  - 沿用：conda base；不动 `src/`、`scripts/`、`infer-opt/`。

## 调研结论（决定方案方向的硬件事实）

| 事实 | 数据 | 对方案的影响 |
|---|---|---|
| GeForce 对 FP32 累加的 HMMA 限速 | BF16 规格 209.5 TFLOPS，实测 233–240（我们测得 225–234）。FP16 用 FP16 累加 350–419；普通 FP8（kind::f8f6f4）用 FP32 累加 465–517，同样减半 | BF16 主线的 GEMM 和 attention 最多约 230 TFLOPS |
| **block-scaled MMA 不限速** | `mma.sync…kind::mxf8f6f4.block_scale`：指令峰值 1014 TFLOPS；整 GEMM 实测 MXFP8 630、fp8gemm 703；NVFP4 1369 | **MXFP8 约为 BF16 的 2.7–3 倍，是 5090 上最大的单项杠杆**；所有 FP8 都必须走 block-scaled 指令 |
| sm_120 有什么 | mma.sync、TMA+mbarrier、cluster≤8、DSMEM、PDL | 没有 TMEM/tcgen05/wgmma/2-CTA，**也没有 TMA multicast**。FA3/FA4 的 sm100 路径不可用，要用 sm80 系设计加 TMA |
| 8-bit 转置限制 | ptxas 拒绝 `ldmatrix.b8 .trans`；cuBLAS 的 MXFP8 在 sm_120 只支持 TN | 每个张量都需要 K-major 的两份量化，在量化时直接写出转置份（软件 prmt 转置作为备选） |
| attention 的 BF16 已接近限速峰值 | 前向 D=128 时：cuDNN 203.6（97%），FA2 190.6，自研最高 197.7；Spark 前向是 FA2 的 1.18–1.34×，反向 1.04–1.13× | BF16 attention 的收益只能来自融合（RoPE/布局/mask）；**FP8 attention 是 BF16 的 2.63×** |
| MXFP8 训练配方 | NVIDIA：所有张量用 E4M3，scale 取 ceil/round-to-inf，8B/15T token 与 BF16 持平。Meta LP-FA4：量化融合进生产者 kernel（快 4.4–4.7×）、32×32 方块量化（转置不变）、P 用常数 scale、dS 方块量化、dQ 用 FP16 规约 | 直接作为 r17 和 attention-FP8 的设计依据 |
| 8-bit attention 训练 | SageBwd：7 个 matmul 中 6 个用 INT8，dP=dO·Vᵀ 保留 FP16，fine-tuning 无损 | 我们正是 fine-tuning，FP8 attention 前反向可以尝试 |
| 时钟和功耗 | 600 W 封顶下 BF16 约 240；FMA 1.9 GHz，mma 2.7 GHz；170 SM 用 2 的幂网格切不均匀 | persistent + stream-K 调度；逐元素 kernel 走带宽极限 |

**可直接利用的开源实现**（服务器可访问 GitHub 和阿里云 PyPI 镜像）：

- **Spark kernels**（`aidxhxr/5090-blackwell-kernels`，MIT，CUDA C++，带 torch 扩展）：
  - GEMM：hgemm 250 TFLOPS（TMA + stream-K，epilogue 支持 bias/gelu/residual）、fp8gemm 703、MXFP8 630、fp4 1369；
  - attention：bf16 前向和反向、fp8 前向（block-scaled 双乘积）；
  - 其他：rmsnorm 1538 GB/s、`quantize_mx`。
- **CuTe DSL 生态**（需要 `nvidia-cutlass-dsl`）：
  - QuACK（LayerNorm 前反向；SM120 GEMM 及 `gemm_act/gemm_dact/gemm_add` epilogue）；
  - cuDNN-frontend 1.30 的 `sdpa_bwd_wrapper_dsl_sm120`（支持 additive bias 和 dBias）；
  - FA4 的 SM120 类（`FlashAttentionForwardSm120/BackwardSm120`，D≤128，block-sparse 前向）；
  - CUTLASS PR#3030（SM120 FA2/FP8 FA）、gn-kernels（SM120 matmul，含 MXFP8/NVFP4）。
- **本地已有**：
  - CUTLASS 4.2 C++（`sm120_mma_tma`、`sm120_blockscaled_mma_tma`、sm120 EVT，示例 79/87）；
  - torch SDPA（cuDNN/FA2/mem-eff）、cuBLASLt epilogue、TileLang 和 Triton；
  - infer-opt 中为 sm_120 调过的 kernel（r3/r4/r5/r6）；
  - train-opt 的 r3 CUDA loss/matcher、r7 multi_tensor。
- **参考设计**：thu-ml/SageAttention（SageBwd，Triton）、Meta lp_fa4、NVIDIA MXFP8 recipe、QuACK 关于访存类 kernel 跑满带宽的文章。

**预估**（粗略上界，以实测为准）：

| 模型 | r11 torch-BF16 | BF16 融合 + Graph | MXFP8 GEMM（+FP8 attention） |
|---|---|---|---|
| ViT-L | 33–38 img/s | 45–52 img/s | **65–85 img/s（约 4–5× R0）** |
| ViT-B | 80–95 img/s | 110–130 img/s | **150–190 img/s** |

## 总体策略

1. **每个算子都做候选实测矩阵**：在真实训练形状下，把现成实现（torch、开源、我们已有的）全部实测一遍，取最快的。只在融合或功能有缺口时才改写或自研。改写优先基于 Spark 的 CUDA C++ 或 CUTLASS 4.2 C++，CuTe DSL 版作为强对照。
2. **精度阶梯**：先 BF16（r11 torch 版作为同精度基线），再 MXFP8 GEMM（所有 Linear 的 fwd/dgrad/wgrad，以及 head ConvT），再 FP8 attention。每一级都过 few-batch 门槛，敏感层回退 BF16。NVFP4 只做实验。
3. **门槛**（1–2 个真实 batch × 3 seed，不跑 epoch 训练）：
   - **A**：算子对 fp32 参考，覆盖尖峰 scale-8、尾块 N=1029/1129、mask 边界。
   - **B**：单步比较。
     - 同精度换 kernel：阈值用 `common/check.py` 现有值，先用 BF16 自身噪声标定。
     - 改精度：loss_rel ≤2e-2，全局 grad cos ≥0.99，逐层 cos 只记录。
   - **S**：在同样 2 个 batch 上跑 50 步过拟合轨迹，EMA 偏差 ≤5%，无 NaN/Inf。
4. **torch 作为基线**：
   - 算子级：和 torch 同精度实现做 microbench（CUDA event，同进程交替）。
   - 整步：对比 R0、r11 和上一最优。
   - 每轮 NOTES 第 3 节写清 **nsys 分析过程**（命令、NVTX 表、改动前后 Top kernel、空泡和同步）。ncu 在本容器不可用，改用 `kernel_static.sh`（ptxas、SASS 中 QMMA/HMMA/UTMALDG 的条数）加 roofline。

## 创新点

- **全 MXFP8 线性层，量化零额外开销**：
  - LN 融合量化：一次读入 32×32 tile，同时输出行向和列向两份 1×32 量化；32×32 方块量化作为对照。
  - GEMM epilogue 融合量化：fc1+GELU 后直接输出两份 MXFP8。
  - 反向的 `γ⊙dX` kernel 输出 dY 的两份量化。
  - AdamW 每步写出权重的两份 MXFP8，没有任何单独的量化 kernel。
- **head 也降精度**：ConvT（k2s2）约占 20% FLOPs，改写为 MXFP8 GEMM，pixel-shuffle 在 epilogue 写回。
- **FP8 attention**：Spark 的 block-scaled 双乘积前向；反向沿用 SageBwd 和 LP-FA4 的思路（dP 保留 BF16，dS 用 32×32 方块量化）。
- **相位无关的单张整步 Graph**：mask 的开、部分、关全部由设备数据驱动，在融合 kernel 内部分支，全程不重捕获。
- **双 stream Graph**：loss/matcher 和 head 的小分支与 backbone 重叠。
- **PDL**：后继 kernel 的 prologue 与前驱的尾部重叠。
- **梯度范数提前就绪**：wgrad epilogue 累加 Σg²，反向结束时 clip 系数已经可用，随后单趟 AdamW。
- **Cluster/DSMEM 融合 LN**（stretch）：fc2/proj 的 epilogue 用 8 个（ViT-B 是 6 个）CTA 组成 cluster 覆盖整行，在 DSMEM 上交换行统计，直接写出下一层 LN 的输出，同时完成量化。
- **RoPE 表一次生成**：每步 1 个 kernel 生成全部 block 的 RoPE 表。随机数仍由 torch 生成后传入，保证 RngTape 的 B 检查依然可比。
- **整体训练时间**：显存下降后测更大 B 的吞吐（像原脚本的 auto batch，作为次要指标）；val 走 BF16+Graph 快路径；checkpoint 异步保存；加载扩到更多 worker（机器 208 核）并在 GPU 上 normalize；各跑 1 个 epoch 的真实墙钟时间。

## 分工（3 个 `train-opt-worker`：model sonnet，effort xhigh，后台并行）

| 角色 | 轮次 | ops 接口 |
|---|---|---|
| **A 系统** | r11_bf16_mixed → r15_optimizer_flat → r16_full_graph → r18_data_val | autocast、loss 前 fp32 cast、`ops.rope`、`ops.opt_step`、`meta.forward_train_fn`、loader/val |
| **B Attention+Head** | r12_attention_sm120 → r14_head_sm120（含 FP8 attention stretch） | `ops.attn_core`、`ops.build_mask`、`ops.predict` |
| **C GEMM+低精度** | r13_gemm_fusion_sm120 → r17_mxfp8_sm120（含 LN+量化，NVFP4 实验） | `ops.block`（内部调用 `ctx.ops.attn_core`）、MXFP8 张量和 GEMM 的 API、权重量化钩子 |
| **我** | Step 0、契约、审查、`common/` 修改、round19_final | — |

**各轮的候选实测矩阵与目标：**

- **A-r11**（torch-BF16 基线，**A 的第一个交付，尽快完成**）
  - 候选：autocast；SDPA 的 cuDNN/flash/efficient；joint mask 用 bias 还是 bool；head 用 channels_last；fused AdamW；r1/r3/r4/r5/r7 在 BF16 下重新组合；Inductor 的 regional compile 和 max-autotune 作为对照。
  - 交付：两模型三相的 img/s，nsys 分类时间表。
- **A-r15**（优化器）：flat fp32 buffer，2 个 kernel 完成 clip+AdamW，lr/wd 设备表，设备端调度；写出 bf16 和 MXFP8 权重（复用 C 的量化 device 函数）。对照 torch `AdamW(fused)` 和 r7。目标 ViT-L 从 12 ms 降到约 6 ms。
- **A-r16**（整步 Graph）：
  - 修复 r10 的 capture 失败（CPU→CUDA 非 pinned 拷贝）。
  - 静态输入加侧 stream H2D；相位无关单图；双 stream；PDL 开/关对照。
  - 对照 eager、r2 segmented、`torch.compile(reduce-overhead)`。
- **A-r18**（数据与 val）：worker/prefetch/pin 扫描，uint8 H2D 后在 GPU 上 normalize；val 快路径；异步 checkpoint。测 200 步 e2e 和 1 次 val。
- **B-r12**（attention）
  - 候选：
    - torch SDPA 的 cuDNN 和 FA2；
    - cuDNN-frontend 的 sm120 DSL 反向（mask 以 bias 形式传入）；
    - FA4-sm120（cute）；
    - Spark 的 attention 和 attention_bwd，改写到 D=64、packed QKV `[B,N,3,H,64]` 输入和 `[B,N,H*64]` 输出、融合 RoPE、compact query bitmask 加整 tile 跳过、dQKV 打包输出并融合逆 RoPE；
    - r4 TileLang。
  - 融合 mask 构建 kernel：logits → 2×2 平均 → >0 → 按随机数禁用行 → bitmask `u32[B,100,32]`。
  - stretch：FP8 attention 前反向。
- **B-r14**（head）
  - ConvT(k2s2) 改写为 GEMM，按 pixel-shuffle 写回 channels-last。候选：QuACK `gemm_act`(GELU)、Spark hgemm epilogue、CUTLASS EVT，之后接 C 的 MXFP8 GEMM。
  - GELU+dwconv3×3+LN2D 做 channels-last 融合，前反向都做（LN 部分参考 QuACK）。
  - einsum 改为 batched GEMM，直接读 channels-last 输出。
  - 对照 cuDNN bf16 channels_last 和 r6 的各变体。
- **C-r13**（BF16 GEMM 融合）
  - 候选：cuBLAS/cuBLASLt 及其 epilogue、QuACK SM120 的 `gemm_act/gemm_dact/gemm_add`、Spark hgemm（stream-K+TMA）、CUTLASS 4.2 sm120 EVT、infer-opt 调过的 TileLang/MG kernel。
  - 选出最优后组装融合整块的 `autograd.Function`：
    - QKV+bias，可选 RoPE；
    - proj/fc2 做 +bias·γ+fp32 残差，aux 存 y；
    - fc1 做 +bias+GELU，aux 存 z；
    - fc2 的 dgrad epilogue 做 GELU' 和 dbias；wgrad 直接写 fp32 grad，并累加 Σg²；
    - 每块的逐元素部分合并成 2 个 kernel。
  - 对比 LN：QuACK LayerNorm、r5 Triton、Spark 风格的行驻寄存器实现。stretch：Cluster LN。
- **C-r17**（MXFP8）
  - GEMM 候选：`torch._scaled_mm` MXFP8（cuBLASLt，只支持 TN）、Spark MXFP8 和 fp8gemm、CUTLASS `sm120_blockscaled_mma_tma`（示例 79）、gn-kernels/QuACK（若支持）。
  - 量化：采用 NVIDIA 的 ceil scale 配方，用 LN+量化和 epilogue 量化的融合 kernel。
  - 层策略：从 fc1/fc2 开始，再扩到 qkv、proj 和 head（与 B 协作）。
  - 实验：NVFP4（只做 fc1 前向，基于 Spark fp4gemm）。

**协作规则**（写入 `CONTRACTS.md` 和 agent 定义）：
- **目录**：
  - 每个代理只写自己的 round 目录和 `/root/autodl-tmp/train-opt-tmp/<round>/`。
  - 第三方完整仓库 clone 到 `/root/autodl-tmp/train-opt-tmp/third_party/<repo>@<commit>`。改编的文件放进 `impl/third_party/` 并注明来源和 license。
  - `common/` 和 round0–10 对代理只读，契约变更由我统一修改。
- **GPU**：
  - **所有** GPU 进程都必须持双锁（`common/bench.gpu_lock`、`gpu_run.sh`、`profile_nsys.sh`），每次持锁不超过 10 分钟。
  - 长任务用 nohup 加日志轮询。
  - pip 安装由我统一执行，代理不装包。
- **kernel 契约**：
  - 在 current stream 上发射；无 host 同步，无数据相关形状。
  - workspace 常驻，不随 replay 释放。
  - fp32 累加。残差流、LN 统计、softmax 统计、loss、优化器保持 fp32。
- **交付与汇报**：
  - 每轮交付 `NOTES.md`（7 节模板）、`results.json`（未测字段写 null）、`tests/`、`profile/`。
  - 整步不快于最优现成实现的，只作为实验记录。
  - 每个阶段结束向我汇报实测表、失败项和下一个瓶颈。

## 执行步骤

**Step 0（我）**
1. **安装**：
   - 先 `pip install --dry-run` 检查 `nvidia-cutlass-dsl`（cu13 变体）、`quack-kernels`、`nvidia-cudnn-frontend` 的依赖；如会改动 torch/numpy/cuda 相关包，则用 `--no-deps` 或暂停问用户。
   - 安装后跑 import smoke 和现有 CPU pytest，确认环境无回归。
   - 把版本记录到 `PLAN_v2.md`。
2. **第三方源码**：clone Spark kernels、flash-attention（cute）、SageAttention、gn-kernels 到 third_party 目录，记录 commit。
3. **agent 定义**：本地新建 `.claude/agents/train-opt-worker.md`，仿照 infer-opt-worker，`model: sonnet`、`effort: xhigh`，规则换成 train-opt 的路径和锁。若当前会话识别不到新类型，回退到 `general-purpose` + `model: sonnet`，把规则写进 prompt，并告知用户。
4. **项目文档**：在服务器写入 `train-opt/PLAN_v2.md`（本计划的中文版，含调研表）和 `train-opt/CONTRACTS.md`，契约包括：
   - 张量布局：fp32 残差；LN 输出 bf16 或 MX8；QKV `[B,N,3,H,64]`；attention 输出 `[B,N,H*64]`；RoPE 表 fp32 `[1024,64]`，prefix=N−1024。
   - bitmask 格式 `u32[B,100,32]`。
   - `MX8Tensor`：E4M3 数据、UE8M0 swizzled scale、方向标记、可选转置副本。
   - predict 输出：fp32 的 `[B,Q,128,128]` 和 `[B,Q,K+1]`。
   - 随机数输入约定、可捕获性约定、flat 优化器 API。
5. **代码**：
   - `common/ops.py` 增加 `attn_core`，默认 torch SDPA，与现有 `_attn` 数学等价；默认 `attn` 改为 qkv → `attn_core` → proj。用 CPU pytest 加 GPU 上 default-vs-default 的 B 检查验证无回归。
   - `common/config.py` 预留 `P1_STACK`/`BEST_STACK_V2`；`common/summarize.py` 支持 r11–r19。

**Phase 1（并行）**：A 做 r11 后接 r15，B 做 r12，C 做 r13。A 交付 r11 的 `t_bf16` 参考配置后，我转告 B 和 C 用作 B/S 门槛基准。

**Phase 2（并行，用 SendMessage 续接同一代理）**：A 做 r16，B 做 r14 和 FP8 attention，C 做 r17。

**Phase 3**：A 做 r18；我做 round19_final：
- 组合 `BEST_STACK_V2`，接入单图 runner。
- 测固定 B 和最大可行 B 下的三相加权 img/s。
- nsys、roofline，以及 R0→P0→r11→…→r19 的瀑布表。
- 两模型各跑 1 个 epoch 的墙钟时间（train+val+ckpt）。
- 最终栈对 FP32 做 B/S 检查。
- 导出兼容：state_dict 键和形状，`load_model().predict()` smoke。
- 更新 README 和 `results/summary.md`。

## 关键文件

- **新增**：
  - `train-opt/{PLAN_v2.md, CONTRACTS.md}`
  - `train-opt/round11_bf16_mixed … round19_final/`，各含 `{NOTES.md, run.py, results.json, impl/(含 third_party/), tests/, profile/}`
  - 本地 `.claude/agents/train-opt-worker.md`
- **修改**（只由我改）：`common/ops.py`（`attn_core`）、`common/config.py`、`common/summarize.py`。
- **复用**：
  - `common/{harness,bench,check,rng,roofline,peak_bench,fast_forward}.py`
  - `common/{profile_nsys.sh,nsys_analyze.py,kernel_static.sh,gpu_run.sh}`
  - r3 的 `cuda_ops.cu` 和 `fast_criterion.py`；r4 的 `rope.py`、`mask.py`、`tma_attention.py`；r5 的 `fused_norm.py`、`lt_epilogue`；r6 的 `fused_head.py`；r7 的 `multi_tensor.cu`、`patch.py`；r9 的 `mxfp8.py`
  - r10 的 `train.py`、`full_graph.py`（r16 复制后修复）、`verify_export.py`
  - infer-opt 的 r3/r4/r5/r6（只读）
  - CUTLASS 头文件：`site-packages/cutlass_library/source/include`

## Verification

- **每轮**：
  - pytest 通过；A/B/S 门槛的 JSON 写进 `profile/`。
  - 持锁的三相整步计时，两模型都测，给出相对 R0、r11 和上一最优的倍数。
  - 算子级 microbench 对 torch 同精度及各开源候选。
  - nsys 前后对比写进 NOTES。
- **最终**：
  - BEST_STACK_V2 对 R0 的固定 B 倍数，以及最大 B 吞吐。
  - e2e（200 步真实 loader）与 GPU step 的差距 ≤3%。
  - 1 个 epoch 墙钟时间对比。
  - 最终栈对 FP32 的 B/S 通过；导出兼容通过。
  - `python -m common.summarize` 重新生成汇总。
- **执行后更新 memory**：
  - 用户偏好：快速 batch 级验证、不跑长训练；追求速度极限；允许 dry-run 后 pip 安装。
  - 5090 的 FP32 累加限速，以及 block-scaled MXFP8 不限速的结论。
  - 第二阶段的分工和契约。

## 调研来源

- 5090 MMA 指令实测：[gau-nernst NVRTC MMA](https://gau-nernst.github.io/nvrtc-matmul/)、[HN 关于 209.5 TFLOPS 的讨论](https://news.ycombinator.com/item?id=44997957)、[flashinfer #3628（MXFP8 不限速）](https://github.com/flashinfer-ai/flashinfer/issues/3628)
- 5090 kernel：[Spark / 5090-blackwell-kernels（MIT）](https://github.com/aidxhxr/5090-blackwell-kernels)、[gau-nernst FA for 5090](https://gau-nernst.github.io/fa-5090/)、[gn-kernels](https://github.com/gau-nernst/gn-kernels)
- sm_120 限制：[FA4 不能跑 5090 的分析](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739)、[cluster 支持（8，无 multicast）](https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471)、[sm120 上 ldmatrix b8 的限制](https://github.com/NVIDIA/cutlass/pull/3030)、[cuBLAS 的 MXFP8 在 sm_120 只支持 TN](https://github.com/pytorch/ao/issues/4932)
- attention：[cuDNN SDPA bwd SM120](https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/attention/sdpa_bwd_sm120.html)、[flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)、[CUTLASS PR#3030 SM120 FA2/FP8](https://github.com/NVIDIA/cutlass/pull/3030)、[LP-FA4 MXFP8 前反向](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)、[SageAttention3/SageBwd](https://arxiv.org/abs/2505.11594)
- 低精度训练：[NVIDIA MXFP8 预训练配方](https://arxiv.org/abs/2506.08027)、[TE 不支持 SM120 MXFP8](https://forums.developer.nvidia.com/t/transformer-engine-and-gb10-mxfp8-and-mxfp4-training-not-yet-supported/351220)
- 访存类 kernel 与 GEMM epilogue：[QuACK](https://github.com/Dao-AILab/quack)
- cuBLAS 在 5090 上的 dispatch 问题：[cloudrift](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090)

## 附：Step 0 执行记录（2026-10-02）

- **安装**：dry-run 确认只新增包后安装，前后的 freeze 在 `/root/autodl-tmp/train-opt-tmp/pip_freeze_{before,after}_phase2.txt`，新增内容如下：
  - einops 0.8.2
  - nvidia-cudnn-frontend 1.30.0
  - nvidia-cutlass-dsl 4.8.0（含 libs-base/core/cu12/cu13）
  - quack-kernels 0.6.5
- **smoke 测试**（持锁）：
  - QuACK 的 `layernorm_fwd` 正确，JIT 耗时 0.3 s。
  - QuACK 的 SM120 BF16 GEMM：形状 7203×4096×1024 下为 0.254 ms / **238 TFLOPS**，`torch.mm`（cuBLAS）为 0.297 ms / 203 TFLOPS。
- **第三方源码**：放在 `/root/autodl-tmp/train-opt-tmp/third_party/`，commit 见 `SOURCES.txt`：
  - spark = aidxhxr/5090-blackwell-kernels@a401bfa9e363
  - gn-kernels@46cb3fb01c33
  - SageAttention@d1a57a546c3d
  - flash-attention@843bf0b86bda
- **契约**：`train-opt/CONTRACTS.md`。
- **子代理定义**：本地 `.claude/agents/train-opt-worker.md`（sonnet，xhigh）。
