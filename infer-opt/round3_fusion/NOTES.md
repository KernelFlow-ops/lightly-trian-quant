# Round 3：Transformer block 算子融合（TileLang + CUDA）

工作目录 `/root/lightly-train/infer-opt/round3_fusion/`（worker "blockfuse"）。下面所有数字都来自本目录里的脚本实测（`results.json`、`nsys/`、`static/`），表格由 `make_tables.py` 直接从 `results.json` 生成。

## 1. 目标与结论

把每层 eager 的约 40 个 kernel（bf16-pure 30 个，autocast 40 个）压成 **7 个**：LN1 → qkv+RoPE → SDPA → proj+残差 → LN2 → fc1+GELU → fc2+残差。attention 暂时仍用 torch SDPA（flash），round4 再换。

- 4 个融合 kernel 全部正确（cos ≥ 0.999994，误差在 bf16 舍入量级，且比 torch 自己的 bf16 链更接近 fp32 真值）。
- 单 kernel 在 kernel-time 口径下都不慢于对应的 torch 多算子序列：K1 约 1.5–2.2×，K2 约 2.6×（17 个 kernel → 1 个），K3/K4 在 ViT-L 上约 1.1–1.15×、ViT-B 上 1.0–1.4×（GEMM 主循环本身与 torch cutlass 基本打平，收益来自吃掉 bias/GELU/LayerScale/残差这些 elementwise kernel；ViT-B 的 fc1/fc2 与 torch 持平，1.00–1.08×）。
- **整层**（ViT-L，N=1029，真实 GPU 忙时间）：eager autocast 551 µs → fused 367 µs；eager bf16-pure 486 µs → fused 361–367 µs。event 计时 727 → 405 µs（1.8×）。
- **整个 model 阶段**（batch=2，24 层全部换成 fused block）：ViT-L 17.05 ms → **11.25 ms**（1302 → 233 个 kernel）；ViT-B 8.03 ms → **4.73 ms**（682 → 153 个 kernel）。
- **重要精度发现**：残差流必须用 **fp32**（autocast 的行为）。bf16 残差流在 ViT-L 上模型级门槛不过（conf 相对误差 9.71%，pooled IoU 0.9846，单帧 IoU 0.915）；fp32 残差流通过（ViT-L conf 2.51%，ViT-B 0.31%）。原因：ViT-L 从第 0 层起出现 |x|≈1.5e5 的 massive activation（register token 等），bf16 的 ulp 在这个量级是 1024。K1、K3 都支持 fp32 残差（x/r/out 的 dtype 可以是 bf16 或 fp32），`fused_block` 按输入 x 的 dtype 自动选择，**集成时请传 fp32 的 x**。

## 2. 设计

| kernel | 实现 | 要点 |
|---|---|---|
| K1 `add_layernorm(x, r, w, b)` | CUDA C++ (`impl/ln_kernels.cu`, torch cpp_extension) | 一个 warp 一行，128-bit（uint4 / 2×float4）向量化；x、r 可以是 bf16 或 fp32，w/b 为 fp32，统计量在 fp32；输出 s（新残差，dtype 同 x）和 y=LN(s)（bf16）。r=None 时只做 LN，且不写 s。bf16 x 时先把 x+r 舍入成 bf16 再算统计（与 eager 的 bf16 加法一致）。d=1024/768 模板特化。 |
| K2 `qkv_rope` | TileLang (`impl/gemm_tl.py`) | GEMM 累加器（fp32）+ K 段被 mask 的 bias → 写入 smem（fp32）→ 同 head 内 (j, j^32) 配对做 rotate_half（fp32，sin/cos fp32 [1024,64]，token 下标 ≥ prefix 才转，prefix 为编译期参数：5 或 105）→ 只舍入一次成 bf16，直接写 `[3,B,N,H,64]` 物理布局（返回逻辑 `[3,B,H,N,64]` 的 permute view）。物理 bnhd 布局让 flash 直接输出 `[B,N,H,64]`，`o.transpose(1,2).reshape` 变成零拷贝（之前用 bhnd 布局时 SDPA 后多一个 copy kernel，改成 bnhd 后整层 8→7 个 kernel）。也提供 `layout="bhnd"` 和 `round_bf16=True`（先把 linear 输出舍入成 bf16 再 rope，逐位对齐 eager）。 |
| K3 `gemm_bias_residual(h,W,b,r,gamma)` | TileLang | `out = r + gamma*(h@W^T + b)`，gamma（LayerScale）是 fp32 向量，在 fp32 epilogue 里乘；r/out dtype 可以是 bf16 或 fp32。 |
| K4 `gemm_bias_gelu(h,W,b)` | TileLang | fp32 累加 + bias → erf-GELU → 一次舍入成 bf16。 |
| `prepare_block_weights(block, fold_gamma=False)` | Python | K 段 bias 乘 bias_mask 预计算、权重转 bf16 contiguous、LN affine 保持 fp32、LayerScale 保存为 fp32 向量 g1/g2。`fold_gamma=True` 时按任务书把 γ 折进 W'/b'（bf16 舍入）。 |
| `fused_block(x, w, sin, cos, prefix, heads)` | Python | 7 个 kernel，见上。 |

**为什么默认不折 γ 到权重里**：任务书建议 W'=diag(γ)W。实测折叠版本在整层和整模型上精度基本相同（见 §4 的 `foldgamma` 行，ViT-L conf 2.72% vs 2.51%），没有本质区别；但 epilogue 里乘 fp32 γ 更接近 eager（`x + γ*linear`）且免去 γ 很大（joint 层 γ≈9.5–17）时的权重舍入，代价为零（一个 fp32 向量读），所以默认走 epilogue，`fold_gamma=True` 保留备用。

**sm_120 上的 TileLang 行为**（用 `get_kernel_source()` + nvcc + cuobjdump 静态确认，见 §6）：
- 只有 `mma.sync`（SASS 里是 HMMA），没有 wgmma/tcgen05。
- 默认代码生成走 **TMA（UTMALDG）+ mbarrier + warp specialization**（384 线程 = 256 消费者 + 128 生产者，`setmaxnreg`），在 sm_120 上编译、运行都正确。`pass_configs={TL_DISABLE_WARP_SPECIALIZED, TL_DISABLE_TMA_LOWER}` 可退回 cp.async（LDGSTS）+ ldmatrix 的 Ampere 式流水线。两种都可用，调参时都扫了，不同形状最优不同（`ops.TILE_CFG`）。
- 所有 kernel 0 spill；GEMM 168 寄存器（qkv_rope 218，因为 fp32 smem staging + rope）。

## 3. 正确性（Level A：`test_kernels.py`，真实模型形状 ViT-L/ViT-B × N=1029/1129，取 4 个形状里的最差值）

对比对象：`torch bf16`（同 dtype 的 torch 算子序列）和 `fp32`（fp32 真值）。"torch vs fp32" 是 torch 自己的 bf16 链相对 fp32 的误差，作为舍入底线参照。

| kernel | 对比 | min cos | max_abs |
|---|---|---|---|
| K1 add_layernorm（bf16） | s vs torch | 1.0000000 | 0 |
| | y vs torch bf16 | 0.999996 | 0.031 |
| | y vs fp32 / torch vs fp32 | 0.999997 / 0.999995 | 0.030 / 0.044 |
| K1（fp32 残差，x fp32 + r bf16/fp32） | s vs fp32 | 1.0000000 | 0 |
| | y vs fp32（bf16 输出 1 ulp） | 0.999999 | 0.0156 |
| K4 gemm_bias_gelu | vs torch bf16 | 0.999995 | 0.031 |
| | vs fp32 / torch vs fp32 | 0.999999 / 0.999997 | 0.0156 / 0.0161 |
| K3 fc2（K=d_ff） | vs torch bf16（含 γ*，eager 形式） | 0.999994 | 0.0625 |
| | vs fp32 / torch vs fp32 | 0.999999 / 0.999995 | 0.031 / 0.065 |
| | fp32 残差输出 vs fp32 | 1.0000000 | 4.2e-5 |
| K3 proj | vs torch bf16 | 0.999995 | 0.0625 |
| | vs fp32 / torch vs fp32 | 0.999999 / 0.999996 | 0.031 / 0.066 |
| | fp32 残差输出 vs fp32 | 1.0000000 | 1.3e-5 |
| K2 qkv_rope（`round_bf16=True`，与 eager 同舍入） | q / k vs torch | 1.0000000 | 0.0078 / 0.0156 |
| | v vs torch | 1.0000000 | 0 |
| K2（默认：fp32 累加器上 rope，一次舍入） | q / k / v vs fp32 | 0.999999 | 0.0156 |
| | torch q / k vs fp32 | 0.999997 | 0.029 / 0.031 |

结论：bf16 GEMM 的 cos 全部 ≥ 0.999994（要求 0.9999）；默认（fp32 累加器上 rope）的 K2 比 torch 的 bf16 链更接近 fp32（误差只剩最终一次 bf16 舍入）。

### Level B：真实权重的整层对比（`test_block.py`，输入是真实帧上 fp32 前向捕获的每层 x，ViT-L 层 0/12/21/22/23，ViT-B 层 0/9/10，2 帧）

参照是 fp32 eager block。指标 rel_l2 = ‖out−ref‖/‖ref‖。注意：ViT-L 第 0 层输出含 |x|≈1.5e5 的 massive activation，rel_l2 基本由少数几个元素决定（单元素舍入就是 ~4e-3 量级），所以整层 rel_l2 有很强的"运气"成分，**真正的判据是 §4 的模型级门槛**。

| ViT-L 层 | eager bf16 | eager autocast | fused bf16 残差 | fused fp32 残差 | fused fp32 残差 + 折 γ |
|---|---|---|---|---|---|
| 0（frame0） | 1.81e-3 | 1.81e-3 | 1.37e-3 | 3.99e-3 | 4.05e-3 |
| 12 | 1.25e-3 | 4.64e-5 | 1.25e-3 | 3.61e-5 | 3.89e-5 |
| 21 | 2.22e-3 | 5.09e-4 | 2.15e-3 | 4.22e-4 | 4.14e-4 |
| 22（joint） | 2.23e-3 | 6.26e-4 | 2.13e-3 | 5.56e-4 | 5.67e-4 |
| 23（joint） | 1.87e-3 | 6.69e-4 | 1.66e-3 | 5.30e-4 | 5.07e-4 |

- 层 12–23：fused fp32 残差 ≤ eager autocast 的误差；fused bf16 残差和 eager bf16 一样差（1e-3 量级），是 bf16 残差流本身的误差。
- 层 0 fused fp32 残差 rel_l2 更大（3.99e-3 vs 1.81e-3），单独拆开验证过（`/root/autodl-tmp/infer-opt-tmp/blockfuse/dbg_l0.py`）：从同一个精确 x1 出发，MLP 半层 fused 1.48e-3 vs autocast 3.96e-3；同一个 bf16 h2 上，fc1→GELU→fc2 的 fused 相对 fp64 链误差 4.3e-5，eager 是 2.9e-3（eager 把 linear 输出先舍入成 bf16 再乘 γ）。也就是每一段单独看 fused 都更准，整层 rel_l2 差异来自 massive-activation 元素上误差方向的偶然相消。ViT-B 上 fused fp32 残差在每一层都 ≤ eager autocast（例如层 9：2.19e-3 vs 2.62e-3）。

## 4. 模型级精度门槛（`test_model_fused.py`，8 帧真实视频，`mf.gate`：pooled head IoU ≥ 0.95，每帧 conf 相对误差 ≤ 5%）

把 24（12）层全部换成 `fused_block`，其余（patch embed、query 拼接、backbone.norm、head）按原样 autocast。

| 模型 | 变体 | pooled IoU | min 帧 IoU | max conf 相对误差 | 通过 |
|---|---|---|---|---|---|
| ViT-L | eager autocast（基线） | 0.9931 | 0.9898 | 1.11% | 是 |
| ViT-L | fused，**fp32 残差** | 0.9938 | 0.9793 | 2.51% | 是 |
| ViT-L | fused fp32 残差，rope 先舍入 bf16 | 0.9926 | 0.9669 | 3.30% | 是 |
| ViT-L | fused fp32 残差，折 γ | 0.9941 | 0.9833 | 2.72% | 是 |
| ViT-L | fused，**bf16 残差** | 0.9846 | 0.9149 | **9.71%** | **否** |
| ViT-B | eager autocast | 0.9926 | 0.9870 | 0.38% | 是 |
| ViT-B | fused，fp32 残差 | 0.9932 | 0.9827 | 0.31% | 是 |
| ViT-B | fused，bf16 残差 | 0.9900 | 0.9841 | 0.63% | 是 |

ViT-L fused fp32 残差的 conf 误差 2.51% 高于 autocast 基线的 1.11%，但在 5% 门槛内；各个子选项（rope 舍入方式、折 γ）之间的差异在 2.5–3.3% 内，属于同一噪声量级，没有找到可以确定改善它的开关。`logits_cos` 全部 ≥ 0.999999。

## 5. 性能

计时口径说明：
- **kernel-time**：`torch.profiler` 的 CUPTI kernel 时长之和 / 调用数，即纯 GPU 忙时间，不含 launch 间隙，也不含 CUDA event 本身的 ~10 µs 开销，是评价小 kernel 最准的口径（`nsys` 实测整层结果与之吻合，见下）。
- **gpu us (events)**：`bench.time_fn` 的 CUDA event p50（每次迭代 sync，包含 kernel 间隙和 launch 抖动），对多算子序列反映"实际调用成本"。
- 峰值：bf16 GEMM 8192³ = 232 TFLOPS；torch 在模型形状上 M=1029 的 GEMM 峰值 qkv 126 / proj 131 / fc1 168 / fc2 168。
- 机器上还有其他 worker 的 CPU 负载（load average 14–27），个别行有抖动（例如 `k2_qkv_rope_vitl16_N1129` 的 torch 侧 324.9 µs 是一次噪声，同形状 N=1029 是 188.5 µs；ViT-B 的 K3 fc2 一次 event 计时也出现过 +15%）。GPU 侧的 kernel-time 更稳定。所有计时都持有 GPU 锁。

### microbench K1-K4 (gpu = CUDA-event p50 per call incl. launch gaps of multi-op torch sequences; wall = host view)
| kernel | ours gpu us | torch gpu us | speedup | ours kernel-time us (n) | torch kernel-time us (n) | speedup (kernel) | ours TFLOPS by kernel-time (% of 232) | torch linear-only kernel us | tiles / waves |
|---|---|---|---|---|---|---|---|---|---|
| k1_add_ln_fp32res_vitl16_N1029 | 21.3 | 33.0 | x1.55 | 6.9 (1) | 12.9 (2) | x1.87 | - | - | - |
| k1_add_ln_vitl16_N1029 | 15.6 | 32.7 | x2.10 | 4.2 (1) | 7.6 (2) | x1.80 | - | - | - |
| k1_ln_only_vitl16_N1029 | 18.8 | 24.2 | x1.28 | 3.3 (1) | 5.1 (1) | x1.55 | - | - | - |
| k4_gemm_gelu_vitl16_N1029 | 112.9 | 123.6 | x1.09 | 95.8 (1) | 106.9 (2) | x1.12 | 180 (78%) | 100.5 | 1088 / 6.40 |
| k3_fc2_vitl16_N1029 | 114.1 | 127.8 | x1.12 | 96.7 (1) | 106.9 (3) | x1.11 | 179 (77%) | 100.5 | 136 / 0.80 |
| k3_proj_vitl16_N1029 | 46.0 | 50.7 | x1.10 | 28.4 (1) | 32.7 (3) | x1.15 | 152 (66%) | 26.2 | 264 / 1.55 |
| k2_qkv_rope_vitl16_N1029 | 87.7 | 202.7 | x2.31 | 71.7 (1) | 188.5 (17) | x2.63 | 181 (78%) | 69.3 | 816 / 4.80 |
| k1_add_ln_fp32res_vitl16_N1129 | 21.4 | 33.4 | x1.56 | 7.5 (1) | 13.4 (2) | x1.78 | - | - | - |
| k1_add_ln_vitl16_N1129 | 21.3 | 33.1 | x1.55 | 4.6 (1) | 8.2 (2) | x1.80 | - | - | - |
| k1_ln_only_vitl16_N1129 | 18.7 | 24.6 | x1.32 | 3.6 (1) | 5.5 (1) | x1.56 | - | - | - |
| k4_gemm_gelu_vitl16_N1129 | 113.5 | 123.2 | x1.09 | 97.1 (1) | 107.7 (2) | x1.11 | 195 (84%) | 100.6 | 1152 / 6.78 |
| k3_fc2_vitl16_N1129 | 115.3 | 128.0 | x1.11 | 96.8 (1) | 107.5 (3) | x1.11 | 196 (84%) | 100.4 | 144 / 0.85 |
| k3_proj_vitl16_N1129 | 45.5 | 50.4 | x1.11 | 28.6 (1) | 33.2 (3) | x1.16 | 166 (71%) | 26.3 | 288 / 1.69 |
| k2_qkv_rope_vitl16_N1129 | 99.2 | 206.8 | x2.09 | 82.6 (1) | 324.9 (17) | x3.93 | 172 (74%) | 156.3 | 864 / 5.08 |
| k1_add_ln_fp32res_vitb16_N1029 | 21.0 | 33.1 | x1.58 | 5.0 (1) | 10.9 (2) | x2.17 | - | - | - |
| k1_add_ln_vitb16_N1029 | 21.2 | 32.9 | x1.55 | 3.6 (1) | 6.8 (2) | x1.90 | - | - | - |
| k1_ln_only_vitb16_N1029 | 18.7 | 24.5 | x1.31 | 2.8 (1) | 4.8 (1) | x1.76 | - | - | - |
| k4_gemm_gelu_vitb16_N1029 | 75.8 | 80.6 | x1.06 | 60.6 (1) | 65.5 (2) | x1.08 | 160 (69%) | 60.5 | 1584 / 9.32 |
| k3_fc2_vitb16_N1029 | 79.7 | 80.8 | x1.01 | 62.2 (1) | 62.0 (3) | x1.00 | 156 (67%) | 56.5 | 396 / 2.33 |
| k3_proj_vitb16_N1029 | 34.4 | 41.1 | x1.19 | 18.4 (1) | 25.4 (3) | x1.38 | 132 (57%) | 20.1 | 396 / 2.33 |
| k2_qkv_rope_vitb16_N1029 | 58.3 | 201.4 | x3.45 | 43.6 (1) | 112.7 (17) | x2.58 | 167 (72%) | 41.0 | 594 / 3.49 |
| k1_add_ln_fp32res_vitb16_N1129 | 21.0 | 33.3 | x1.59 | 5.4 (1) | 11.3 (2) | x2.09 | - | - | - |
| k1_add_ln_vitb16_N1129 | 21.0 | 33.0 | x1.57 | 3.6 (1) | 7.3 (2) | x2.04 | - | - | - |
| k1_ln_only_vitb16_N1129 | 18.8 | 24.6 | x1.31 | 2.7 (1) | 5.1 (1) | x1.88 | - | - | - |
| k4_gemm_gelu_vitb16_N1129 | 80.2 | 81.5 | x1.02 | 65.6 (1) | 66.3 (2) | x1.01 | 163 (70%) | 60.5 | 1728 / 10.16 |
| k3_fc2_vitb16_N1129 | 80.1 | 80.9 | x1.01 | 62.6 (1) | 62.7 (3) | x1.00 | 170 (73%) | 56.9 | 432 / 2.54 |
| k3_proj_vitb16_N1129 | 34.8 | 41.9 | x1.20 | 18.7 (1) | 25.9 (3) | x1.38 | 142 (61%) | 20.2 | 432 / 2.54 |
| k2_qkv_rope_vitb16_N1129 | 59.0 | 201.2 | x3.41 | 44.8 (1) | 115.1 (17) | x2.57 | 179 (77%) | 41.6 | 648 / 3.81 |

### whole layer
| model | layer / variant | kernels | kernel-time us (GPU busy) | gpu us (events) | wall us |
|---|---|---|---|---|---|
| vitl16 | normal_N1029_eager_bf16 | 30 | 486.3 | 612.4 | 627.1 |
| vitl16 | normal_N1029_eager_autocast | 40 | 551.4 | 726.8 | 741.6 |
| vitl16 | normal_N1029_fused_fp32res | 7 | 366.9 | 404.6 | 418.4 |
| vitl16 | normal_N1029_fused_bf16res | 7 | 361.0 | 400.3 | 414.2 |
| vitl16 | joint_N1129_eager_bf16 | 30 | 501.5 | 613.3 | 628.0 |
| vitl16 | joint_N1129_eager_autocast | 40 | 568.3 | 728.2 | 743.2 |
| vitl16 | joint_N1129_fused_fp32res | 7 | 382.1 | 422.3 | 436.1 |
| vitl16 | joint_N1129_fused_bf16res | 7 | 376.3 | 414.8 | 428.6 |
| vitb16 | normal_N1029_eager_bf16 | 30 | 342.2 | 516.9 | 531.4 |
| vitb16 | normal_N1029_eager_autocast | 40 | 387.8 | 627.8 | 642.6 |
| vitb16 | normal_N1029_fused_fp32res | 7 | 254.6 | 290.8 | 304.7 |
| vitb16 | normal_N1029_fused_bf16res | 7 | 254.0 | 290.2 | 304.5 |
| vitb16 | joint_N1129_eager_bf16 | 30 | 348.8 | 522.1 | 536.5 |
| vitb16 | joint_N1129_eager_autocast | 40 | 393.8 | 629.6 | 644.0 |
| vitb16 | joint_N1129_fused_fp32res | 7 | 261.9 | 300.9 | 314.9 |
| vitb16 | joint_N1129_fused_bf16res | 7 | 260.8 | 298.6 | 312.4 |

### model stage (batch=2 crops)
| vitl16 | eager_autocast | gpu 17.05 ms | wall 17.06 ms | kernels 1302 |
| vitl16 | fused_fp32res | gpu 11.25 ms | wall 11.27 ms | kernels 233 |
| vitb16 | eager_autocast | gpu 8.03 ms | wall 8.04 ms | kernels 682 |
| vitb16 | fused_fp32res | gpu 4.73 ms | wall 4.75 ms | kernels 153 |


### nsys 实测整层（ViT-L 普通层 N=1029，fp32 残差，`nsys/block_L_fused_fp32res*`，10 次迭代取 p50，每个 kernel 的 grid/寄存器来自 sqlite）

| # | kernel | grid | 时长 µs | 折算 |
|---|---|---|---|---|
| 0 | add_ln（LN1，fp32 x → bf16 y） | 515 | 4.0 | |
| 1 | qkv + bias + RoPE + 布局（TileLang，128×64 tile，cp.async） | 48×17 | 72.0 | 180 TFLOPS（77.6% 的 232 峰值） |
| 2 | flash_fwd（torch SDPA） | 9×2 | 59.5 | round4 换掉 |
| 3 | proj + γ + 残差（fp32） | 8×33 | 29.9 | 144 TFLOPS |
| 4 | add_ln（LN2） | 515 | 4.7 | |
| 5 | fc1 + bias + GELU | 64×17 | 96.4 | 179 TFLOPS（77% 峰值） |
| 6 | fc2 + γ + 残差（fp32） | 8×17 | 99.4 | 174 TFLOPS（75%） |

整层一次迭代：kernel 忙时间 365.6 µs，首尾跨度 376.1 µs（kernel 间隙只有 ~10 µs）。同一份 nsys，eager autocast 的整层是 40 个 kernel、553 µs 忙时间（16 种 kernel：cutlass GEMM 63.5×2 / 101.4 / 69.9 µs、flash 59.3、120 个 4.3 µs 的 elementwise、40 个 CatArrayBatchedCopy 9.3 µs、LayerNorm 9.9 µs……）。**GEMM 加起来两边都是 ≈298 µs**，融合省下的 ~187 µs 全部来自 elementwise / cat / cast / layernorm。

## 6. 静态分析（ncu 不可用的替代，`static_analysis.py` → `static/`）

TileLang 生成的 CUDA 用 `-I tilelang/src -I tilelang/3rdparty/cutlass/include -DENABLE_BF16` 编成 cubin，再用 `kernel_static.sh bin`。

| kernel | 寄存器 | spill | HMMA | UTMALDG（TMA） | LDGSTS（cp.async） | LDSM | 备注 |
|---|---|---|---|---|---|---|---|
| K4 fc1+GELU（ViT-L） | 168 | 0 | 64 | 64 | 0 | 40 | 384 线程，TMA + mbarrier + warp specialization |
| K3 fc2（fp32 残差） | 168 | 0 | 64 | 16 | 0 | 24 | 同上 |
| K3 proj（fp32 残差） | 168 | 0 | 32 | 64 | 0 | 16 | 同上 |
| K2 qkv_rope | 218 | 0 | 96 | 0 | 18 | 36 | cp.async 流水线，rope epilogue 走 smem |
| K1 add_ln（12 个模板实例） | 52–65 | 0 | 0 | 0 | 0 | 0 | 纯 LDG/STG 向量化，无 smem；4.0–4.7 µs |

## 7. 调参（`tune_gemm.py`，结果在 `tune_*.json`）

每个 GEMM 扫 tile（128×128 / 128×64 / 64×128 / 64×64 / 128×32）× stages（3/4）× {256 线程 + warp spec + TMA，128 线程 无 ws 无 TMA，128 线程 + ws + TMA}。最终配置在 `ops.TILE_CFG`：

- ViT-L fc1：128×64，bk32，4 stages，256 线程，TMA+ws；fc2：128×128，4 stages，256 线程，TMA+ws（136 个 tile = 0.80 wave）；proj：64×128，256 线程，TMA+ws（264 tile）；qkv：128×64，128 线程 cp.async（fp32 staging 需要小 smem）。
- ViT-B：fc1 64×64（TMA+ws），fc2 / proj 64×64（cp.async），qkv 64×128（cp.async）。
- fc2 / proj 的 wave 问题：128×128 只有 136 个 tile（<170 SM，0.8 wave）。小 tile（64×128 / 64×64，2–3 wave）在 proj 上更好（29.9 µs），在 fc2 上打平（128×128 略优）。**没有尝试 split-K**（需要原子/二次归约，收益在 <20% 空闲 SM 内，且 fp32 残差用原子加会引入非确定性），是可以继续的方向。
- 第一次 sweep 的 noise 较大（同一个 config 重复测差 ±10%），所以表里只用 `bench_kernels.py` 在最终配置上的重复测量。

## 8. TileLang / sm_120 使用中遇到的限制

1. 辅助函数里含 `for ... in T.Pipelined(...)` 这类循环必须用 `@T.macro` 装饰，否则报 `'ForFrame' object is not iterable`。
2. 含 `T.prim_func` 的模块不要 `from __future__ import annotations`：延迟求值的注解引用外层局部变量（如 `HW`）时报 `NameError`。
3. `T.prim_func` 的形状是编译期常量：qkv_rope 每个 `(B, N_tok, d, prefix)` 编译一次（约 3–6 s / kernel），`lru_cache` 缓存；集成时启动前要预热（joint 层 N=1129、普通层 N=1029 是两个实例）。
4. 编译 TileLang 生成的 CUDA 做静态分析要自己加 `-I tilelang/src -I .../cutlass/include -DENABLE_BF16`（`kernel_static.sh cu` 默认不带这些 include，所以我用 `bin` 模式）。
5. T.gemm 只会发 mma.sync；没有更快的 sm_120 专用路径（没有 wgmma）。GEMM 达到 57–84% 的 232 TFLOPS（kernel-time 口径；ViT-L qkv/fc1/fc2 在 75–84%），与 torch cutlass_80 的 linear 相当或略快，但离 8192³ 峰值仍有余量。
6. `T.erf`、`T.if_then_else`、`T.Parallel` 里的 python `if`、共享内存里的任意下标读（`c_s[i, j ^ 32]`）都能用，TileLang 会自动插同步。

## 9. 文件清单

- `impl/ops.py`：Python API（`add_layernorm`、`qkv_rope`、`gemm_bias_residual`、`gemm_bias_gelu`、`prepare_block_weights`、`fused_block`、`TILE_CFG`）
- `impl/gemm_tl.py`：TileLang K2/K3/K4；`impl/ln_kernels.cu`：K1；`impl/__init__.py`
- `test_kernels.py`：Level A；`test_block.py`：真实权重整层；`test_model_fused.py [--time]`：模型级门槛与 model 阶段耗时
- `bench_kernels.py`：K1–K4 microbench；`bench_block.py`：整层 + kernel 数（`--profile <variant>` 供 nsys）；`tune_gemm.py`；`static_analysis.py`；`make_tables.py`
- `results.json`、`tune_*.json`、`nsys/`（block_L_fused_fp32res / block_L_eager_autocast 的 .nsys-rep 与 CSV）、`static/`（生成的 .cu、cubin、SASS、ptxas 输出）

## 10. 遗留问题与下一步

- attention 仍是 torch flash（59.5 µs，占融合后整层 16%），round4 替换；K2 的 bnhd 输出布局是给 SDPA 用的，自己的 attention kernel 可以要求别的布局，改 `layout` 参数即可。
- K1 与前一个 GEMM 的 epilogue 还可以再融合（GEMM 产出整行再做 LN 需要 BLOCK_N=d，tile 太胖；可以做成"下一个 GEMM 的 prologue 里做 LN"，暂未做）。
- fc2 / proj 的 wave 利用率（split-K / stream-K）没做。
- 只在 batch=2 的形状上调过 tile；batch 变化（M 变了）要重新扫。
- 没有接入 CUDA Graph（kernel 都在 current stream 上、除输出外不再分配，理论上可以直接 capture；TileLang kernel 的输出在每次调用时由 torch 分配，capture 时要用 graph 的 mempool）。
- 模型级 conf 相对误差在 ViT-L 上是 2.51%（autocast 基线 1.11%），在门槛内，但比基线大。后续如果叠加 FP8 等更激进的优化，需要预留这部分余量。
