# Round14：EoMT head 融合（Agent B2，2026-10-03）

STATUS: FINAL（所有数字来自 `profile/*.json`、`nsys/` 与 `logs/`，见各节；未测项目明确写 "not measured"）。会话额度中断一次、服务器换过一次容器，已测数据未重跑。

## 0. 摘要

- **做了什么**：`ops.predict` 整体重写（补丁 `round14_head_sm120`，变体 `fused`）。ConvTranspose(k2,s2) 改成 GEMM `[B·h·w,C]x[C,4C]`（QuACK/cuBLAS，**不物化 pixel shuffle**），GELU+dwconv3x3+LayerNorm2D 融合成 3 个自研 CUDA kernel（前向 1、反向 2，pixel shuffle 在读/写端按地址完成），einsum 用 batched GEMM（反向走 QuACK）。upscale+einsum 是**一个 autograd.Function**，Parameter 身份和 state_dict 不变，可被 CUDA graph 捕获（已测）。
- **单次 predict（fwd+bwd，CUDA event，同进程交替）**：ViT-L 11.34 → 7.21 ms（1.57x），ViT-B 15.66 → 8.99 ms（1.74x）。nsys 的 kernel 时间 10.96 → 6.94 ms（ViT-L）、15.24 → 8.66 ms（ViT-B）。
- **整步（三相加权 img/s，50×3，预热 20，同机，持锁）**：

| 模型 | P1R13（TB1+r13） | P1R13 + r14 | P2（+r17 MXFP8） | P2 + r14 |
|---|---:|---:|---:|---:|
| ViT-L B=7 | 43.83 | **48.56**（1.108x） | 51.96 | **59.07**（1.137x vs P2，1.348x vs P1R13） |
| ViT-B B=14 | 101.29 | **125.63**（1.240x） | 110.35 | **139.95**（1.268x vs P2，1.382x vs P1R13） |

  mid 步时：ViT-L 160.0 → 144.2 ms（P2：134.7 → 118.2），ViT-B 138.3 → 111.6 ms（P2：127.2 → 100.0）。ViT-L 省 15.8–16.5 ms（要求 ≥ 4 ms）。峰值显存 ViT-L 17.48 → 15.76 GB，ViT-B 15.31 → 12.38 GB。
- **门**：A 通过（输出 + 全部参数梯度 + x 梯度，gauss 与 scale-8 尖峰数据，两个模型；误差比 TB1 的 torch bf16 head 更小）。B（校准准则，对 TB1_R13）：ViT-L 7/7 批通过，**ViT-B 8/9 批通过，1 批在逐参数准则上失败**（layer-0/1 的 LN/proj 参数，全局指标全过，详见 5.2）。S（50 步、2 个真实 batch，schedule 与 mid 两相）：两个模型全过（最大 EMA 偏差 ≤ 2.6e-3）。
- **可选 MXFP8 ConvT（单独报告，未完成全部门）**：`mx8_bwd`（dgrad/wgrad 走 MXFP8，前向保持 bf16）在 P2_R14 上再 +4.0%（ViT-L 61.45 img/s）/ +2.7%（ViT-B 143.70）；门 B 只在 ViT-L 跑了 3 批（`mx8_bwd` 3/3，`mx8` 全部 GEMM 2/3），ViT-B 的门 B、≥8 批规则、门 S 和 graph 捕获测试均 **未做**，所以**不推荐发布**。
- **未解决/要提的事**：(1) ViT-B 门 B 的 1 次失败（5.2）；(2) eager 下 P2_R14 的 GPU 忙碌率只有 0.746（>20 µs 的间隙 22.7 ms/步），head 变快之后整步更是 host 受限，要等 round16 的整步 graph 才能兑现 kernel 时间 104 ms；(3) 剩下的 head 时间 68% 是 bf16 GEMM（已在 ~230 TFLOPS 限流上限）。

## 1. 目标与假设

round11 的 nsys 里 head 是 ViT-L 22 ms/步（4 次 predict）、ViT-B 30 ms/步的 "head conv" 加上一部分 elementwise；单次 predict（fwd+bwd）11.3 ms / 15.7 ms（TB1，`head_ln_bf16`）。
本轮先把基线按 kernel 拆开（第 3 节）：ViT-L 的 10.96 ms 里 GEMM 类（cuDNN ConvT 前向/dgrad/wgrad + einsum + mask_head）约 5.5 ms，elementwise / 布局类（bias 加、GELU、dwconv、LN2D、fp32 LN 统计、各种 permute/copy、bias/γ/β 梯度归约）约 5.4 ms。假设：

- **H1**：ConvT 的 cuDNN kernel 已经在 bf16 峰值（~230 TFLOPS）附近，GEMM 部分在 bf16 下降不下去；可省的是 elementwise / 布局部分，目标 ≥ 3.5 ms/predict。round6 的变体变慢（14.7 → 18–29 ms，round11 §4.1 实测）。读代码的判断（未单独测）：Triton kernel 一个 program 一个像素（9 邻点各读一遍、GELU 重算 9 次）、反向拆成 4–5 个 kernel（其中 `_dw` 为 9 个 program 各读一遍全图）、GEMM 路径带 permute copy。本轮的 Triton v1 已经改成按图像行流式、3 个 kernel，孤立测量就比 torch 快 2.5–3.2x（4.1 (c)），说明数据布局和遍历方式才是问题。
- **H2**：k2s2 ConvT 的 pixel shuffle 不需要在 GEMM epilogue 里 scatter：GEMM 直接写 `[B·h·w, 4C]`，之后的 dwconv kernel 在**读取端**按 `(P>>1, Q>>1, 相位)` 取址即可（每个像素是一段连续的 C 向量，没有合并访问损失）。
- **H3**：einsum 用 `[100,C]x[C,HW]` 的 batched GEMM 直接吃 channels-last 的 LN2D 输出；反向的两个小-M GEMM 用 QuACK（cuBLAS 在 K=100 / 长 K 时很差）。
- **H4（可选）**：ConvT 的 3 个 GEMM 换成 MXFP8（round17 的量化器 + QuACK 块缩放 GEMM）理论上把 4.4 ms 的 GEMM 部分砍半，但改变精度，要单独过门。

## 2. 改动（全部在 `train-opt/round14_head_sm120/`）

- `impl/head.py`：`ops.predict` 的实现。`backbone.norm`、`class_head`、`mask_head` 仍是 torch（很小）；ConvT×2 + GELU + dwconv + LN2D×2 + einsum 是**一个 `torch.autograd.Function`**（`_Upscale`），参数（fp32 Parameter）直接作为输入，反向返回 fp32 梯度，autograd 负责累加，不改 Parameter 身份和 state_dict。
  GEMM 策略 `POLICY`（来自第 4.1 节的候选矩阵）：ConvT fwd / dgrad 用 QuACK，wgrad 用 cuBLAS（fp32 输出），einsum 前向 torch.bmm（fp32 输出），einsum 反向 QuACK。`mx8` 开关（`mx8` / `mx8_bwd` / `mx8_fwd` / `mx8_dgrad` / `mx8_wgrad`）让 ConvT 的对应 GEMM 走 round17 的 `mx8.quantize` + `gemm_mx8`（QuACK block-scaled）。
  输出契约：fp32 连续 `[B,100,128,128]` mask logits（GEMM 直接写 fp32）和 fp32 `[B,100,3]` class logits。不满足形状条件（grid≠32×32、C∉{768,1024}、token 数不符）时回退到前一个 `ops.predict`。
- `impl/head_kernels.cu` + `impl/head_cuda.py`：三个 CUDA kernel（sm_120，JIT 编到 tmp build 目录）：
  `fwd_kernel`（bias + GELU + dwconv3x3 + LN2D，一个 CTA 沿 Q 方向流式处理一段图像行，线程 = C/CPT，每线程 CPT 个通道，LN 统计是一次块规约；3×3 卷积用"按列累加"，每读一列 3 个像素更新 3 个输出，寄存器里只留 2 个部分和；GELU 每个读入元素只算一次，单次 exp 的 A&S 7.1.26 erf，|误差| < 1.5e-7）、
  `bwd1_kernel`（重算 u，LayerNorm 反向，输出 dU 与 dγ/dβ 的逐 CTA 部分和）、`bwd2_kernel`（dU 窗口 → dh，GELU'，直接按 ConvT GEMM 的布局写 dZ，同时累加 dW 的 9 个 tap 与 dbias）。全 fp32 规约，无数据 atomics（逐 CTA 部分和写工作区，之后 `.sum(0)`），结果可复现。
- `impl/head_triton.py`：同样三个 kernel 的 Triton 版（v1 朴素；v2 预取 + 合并规约），作为候选和备份（`R14_KERNELS=triton`）。
- `impl/patch.py`：补丁 `round14_head_sm120`，变体 `fused`（默认）、`torch_gemm`、`quack_gemm`（消融）、`mx8*`。依赖 round11 autocast；与 round13/17 的 `ops.block` 无关，可叠在它们之后。
- `run.py`：从 round17 的 run.py 改来（bench / profile / gate_b / gate_s / maxbatch），新增预设 `TB1_R14`、`TB1_R13_R14`、`P2`、`P2_R14`、`MXH_*`（MXFP8 head 变体，门 B 按"改精度"准则）。
- `tests/`：`bench_gemm.py`（ConvT / einsum GEMM 候选矩阵）、`bench_stage_matrix.py`（elementwise 候选矩阵）、`test_triton_stage.py`（单级 kernel 对 fp32 参考 + 扫描，`--impl cuda|triton`）、`test_gateA.py`（整个 head 对 fp32 源码实现，输出 + 全部梯度）、`bench_head_r14.py`（单次 predict fwd+bwd）、`test_graph.py`（CUDA graph 捕获/回放）、`host_cost.py`、`prof_head.py` / `profile_head.py` / `nsys_kernels.py` / `nsys_groups.py` / `nsys_step_summary.py`（分析脚本）、`gate_b_multi.py` / `gate_s_multi.py`（从 round17 改来；门 B 加入了 Amendment 2 的 "OR cos(cand,ref) ≥ 0.99" 条款）。
- 第三方代码：没有拷贝。erf 用公开的 Abramowitz–Stegun 7.1.26 公式；GEMM 用已安装的 QuACK / cuBLAS / round17 的 MXFP8 模块（`impl/third_party/` 为空）。

## 3. nsys 分析过程

**命令**（全部持双锁；`profile_nsys.sh` 在 GPU 上有他人上下文时会拒绝，用重试包装 `/root/autodl-tmp/train-opt-tmp/round14_head_sm120/run_nsys_head.sh`；原始 `.nsys-rep/.sqlite` 在 `/root/autodl-tmp/train-opt-tmp/round14_head_sm120/nsys/`，项目里只放 json 摘要）：

```bash
# (a) 单次 predict（fwd + bwd 各一个 NVTX 范围，预热 20，采 5 次）：基线 torch head（TB1 的 head_ln_bf16）与 r14
bash common/profile_nsys.sh $O/nsys/head_{torch,r14}_{vitl16,vitb16} round14_head_sm120/tests/profile_head.py --model <m> --variant {torch|r14}
python round14_head_sm120/tests/nsys_kernels.py <prefix> [--list --step 2]     # 有序 kernel 表 + 每 kernel 总时间
python round14_head_sm120/tests/nsys_groups.py  <prefix>...                    # 按子算子分组
# (b) 整步（mid 相，预热 20，5 步）：TB1_R13、TB1_R13_R14、P2_R14（ViT-L）；TB1_R13、TB1_R13_R14（ViT-B）
bash common/profile_nsys.sh $O/nsys/step_<model>_<stack> round14_head_sm120/run.py --model <m> --stack <stack> --action profile --phases mid --warmup 20
python common/nsys_analyze.py $O/nsys/step_<model>_<stack> --iters 5 --top 8
python round14_head_sm120/tests/nsys_step_summary.py $O/nsys/step_<model>_<stack> --json profile/nsys_step_<model>_<stack>.json
```
NVTX：`:fwd`、`:bwd`、`:predict_*`、`:blk_*`、`:loss`、`:clip_opt_sched`；r14 额外加了 `r14_up_fwd` / `r14_up_bwd`（`ctx.nvtx` 打开时；关闭时输出逐位相同，已测）。

**3.1 基线：一次 predict（ViT-L，B=7，TB1 torch head，kernel 总时间 10.96 ms：fwd 3.10 + bwd 7.86）**

kernel 级表（nsys 5 次平均，µs；`tests/nsys_kernels.py`）：

| 子算子 | kernel | fwd µs | bwd µs |
|---|---|---:|---:|
| ConvT GEMM（cuDNN） | stage1+2 前向（`cutlass__5x_cudnn dgrad_optimized 128x128`，实为 ConvT 前向） | 1330 | |
| | dgrad（`cudnn fprop_optimized` + `sm80_xmma_fprop_implicit_gemm`） | | 1339 |
| | wgrad（`sm80_xmma_wgrad_implicit_gemm 256x128` + `cutlass wgrad 128x256`，+bias 梯度 reduce 160） | | 1707 + 160 |
| bias 加 + GELU | `elementwise_kernel`（bias）350 + `GeluCUDAKernelImpl` 317 | 667 | |
| GELU 反向 | `GeluBackwardCUDAKernelImpl` ×2（+ 小的 ×3） | | 493 |
| dwconv3×3 | `conv2d_c1_k1_nhwc`（前向）/ `dgrad2d_c1_k1_nhwc`、`wgrad2d_c1_k1_nhwc`（反向） | 391 | 376 + 488 |
| LayerNorm2D | `vectorized_layer_norm_kernel<bf16>`（前向）/ `layer_norm_grad_input` + `GammaBetaBackward` + permute 拷贝（反向） | 351 | 517 + 446 + **1101（`direct_copy`，NHWC↔NCHW 布局拷贝）** |
| einsum | `cutlass_80 64x64_tn`（前向）/ `wmma 32x32_nt` 437 + `64x256_nn` 375（反向 dE、dY） | 186 | 812 |
| 其余（backbone.norm fp32 LN、mask_head 3 个 Linear、class_head、reduce/copy 小 kernel；与总数的差额） | | ~180 | ~420 |

分组（ViT-L / ViT-B，µs per predict；`nsys_groups.py`）：

| 组 | ViT-L torch | ViT-L r14 | ViT-B torch | ViT-B r14 |
|---|---:|---:|---:|---:|
| GEMM（ConvT / einsum / mask_head Linear） | 5524 | 4704 | 6794 | 5317 |
| stage elementwise（GELU / dwconv / LN / bias；r14 为融合 kernel） | 3487 | 1767 | 5522 | 2818 |
| 布局拷贝 / reduce / 其它 | 1949 | 468 | 2920 | 527 |
| **合计** | **10960** | **6938** | **15236** | **8662** |
| 其中 fwd / bwd | 3102 / 7859 | 2103 / 4835 | 4294 / 10943 | 2573 / 6090 |

**结论 3.1**：基线 head 约 40% 是 ConvT 的 3 个 GEMM（4.4 ms，实测约 200 TFLOPS，已在 bf16 限流上限附近）、约 9% 是 einsum（1.0 ms；cuBLAS 在 K=100 和 M=100/长 K 时只有 50–60 TFLOPS）、**约 44% 是 elementwise / 布局**（bias、GELU、dwconv、LN2D 及它们反向的 permute 拷贝，共约 4.8 ms，其中 LN2D 反向前的一次 permute 拷贝就是 1.1 ms）、其余约 5%。所以收益主要来自融合（H1）；GEMM 部分的可抠空间是 ~0.3 ms（ConvT，4.38 → 4.12 ms）+ ~0.45 ms（einsum，1.0 → 0.55 ms）。

**3.2 r14 之后（ViT-L 一次 predict，6.94 ms）**：QuACK ConvT fwd 1277（2 个）、dgrad 1038 + 小 kernel、cuBLAS wgrad 1415（2 个）、QuACK einsum 反向/小 GEMM 649（3 个）、融合 kernel 3 个：`bwd2` 632 + `bwd1` 593 + `fwd` 439 = 1.66 ms；einsum 前向 cutlass 174；其余 < 0.5 ms。整个 predict 剩下 ~16 个小 kernel 的布局拷贝/reduce（0.47 ms）。

**3.3 整步（mid 相，5 步平均；kernel 总时间 / 墙钟 / 忙碌率 / >20 µs 间隙 / 同步）**

| 栈 | kernel ms/步 | kernels/步 | 忙碌率 | >20µs 间隙 | 同步/步 | NVTX 步墙钟 GPU 投影 |
|---|---:|---:|---:|---:|---:|---:|
| ViT-L TB1_R13 | 148.59 | 3010 | 0.900 | 99.8 个 / 6.28 ms | 8 | 163.5 ms |
| ViT-L TB1_R13 + r14 | **132.29** | 2912 | 0.860 | 175.6 个 / 11.37 ms | 8 | 152.9 ms |
| ViT-L P2 + r14 | **104.10** | 3198 | **0.746** | 409.8 个 / **22.71 ms** | 8 | 138.7 ms |
| ViT-B TB1_R13 | 127.27 | 2048 | 0.890 | 134.6 个 / 7.32 ms | 8 | 142.1 ms |
| ViT-B TB1_R13 + r14 | **100.58** | 1920 | 0.847 | 166.8 个 / 9.13 ms | 8 | 117.7 ms |

（P2 无 r14 的整步 nsys 本轮没有重采，round17 NOTES §3.1 的 MX8_fc+proj+qkv_dw 为 kernel 120.4 ms / 忙碌率 0.794，head conv 22.3 ms。）

类别表（ms/步；`nsys_step_summary.py` 的正则分类，按 kernel 名）：

| 类别 | ViT-L TB1_R13 | ViT-L +r14 | ViT-L P2+r14 | ViT-B TB1_R13 | ViT-B +r14 |
|---|---:|---:|---:|---:|---:|
| head conv（cuDNN ConvT / dwconv 等） | **22.34** | 0.73* | 0.73* | **30.39** | 1.11* |
| head 融合 kernel（r14 CUDA） | – | **6.70** | 6.69 | – | **10.34** |
| GEMM bf16 QuACK | 49.43 | 61.35 | 42.61 | 29.82 | 43.32 |
| GEMM bf16 cuBLAS/cutlass sm80 | 20.92 | 21.96 | 7.32 | 11.17 | 13.26 |
| GEMM MXFP8（QuACK，名字与 bf16 相同，按上两行计入） | – | – | （含在 QuACK 行） | – | – |
| MXFP8 量化 | – | – | 5.06 | – | – |
| elementwise / copy / reduce / 其它 | 17.31 | 8.24 | 8.42 | 25.63 | 10.54 |
| LN / 行 kernel（r13 CUDA + torch LN） | 10.47 | 5.22 | 5.90 | 12.87 | 4.64 |
| attention（flash/cuDNN SDPA/RoPE） | 16.26 | 16.24 | 16.42 | 12.87 | 12.86 |
| 优化器 + clip | 11.60 | 11.61 | 10.72 | 4.04 | 4.04 |
| loss / matcher / mask | 0.24 | 0.24 | 0.24 | 0.48 | 0.48 |
| **合计** | **148.59** | **132.29** | **104.10** | **127.27** | **100.58** |

\* 剩下的 "head conv" 是正则 `cudnn` 匹配到的 joint 块 cuDNN SDPA kernel，不是 head。

head 在 kernel 时间里的份额（predict 单独 nsys × 4）：ViT-L 43.8 ms（29.5%）→ 27.8 ms（21.0% of 132.3）；ViT-B 61.0 ms（47.9%）→ 34.6 ms（34.4% of 100.6）。NVTX（GPU 投影，平均每次）：`predict_*` 前向范围 3.13–3.19 ms → 2.13–2.45 ms（ViT-L），4.33–4.38 → 2.60–2.80（ViT-B）；`r14_up_fwd` 2.02 / 2.46 ms，`r14_up_bwd` 4.64 / 5.71 ms（ViT-L / ViT-B，P2_R14 下相同）。

**结论 3.3**：(1) head 的 kernel 时间 ViT-L −16.3 ms/步、ViT-B −26.7 ms/步，与整步 bench 的 mid 步时变化（−15.8 / −26.7 ms）吻合，说明这次收益基本是 GPU 侧的真实节省。(2) kernel 总时间降得比墙钟多一点点：>20 µs 的间隙从 6.3 → 11.4 ms（ViT-L），P2_R14 下达到 22.7 ms/步、忙碌率 0.746，整步是 **host/launch 受限**（CONTRACTS 10.3 的结论），r14 本身每次 predict 的 host enqueue 反而更省（ViT-L 5.93 → 5.46 ms，ViT-B 6.62 → 4.28 ms，`tests/host_cost.py`），不是新增瓶颈。(3) 同步数不变（8/步）。

## 4. 实测结果

### 4.1 算子级候选矩阵（真实形状，CUDA event，预热 20、3 × 30 次迭代取中位数，同进程交替；`profile/bench_gemm.json`、`profile/bench_stage_matrix.json`）

**(a) ConvT(k2,s2) 的三个 GEMM**（`[M=B·h·w, K=C] x [C, N=4C]`；wgrad 输出 fp32；TFLOPS 按 2·M·K·N 计）

| 形状 | GEMM | cuBLAS `torch.mm` | QuACK (tuned) | QuACK split-K auto | cuDNN ConvTranspose2d（torch 路径） |
|---|---|---:|---:|---:|---:|
| ViT-L stage1 M=7168 | fwd | 0.315 ms (191 T) | **0.268 (225 T)** | – | 0.331 (182 T) |
| | dgrad | 0.298 (202) | **0.289 (208)** | – | |
| | wgrad | **0.306 (196)** | 0.339 (178) | 0.338 | |
| ViT-L stage2 M=28672 | fwd | 1.124 (214) | **1.038 (232)** | – | 1.431 (168) |
| | dgrad | 1.166 (206) | **1.071 (225)** | – | |
| | wgrad | **1.143 (210)** | 1.349 (178) | 1.345 | |
| | fwd+dgrad+wgrad 合计 | 3.43 ms | 3.46 | | 3.895（185 T） |
| ViT-B stage1 M=14336 | fwd | 0.325 (208) | **0.296 (229)** | – | 0.443 (153) |
| | dgrad | 0.305 (222) | **0.295 (229)** | – | |
| | wgrad | **0.328 (206)** | 0.343 (198) | 0.337 | |
| ViT-B stage2 M=57344 | fwd | 1.260 (215) | **1.183 (229)** | – | 1.834 (148) |
| | dgrad | 1.293 (209) | **1.189 (228)** | – | |
| | wgrad | 1.306 (207) | 1.360 (199) | 1.307 (207) | |
| | 合计（cuDNN 三个 kernel） | | | | 4.967（163 T） |

读法：k2s2 ConvT 作为 GEMM 就是一个 `nn.Linear(C → 4C)`，QuACK 的 sm_120 GEMM 在 fwd/dgrad 上是 225–232 TFLOPS（bf16 带限流的上限附近），比 cuBLAS 快 6–8%，前向比 cuDNN 的 ConvT 快 19–35%；wgrad 要 fp32 输出，cuBLAS 反而更好。**bf16 下 ConvT 这三个 GEMM 已经在硬件上限上，要再降只能换 MXFP8**（第 4.5 节）。

**(b) einsum `[B,100,C] x [B,C,16384]`**（Y 读取量 ViT-L 235 MB / ViT-B 352 MB）

| 操作 | torch.bmm | QuACK | 选择 |
|---|---:|---:|---|
| ViT-L fwd（fp32 输出） | bf16 输出 0.167 / fp32 输出 0.178 ms | 0.174 | torch.bmm fp32 输出（持平） |
| ViT-L dY = dLᵀ·E（K=100） | 0.451 | **0.182** | QuACK（2.5x） |
| ViT-L dE = dL·Y（K=16384，M=100） | 0.399（fp32 输出 0.386） | 0.197 / split-K auto **0.180** | QuACK（2.1x） |
| ViT-B fwd | 0.280 / 0.289 | 0.281 | torch |
| ViT-B dY | 0.758 | **0.280** | QuACK（2.7x） |
| ViT-B dE | 0.406 | **0.257** | QuACK（1.6x） |

**(c) GELU + dwconv3x3 + LN2D（bias 加在内）的 elementwise 部分**，一个 ScaleBlock stage，ms（fwd / fwd+bwd；torch 候选拿到的是已经 pixel-shuffle 过的 NHWC 张量，即给 cuDNN ConvT 的写出布局免费；融合候选读未 shuffle 的 GEMM 输出；反向 = bwd1 + bwd2 含部分和归约）

| 形状（一次遍历 = 特征图读/写一遍） | torch bf16 (TB1) | Triton v1 | Triton v2 | **CUDA（采用）** |
|---|---:|---:|---:|---:|
| ViT-L stage1 → 64² (59 MB, 0.033 ms @1.79 TB/s) | 0.476 / 1.347 | 0.171 / 0.550 | 0.162 / 0.483 | **0.095 / 0.417** |
| ViT-L stage2 → 128² (235 MB, 0.131 ms) | 2.039 / 5.743 | 0.556 / 1.817 | 0.507 / 1.648 | **0.355 / 1.446** |
| ViT-B stage1 → 64² (88 MB, 0.049 ms) | 0.781 / 2.156 | 0.359 / 1.099 | 0.250 / 0.830 | **0.165 / 0.631** |
| ViT-B stage2 → 128² (352 MB, 0.197 ms) | 3.062 / 8.659 | 1.377 / 3.966 | 0.943 / 2.836 | **0.535 / 2.173** |

CUDA 版比 torch 快 4.7–5.7x（fwd）/ 3.2–4.0x（fwd+bwd），比 Triton v2 快 1.4–1.8x（fwd）/ 1.14–1.32x（fwd+bwd）。ViT-L stage2 的 DRAM 遍历下限（一遍 = 235 MB；前向读 Z 写 Y = 2 遍；bwd1 读 Z、dY 写 dU = 3 遍；bwd2 读 Z、dU 写 dZ = 3 遍；共 8 遍）在 1.79 TB/s（取的假设值，r17 在拷贝类 kernel 上观察到约 1.85 TB/s）下 ≈ 1.05 ms，实测 1.446 ms ≈ 73%。
Triton 版为什么比 CUDA 版慢没有诊断（没有 ncu）；假设是指令发射受限（逐元素地址/掩码、GELU 在相邻行 CTA 里各算一次）而不是带宽受限；预取 + 合并规约（v2）只省 7–10%。CUDA 版用 4/8 通道的向量化装载、按列累加（每读一列 3 个像素更新 3 个输出）、每个读入元素只算一次 GELU（单次 exp 的 erf）。
CUDA 配置扫描（ViT-L stage2，`tests/test_triton_stage.py --impl cuda --bench`）：前向 CPT=8、每 CTA 64 像素最好（0.360 ms，CPT=4 0.373，整行 0.377–0.395）；反向 CPT=4 + 每 SM 至少 2 个 CTA 的 launch bound 最好（bwd1 0.505、bwd2 0.547 ms；不加 launch bound 时 bwd1 0.689 / bwd2 0.590——bwd2 的 dW 累加器让寄存器到 162，只能放 1 个 CTA/SM）；每 CTA 多行（RPB=4）会把 CTA 数降到 224 以下，反而慢（0.76–0.96 ms）。

### 4.2 一次 predict（fwd+bwd 与仅 fwd），整个 head 的候选矩阵（`tests/bench_head_r14.py`，`profile/bench_head_final.json`；真实形状，预热 20、3×20 次取中位数，候选交替）

| 候选（ms） | ViT-L fwd+bwd | ViT-L fwd | ViT-B fwd+bwd | ViT-B fwd |
|---|---:|---:|---:|---:|
| TB1 torch head（`head_ln_bf16`，基线） | 11.344 | 3.276 | 15.655 | 4.478 |
| **r14 fused（采用：QuACK fwd/dgrad，cuBLAS wgrad，QuACK einsum 反向）** | **7.214**（1.57x） | **2.236** | **8.993**（1.74x） | **2.761** |
| r14，GEMM 全部 cuBLAS | 7.928 | 2.395 | 9.854 | 2.867 |
| r14，GEMM 全部 QuACK | 7.455 | 2.221 | 9.160 | 2.768 |
| r14 + MXFP8 ConvT 全部 GEMM（eager，见 4.5） | 7.838 | 1.506 | 7.098 | 2.062 |
| r14 + MXFP8 ConvT dgrad+wgrad（`mx8_bwd`） | 7.194 | 2.229 | 7.870 | 2.770 |
| r14 + MXFP8 ConvT wgrad（`mx8_wgrad`） | 7.127 | 2.233 | 8.578 | 2.773 |
| 历史（round11 §4.1，同机型）：round6 `fused_dwln` / `convt_gemm|dwln|all` | 20.4 / 18.25 | | 29.1 / 25.5 | |

ViT-L 的 MXFP8 行在 eager 单 predict 微基准里反而没快（`runs` 抖动大：全部 GEMM 的三次为 5.33 / 8.32 / 7.84 ms）：host enqueue 7.3–7.8 ms（`tests/host_cost.py`，量化 kernel + block-scaled 派发的 Python 开销）超过 GPU 时间，微基准被 host 限制；GPU 时间见 4.5（torch.profiler 的 kernel 总和 7.26 → 4.87 ms）。

### 4.3 整步（`run.py --action bench --phases all`，warmup 20、50×3，B 固定，三相 0.2/0.6/0.2 加权；`profile/bench_<model>_<stack>_all.json`）

| 模型 | 栈 | early / mid / late ms | 加权 img/s | vs P1R13 | vs P2 | 峰值 GB | mid 的 fwd / loss / bwd / opt ms |
|---|---|---|---:|---:|---:|---:|---|
| ViT-L | P1R13 = `TB1_R13` | 159.8 / 160.0 / 158.8 | 43.83 | 1.000x | | 17.48 | 43.5 / 4.0 / 98.4 / 12.9 |
| ViT-L | **P1R13 + r14**（`TB1_R13_R14`） | 145.2 / 144.2 / 143.0 | **48.56** | **1.108x** | | 15.76 | 39.5 / 3.9 / 86.3 / 13.2 |
| ViT-L | P2（`P2` = r13 + r17 `fc+proj+qkv_dw`） | 135.1 / 134.7 / 134.6 | 51.96 | 1.185x | 1.000x | 17.23 | 40.4 / 4.0 / 76.2 / 12.9 |
| ViT-L | **P2 + r14**（`P2_R14`） | 120.3 / 118.2 / 117.7 | **59.07** | 1.348x | **1.137x** | 15.51 | 36.3 / 3.7 / 64.2 / 12.8 |
| ViT-L | P2 + r14 `mx8_bwd`（`MXH_P2_bwd`，可选，未全部过门） | 114.7 / 113.7 / 113.7 | 61.45 | 1.402x | 1.183x | 15.60 | 37.1 / 3.7 / 58.6 / 13.1 |
| ViT-B | P1R13 = `TB1_R13` | 139.7 / 138.3 / 136.3 | 101.29 | 1.000x | | 15.31 | 36.9 / 6.4 / 86.9 / 7.0 |
| ViT-B | **P1R13 + r14** | 112.9 / 111.6 / 109.6 | **125.63** | **1.240x** | | 12.38 | 29.9 / 6.1 / 67.7 / 7.0 |
| ViT-B | P2 | 127.9 / 127.2 / 124.9 | 110.35 | 1.089x | 1.000x | 14.80 | 36.0 / 6.6 / 76.4 / 7.2 |
| ViT-B | **P2 + r14** | 101.3 / 100.0 / 98.7 | **139.95** | 1.382x | **1.268x** | 11.87 | 29.1 / 5.8 / 56.9 / 7.3 |
| ViT-B | P2 + r14 `mx8_bwd`（可选） | 97.6 / 98.1 / 95.2 | 143.70 | 1.419x | 1.302x | 12.04 | 30.5 / 6.2 / 52.0 / 8.3 |

相对 R0（CONTRACTS 9.3 的 16.2 / 37.2 img/s；本轮没有重测 R0）：P2+r14 为 3.65x / 3.76x。同批次测得的 P2 与协调者给的 51.6 / 110.0 相差 ≤ 0.7%（run-to-run）。步时节省（mid）：ViT-L −15.8 ms（P1R13）/ −16.5 ms（P2），ViT-B −26.7 / −27.2 ms，均与 3.3 的 nsys kernel 节省吻合。收益在 P2 上比在 P1R13 上大（1.137x 对 1.108x），因为 MXFP8 已经把块的时间压低，head 占的比例更高。

### 4.4 head 的 GEMM 部分（`tests/bench_gemm.py`，见 4.1 的 (a)(b)）与融合 kernel 的参数扫描
见 4.1；扫描结果在 4.1 (c) 末尾。此外整个 stage kernel 与 fp32 参考的误差见 5.1（`tests/test_triton_stage.py`，两个实现都测）。

### 4.5 可选：MXFP8 ConvT 变体（单独报告）

实现：`head.py::_ConvT`，`mx8` 可选 fwd / dgrad / wgrad 子集；量化器和块缩放 GEMM 复用 round17（`mx8.quantize` 双输出 + `gemm_mx8`，QuACK block-scaled），einsum 保持 bf16。

| 项 | ViT-L | ViT-B |
|---|---|---|
| 单次 predict kernel 时间（torch.profiler，`prof_head.py`） | bf16 7.256 → mx8 全部 **4.872 ms**（量化 kernel 0.407 ms，6 个；QuACK 块缩放 GEMM 835 + 662 + 370 µs） | not measured（nsys 版：只有 bf16） |
| 单次 predict host enqueue（fwd+bwd） | bf16 5.46 → `mx8_bwd` 7.32 → mx8 全部 7.76 ms | 4.28 → 6.91 → 7.48 ms |
| 整步加权 img/s：P2_R14 → `mx8_bwd` | 59.07 → **61.45**（+4.0%） | 139.95 → **143.70**（+2.7%） |
| 门 A（对 fp32，mask logits cos / 参数梯度 cos min / rel max） | 全部 GEMM：0.99887 / 0.99834 / 5.8e-2；`mx8_bwd`：mask 与 bf16 相同（0.99998）/ 0.99864 / 5.2e-2；`mx8_wgrad`：0.99927 / 3.8e-2 | 全部：0.99871 / 0.99803 / 6.3e-2；`mx8_bwd`：0.99856 / 5.4e-2；`mx8_wgrad`：0.99925 / 3.9e-2 |
| 门 B（改精度准则：loss_rel ≤ 2e-2、全局 grad cos ≥ 0.99，对 R0 与 TB1_R13；3 批） | `mx8_bwd` **3/3**（对 ref 全局 cos 最小 0.99978、对 R0 最小 0.99697，loss_rel ≤ 8.7e-5）；`mx8` 全部 GEMM **2/3**（失败的一批：对 R0 的全局 cos 0.98915，对 ref 0.99123） | **not measured** |
| 门 S | **not measured** | **not measured** |
| CUDA graph 捕获 | **not measured**（r17 的块级捕获已验证，head 里的路径没测） | not measured |

结论：`mx8_bwd` 前向与 bf16 完全一致、只量化反向的 ConvT dgrad/wgrad，ViT-L 3 批的精度很好，但还差 Amendment 2 §10.2 要求的 ≥ 8 批（中位数 ≥ 0.995、最小 ≥ 0.98）、ViT-B 与门 S，所以只作为后续选项；前向也 MXFP8（`mx8`）会让 mask logits 有 ~5% 的相对误差，且在 3 批里有 1 批过不了门 B，不建议。eager 下整步只多 2.7–4.0%，因为 host enqueue（7.3–7.8 ms/次）高于 GPU 时间；graph 之后 GPU 时间节省（ViT-L 每次 predict −2.4 ms，约 −9.6 ms/步）才可能兑现，这是 *预期*，没有测过。

### 4.6 CUDA graph 捕获与 host 开销（`tests/test_graph.py`、`tests/host_cost.py`）

| 模型 | graph 回放 vs eager（新输入，同参数） | 回放 ms/次 predict | eager ms（4.2） |
|---|---|---:|---:|
| ViT-L | mask cos 1.00000000，class cos 1.00000000，全部梯度 cos ≥ 0.99999988，max abs(mask 差) = 0 | 6.990 | 7.214 |
| ViT-B | mask cos 0.99999988，class cos 0.99999982，全部梯度 cos ≥ 1.0000 | 8.755 | 8.993 |

捕获的是完整 fwd+bwd（含 autograd.Function 的 backward，Triton/CUDA kernel、QuACK GEMM、cuBLAS 都在当前 stream 上，没有 host 同步，工作区走缓存分配器；JIT/autotune 在 3 次预热里完成）。host enqueue（eager，GPU 同步后单次，中位数）：ViT-L torch head 5.93 ms → r14 5.46 ms；ViT-B 6.62 → 4.28 ms。

### 4.7 显存
ViT-L 峰值 17.48 → 15.76 GB（P2：17.23 → 15.51），ViT-B 15.31 → 12.38 GB（P2：14.80 → 11.87）：r14 只保存各一份 bf16 的 Z1/Z2/Y1/Y2（解释，未单独量化：基线 autograd 要为 ConvT 输出、GELU 输入、dwconv 输出、LN 输入等保存多份 NCHW/NHWC 拷贝）。

## 5. 正确性

### 5.1 门 A：对 fp32 源码实现（`tests/test_gateA.py`，`profile/gateA_head_final.json`）
参考：源码 `_predict`（`backbone.norm` + class_head + mask_head + upscale + einsum），fp32、关闭 TF32；输入 `x [B,1129,C]`，gauss（σ=2）和尖峰数据（×8、4 个通道 +300、register token ×6）；随机化 LN 的 γ/β、dwconv 权重和 ConvT bias，使每条梯度路径都非平凡；对比 mask logits、class logits、对 x 的梯度和**全部参数梯度**。门槛：mask/class cos > 0.9999、参数梯度 cos_min > 0.999、梯度 rel_L2 不超过 TB1 torch bf16 head 的 2 倍。（注意力的 N=1029/1129 尾块、mask 边界与 head 无关；head 的输入固定是 1129 个 token。）

| 用例 | 候选 | mask logits cos / rel_L2 | class cos | 参数梯度 cos 最小 / rel_L2 最大 |
|---|---|---|---:|---|
| ViT-L gauss | **r14** | 0.9999791 / 6.46e-3 | 0.9999962 | 0.999974 / 7.2e-3 |
| | TB1 torch bf16 head | 0.9999633 / 8.56e-3 | 0.9999962 | 0.999960 / 8.9e-3 |
| ViT-L spiky ×8 | **r14** | 0.9999804 / 6.24e-3 | 0.9999965 | 0.999974 / 7.2e-3 |
| | TB1 torch | 0.9999631 / 8.59e-3 | 0.9999965 | 0.999960 / 9.0e-3 |
| ViT-B gauss | **r14** | 0.9999769 / 6.81e-3 | 0.9999956 | 0.999969 / 7.9e-3 |
| | TB1 torch | 0.9999609 / 8.84e-3 | 0.9999956 | 0.999958 / 9.2e-3 |
| ViT-B spiky ×8 | **r14** | 0.9999694 / 7.95e-3 | 0.9999949 | 0.999957 / 9.3e-3 |
| | TB1 torch | 0.9999471 / 1.04e-2 | 0.9999949 | 0.999907 / 1.4e-2 |

四个用例全部 PASS（`GATE A overall: PASS`）。输出契约：fp32 连续 `[B,100,128,128]` mask logits 与 fp32 `[B,100,3]` class logits（dtype 已检查）。stage kernel 单独对 fp32 参考（`test_triton_stage.py`，Triton 与 CUDA 两版）：y rel_L2 1.66e-3（bf16 输出舍入）、dZ 2.35e-3、dbias 1.6e-3、dW 1.6e-3、dγ/dβ ≤ 8e-7，cos ≥ 0.9999972；覆盖 (C,B,h) = (1024,7,64)、(768,14,32)、(1024,7,32 尖峰)，以及奇数批/小网格 (1024,1,32)、(768,3,8)、(1024,2,16 尖峰)、(768,5,32 尖峰)、(1024,7,64 尖峰)、(768,14,64)（`logs/m_robust.log`），所有用例 cos ≥ 0.9999972。

### 5.2 门 B：同精度换 kernel（`tests/gate_b_multi.py`；R0（fp32）、参考 TB1_R13、候选 TB1_R13_R14，同权重/同 batch/同 RNG tape；准则 = Amendment 1 §9.4 + Amendment 2 §10.2：loss_rel ≤ 1e-2、全局 grad cos ≥ 0.99、grad_norm_rel ≤ 0.02、update cos ≥ 0.99；逐参数 `cos(cand,R0) ≥ min(0.99, cos(ref,R0)−0.02) OR cos(cand,ref) ≥ 0.99`）

批次说明：每个进程的真实 batch 不同（r17 NOTES 已指出），所以"seed k"在不同进程里不是同一批；下表把所有进程的批次合并计数（ViT-L 3 + 4 = 7 批，ViT-B 3 + 6 = 9 批）。

| 模型 | 通过 | 对 ref 的全局 grad cos（最小 / 中位） | loss_rel 最大 | grad_norm_rel 最大 | update cos 最小 | 逐参数 offenders |
|---|---|---|---:|---:|---:|---|
| ViT-L | **7/7** | 0.99913 / 0.99999 | 1.0e-4 | 0.0031 | 1.0000 | 0 |
| ViT-B | **8/9** | 0.99849 / 0.99969 | 1.1e-4 | 0.0124 | 1.0000 | 8 批为 0，**1 批为 6 个** |

**唯一一次失败**（ViT-B，第一个进程的第 3 批；`profile/gateB_vitb16_r14.json`，seeds["2"]）：全局指标全过（loss_rel 3.3e-5、grad cos 0.99956、grad_norm_rel 0.0017、update cos 1.0000），6 个逐参数 offenders 全部在最前面几层：`backbone.cls_token`（cos(cand,ref) 0.981）、`blocks.0.norm1.weight/bias`（0.960 / 0.969）、`blocks.0.attn.proj.weight/bias`（0.952 / 0.950）、`blocks.4.norm2.weight`（0.987）；这一批里 TB1_R13 自己对 R0 的 `blocks.0.norm1.weight` cos 只有 0.846，对 R0 的全局 cos 只有 0.996，cand 对 R0 为 0.769。分组余弦（对 ref）：head 相关组 `class_head` 0.99999、`mask_head` 0.99946、`upscale.0/1` 0.99964 / 0.99963；backbone 0.9913（block00 0.988、block01 0.987、block02–11 ≥ 0.9963）。即 head 自己的梯度是对的，是 head 反传给骨干的 dx 的 bf16 舍入差经 12 层放大后落在 layer-0 的敏感参数上——round11 §4.5 记录过两个 bf16 栈互相比（换 SDPA backend）逐参数 cos 可低到 0.77，是同一类现象。**按 Amendment 2 的严格字面准则这一批是失败，我不把它改判为通过**；是否需要放宽（例如逐参数准则对 ref 自身对 R0 已 < 0.9 的参数只看全局指标）由协调者决定。

对照实验（`CTRL_noLN`：去掉 `head_ln_bf16`；`CTRL_wgq`：r13 的 wgrad 全走 QuACK；两者都不改 head 的算法，ViT-L 4 批、ViT-B 6 批）：对 TB1_R13 的 loss_rel 为 0、全局 grad cos 1.00000、逐参数 cos ≥ 0.999，全部通过。它们**不是有用的 bf16 噪声底线**：`CTRL_noLN` 与基线逐位相同（LayerNorm 的 γ=1、β=0 在 bf16 里精确可表示，`head_ln_bf16` 在初始化权重下与 autocast 路径数值相同），`CTRL_wgq` 只改 wgrad 的 GEMM 后端。真正的噪声底线对照（如换 SDPA backend）见 round11 §4.5。

### 5.3 门 S：50 步、2 个真实 batch（`tests/gate_s_multi.py`，`gate_c(interval=10, max_rel=0.05)`；R0 / TB1_R13 / TB1_R13_R14 同批；相：真实 mask 调度 `schedule` 与固定 `mid`；late 没跑）

| 模型 | 比较 | schedule max EMA 相对偏差 | mid | 全部有限 | grad-norm 比例范围 |
|---|---|---:|---:|---|---|
| ViT-L | TB1_R13_R14 vs R0 | 2.6e-3 | 4e-4 | 是 | [0.984, 1.002] |
| | TB1_R13_R14 vs TB1_R13 | 4e-4 | 2e-4 | 是 | [0.999, 1.000] |
| | （参考：TB1_R13 vs R0） | 2.8e-3 | 6e-4 | 是 | [0.985, 1.002] |
| ViT-B | TB1_R13_R14 vs R0 | 2e-4 | 1e-4 | 是 | [0.997, 1.002] |
| | TB1_R13_R14 vs TB1_R13 | 2e-4 | 2e-4 | 是 | [0.998, 1.008] |
| | （参考：TB1_R13 vs R0） | 2e-4 | 2e-4 | 是 | [0.991, 1.002] |

全部 PASS（门槛 5%）。注意：门 B / S 是在 `TB1_R13` 之上做的；`P2_R14`（再叠 r17 的 MXFP8 块）没有单独跑门 B/S（r14 只替换 `ops.predict`，与 `ops.block` 无交互；P2_R14 的 50×3 整步 bench 正常完成）。

### 5.4 其他测试
`tests/test_graph.py`（4.6）、`tests/smoke_nvtx.py`（NVTX 开/关输出逐位相同）、`tests/test_triton_stage.py`（5.1）。没有 CPU 单元测试（kernel 只有 CUDA 路径；不适用形状回退到前一个 `ops.predict`，`head.eligible`，未单独测试回退分支）。

## 6. 静态分析与 roofline

`nvcc --resource-usage`（sm_120，`-O3 --use_fast_math`，对 `head_kernels.cu` 的所有模板实例）：

| kernel（C=1024 / 768） | 寄存器 | spill | 说明 |
|---|---:|---|---|
| `fwd_kernel<C, CPT=8>`（采用） | 157 / 157 | 0 | CTA = 128 / 96 线程；smem 64 / 48 B（块规约暂存） |
| `bwd1_kernel<C, CPT=4, MINB=2>`（采用） | 127 / 128 | 0 | CTA = 256 / 192 线程，≥ 2 CTA/SM |
| `bwd2_kernel<C, CPT=4, MINB=2>`（采用） | 128 / 146 | 44 B / 0 | dW 的 9 tap × CPT 累加器常驻寄存器；MINB=1 时 168 个寄存器（1 CTA/SM，慢 8%）；CPT=8 为 255 个寄存器并 spill（12–20 B），无收益 |

没有 ncu（`ERR_NVGPUCTRPERM`），只能用 ptxas 加带宽 / FLOPs 的 roofline 估算：

- **带宽**：ViT-L stage2 特征图一遍 235 MB。融合 kernel 的最少遍历数：前向 2、bwd1 3、bwd2 3，共 8 遍；einsum 另读 Y（前向、dE）并写 dY。取 1.79 TB/s（假设值）：stage2 三个 kernel 下限 1.05 ms，实测 1.446 ms（≈ 73%；各 kernel 分别 fwd 0.355 / 0.263 = 74%，bwd1 0.505 / 0.394 = 78%，bwd2 0.547 / 0.394 = 72%）。stage1：fwd+bwd 0.417 ms 对 0.264 ms（63%，小 kernel 的启动/尾部开销占比更大）。整个 predict 的融合 kernel 合计 1.66 ms。
- **GEMM**：bf16 限流上限约 230 TFLOPS。ViT-L 一次 predict 的 ConvT 3 个 GEMM = 3 × (60.1 + 240.5) GF = 902 GF → 3.9 ms @230T；实测 QuACK/cuBLAS 的 ConvT 合计 4.1 ms（≈ 220 TFLOPS），einsum 3 个 GEMM 合计约 0.55 ms（共 70.5 GF，≈ 130 TFLOPS，受 Y 的读带宽限制：235 MB / 0.18 ms ≈ 1.3 TB/s）。整个 predict 的 GEMM 类 kernel 4.70 ms，已在上限。MXFP8（块缩放，~630–700 TFLOPS 量级）下 ConvT 的下限 ≈ 1.5 ms；实测 4.5 的 QuACK 块缩放 GEMM 合计 1.87 ms + 量化 0.41 ms。



## 7. 结论、失败项与下一个瓶颈

**结论**
- **做成了**：融合 head（`round14_head_sm120:fused`）。单次 predict ViT-L 11.34 → 7.21 ms，ViT-B 15.66 → 8.99 ms；整步（三相加权）P1R13 上 +10.8% / +24.0%，P2 上 +13.7% / +26.8%（P2+r14：59.07 / 139.95 img/s，对 R0 约 3.65x / 3.76x）；峰值显存 −1.7 / −2.9 GB。ViT-L 步时省 ≥ 15.8 ms（合入门槛 ≥ 4 ms）。门 A 全过、门 S 全过、门 B ViT-L 7/7、ViT-B 8/9（唯一一次失败见 5.2）。可被 CUDA graph 捕获（回放与 eager 一致）。
- **起作用的设计**：(1) 不物化 pixel shuffle：ConvT 当成 `Linear(C→4C)`，融合 kernel 在读取端按 `(P>>1, Q>>1, 相位)` 取址，省掉基线里 NCHW/NHWC 的 permute 拷贝（基线仅 LN2D 反向前的一次拷贝就 1.1 ms）；(2) 前向 1 个、反向 2 个融合 kernel，把 bias、GELU、dwconv3×3、LN2D（以及反向的 GELU'、dW 的 9 个 tap、dbias、dγ/dβ）全部做完，每个读入元素只算一次 GELU，3×3 卷积按列累加；(3) 反向的 einsum 两个小-M GEMM 走 QuACK（cuBLAS 在 K=100 / M=100 的长 K 上只有 50–60 TFLOPS，QuACK 2–2.7x）；(4) 全部中间量只保存一份 bf16，整个 upscale+einsum 是一个 `autograd.Function`。
- **没起作用/被替换的**：Triton 版（v1、v2）比 torch 快 2.6–3.5x（fwd+bwd）但比 CUDA 版慢 1.14–1.32x（fwd+bwd，4.1 (c)），保留为 `R14_KERNELS=triton` 备份；bwd2 的每线程 8 通道会 spill、每 CTA 多行（RPB=4）使 CTA 数不足而变慢（4.1 (c) 末尾）；ConvT 的 3 个 GEMM 在 bf16 下没有可抠的空间（QuACK 225–232 TFLOPS 对 cuBLAS 190–215，对 cuDNN 163–185；合计只省 ~0.3 ms）。
- **失败/保留项**：(1) 门 B 的 ViT-B 1/9 批在 Amendment 2 的逐参数准则上失败（5.2），已如实记录，需要协调者裁定；(2) 5.2 的两个对照实验不是有用的噪声底线；(3) MXFP8 ConvT 只是可选项，门 B 只测了 ViT-L 3 批、ViT-B / 门 S / ≥8 批规则 / graph 捕获均未做（4.5）；(4) `P2_R14` 没有单独跑门 B/S（r14 与 `ops.block` 无交互）；(5) 门 S 没跑 late 相；(6) P2 无 r14 的整步 nsys 没重采（引用 round17 的数字）；(7) 没有 CPU 单元测试。
- **host 侧**：r14 自己的 host enqueue 比基线更省（ViT-L 5.93 → 5.46 ms，ViT-B 6.62 → 4.28 ms / predict），但整步在 eager 下已经 host 受限（P2_R14 的 GPU 忙碌率 0.746）。

**下一个瓶颈**（P2+r14，ViT-L，mid，nsys kernel 104.1 ms，墙钟 138.7 ms）
1. **host launch / GPU 空泡**（最大）：>20 µs 间隙 410 个共 22.7 ms/步，忙碌率 0.746。只有 round16 的整步 CUDA graph 能兑现 104 ms 的 kernel 时间（r14 已验证可捕获）。
2. 骨干 GEMM（QuACK，含 MXFP8）42.6 ms（41%）、attention 16.4 ms（16%）、优化器 10.7 ms（r15）、MXFP8 量化 5.1 ms、LN/行 kernel 5.9 ms。
3. head 仍是 4 × 6.9 = 27.8 ms（ViT-L，21%）/ 34.6 ms（ViT-B，34%）：其中 68% 是 bf16 GEMM（ConvT 3 个 + einsum + mask_head），已在限流上限；融合 kernel 1.7 ms/predict（stage2 约 DRAM 带宽的 72–78%）；剩下 ~0.47 ms 是 16 个小的拷贝/reduce kernel。进一步的空间是 MXFP8 ConvT（4.5：预期 graph 后 −2.4 ms/predict，需要补全门）和把 `mask_head`/`class_head`/`backbone.norm` 的小 kernel 合并（上限约 0.4 ms/predict，估计值）。
4. 融合 kernel 距离 DRAM 下限还有约 25–30%（4.1 (c)），反向两个 kernel 的寄存器压力（dW 的 9 tap 累加器）是限制；可能的改法是把 dW 的 9 个 tap 拆到共享内存或分成两个 pass，预期收益 < 0.4 ms/predict（由与下限的差额估计，未做）。

**发布的栈**（需要协调者写入 `common/config.py`；本轮没有改 `common/`）
```python
P1R13R14_STACK = P1R13_STACK + ("round14_head_sm120",)
P1R13R14_PATCH_KWARGS = {**P1R13_PATCH_KWARGS, "round14_head_sm120": {"variant": "fused"}}
P2R14_STACK = P2_STACK + ("round14_head_sm120",)           # 推荐的最佳栈
P2R14_PATCH_KWARGS = {**P2_PATCH_KWARGS, "round14_head_sm120": {"variant": "fused"}}
```
`run.py` 的等价写法（`--stack`）：`TB1_R13_R14` = `r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16,r13:full,r14:fused`；`P2_R14` = `…,r13:full,r17:fc+proj+qkv_dw,r14:fused`。补丁依赖 round11 autocast（`REQUIRES`），可放在 r13/r17 之后（本轮实测顺序）；`head_ln_bf16` 在 r14 下是多余的（r14 不走 `LayerNorm2D.forward`），但无害。可选的 MXFP8 head：`r14:mx8_bwd`（未完成门，不发布）。

**契约变更请求**
1. `common/config.py` 增加上面的 `P1R13R14_*` 与 `P2R14_*`。
2. 门 B 的逐参数准则（Amendment 2 §10.2）对 "ref 自己对 R0 的逐参数 cos 已 < 0.9" 的参数（layer-0/1 的 LN、proj、cls_token）在 ViT-B 上会随批次失败（9 批里 1 批，全局指标全过，head 梯度正确）：建议这类参数只看全局指标，或把 `cos(cand,ref) ≥ 0.99` 条款改成相对 ref 自身噪声的阈值。
3. `check.step_equivalence` / `gate_b_multi` 的 "seed" 在不同进程里对应不同的真实 batch（沿用 round17 的请求 4），汇报批次数时应按 (进程, seed) 计。
4. `common/bench.assert_gpu_exclusive` 对别的代理留下的空闲 CUDA 上下文会中止（沿用 round17 的请求 3）：本轮 nsys 靠重试包装才拿到。
