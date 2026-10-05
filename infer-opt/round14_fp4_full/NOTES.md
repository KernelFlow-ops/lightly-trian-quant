# Round 14：整模型 NVFP4 / MXFP4（`round14_fp4_full`）

> 日期 2026-10-05。用户决定：整模型 NVFP4 / MXFP4，精度门关闭（checkpoint 是 bf16 训的，不指望低精度过门）。
> 只测速度；sanity 只报告不判定。所有数字都是本机实测，GPU 任务持双锁（`speed12.py` 自带 `gpu_lock2`；nsys 用 `prof14.sh`，同样双锁）。
> R0–R13 没有改动，只读 import；本轮文件全在 `round14_fp4_full/`，构建 / 日志在 `/root/autodl-tmp/infer-opt-tmp/round14_fp4_full/`。
> 另一个 agent（round13，MXFP8）同时用同一块 GPU；两边的计时都持锁，本轮每次 speed12 日志里 `other gpu procs at start/end` 记录了当时的外来进程。

## 0. 续做指南（被打断后看这里）

- 测速命令（ViT-L；ViT-B 把 `r12+head16` 换成 `r12`）：
  `cd /root/lightly-train/infer-opt/round12_lowp_stack && export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH && python speed12.py --model vitl16 --variants r5_v1,r12+head16,r14_nv4a,r14_mx4a --plugin /root/lightly-train/infer-opt/round14_fp4_full:plug14 --tag <t> --out /root/lightly-train/infer-opt/round14_fp4_full/results_speed.json`
  （`--out` 必须指向本轮目录，否则 speed12 会写 round12 的 results_speed.json。）
- 长任务：`setsid nohup ... > 日志 2>&1 < /dev/null &`，再 `tail` 日志。
- 变体名：`r14_nv4a` / `r14_mx4a` = step 1（QuACK + fp4q2 现成件）；`r14_nv4` / `r14_mx4` = step 2（自写 mma.sync block-scaled kernel，进行中）。
- 状态见 §6 的 checklist。

## 1. Step 1：`r14_nv4a` / `r14_mx4a`（用现成件拼出第一个整模型数字）

r12 的四个骨干 GEMM（每层 qkv / proj / fc1 / fc2）和 head 的两个 ConvT GEMM 全部换成 FP4；其余（patch-embed GEMM、fp32 残差流、LN 统计量、RoPE fp32、
attention、head 的 dw_ln / mask head / einsum、post）保持 r12 / r11 不变。

| 位置 | 实现 |
|---|---|
| LN | `impl/ln14.cu`：`ln16.cu` 的拷贝，输出改 bf16（fp32 残差 -> LN -> bf16） |
| 激活量化 | train-opt r24 `fp4q2.quant_act(row only)`；nv4 的 per-tensor scale 是**静态**的（见下），mx4 无 per-tensor scale |
| GEMM | QuACK 0.6.5 SM120 block-scaled GEMM（`mma.sync kind::mxf4nvf4.block_scale`），epilogue 自定义（`impl/fp4util.py::epilogues`）：`D = acc*ts + bias`（qkv）、`D = R + gamma*(acc*ts + bias)`（proj / fc2，fp32 输出）、`D = GELU(acc*ts + bias)`（fc1，bf16 输出）、`D = acc*ts`（ConvT）。`ts = ts_act*ts_w` 是设备标量，mx4 传 1 |
| RoPE + 布局 | `impl/rope14.cu`：QuACK 只做 bias，RoPE 与 `[3,B,N,H,64]` fp16 布局另走一遍（`epi_rope` 的改编，无 bias，fp16 输出） |
| attention | round-11 `attn16` kernel（fp16 q/k/v，kind 4，layers 0-1 Q 预缩放 1/8，同 r12），bf16 输出 -> fp4q2；`impl/attn14.cu` = 该文件把 namespace `a16` 改名 `a14`（原因见 §5） |
| 权重 | 离线一次量化：1D block scale（沿 K），精确 per-tensor scale（`fp4_ref.quantize_matrix`） |
| 静态 ts（nv4 激活） | 第一次（eager）调用时逐 site 在设备上测 amax -> `ts = 2.0*amax/2688`（余量 2），写进固定 buffer，之后图回放不再算；整个过程无主机同步。第一次调用必须在 CUDA graph capture 之前（speed12 的 warm-up 满足） |

每层 12 个 kernel：LN | quant | qkv GEMM | RoPE | attention | quant | proj+res | LN | quant | fc1+GELU | quant | fc2+res。

### 1.1 op 级 sanity（`tests/t_ops_a.py`，不作门）

M = 2058 / 2258，ViT-L 的 qkv / proj / fc1 / fc2 形状，两种格式：
- 量化器（fp4q2 静态 ts 模式）的 q 码和 scale 字节对 `fp4_ref` **逐位一致**（8 个形状 x 格式组合全部 True）。
- GEMM 对"反量化操作数的 fp64 乘积"：bias 版 / fc1 GELU 版 / plain 版相对 L2 = 1.66e-3 ~ 1.70e-3（bf16 输出舍入），cos 0.9999986；fp32 输出的 residual 版 3e-8 ~ 5e-8。

### 1.2 整模型速度（`speed12.py`，CUDA graph 回放，3 轮交替取中位数；视频 FPS 含 cv2 解码）

**ViT-L**（日志 `logs/speed_L_step1.log`，外来 GPU 进程：开始 / 结束都为空）

| variant | model ms | x vs r5_v1 | frame ms | GPU 上限 FPS | 视频 FPS | finite | mask-logit cos vs r5_v1（只报告） |
|---|---:|---:|---:|---:|---:|---|---:|
| r5_v1 | 9.678 | 1.00 | 9.855 | 101.5 | 96.4 | True | 1.000000 |
| r12+head16 | 7.069 | 1.37 | 7.445 | 134.3 | 124.4 | True | 0.999996 |
| **r14_nv4a** | **4.001** | 2.42 | 4.069 | 245.8 | 213.7 | True | 0.998997 |
| r14_mx4a | 4.039 | 2.40 | 4.103 | 243.7 | 213.2 | True | 0.995952 |

**ViT-B**（日志 `logs/speed_B_step1.log`；视频 FPS 受 host / 解码限制，比较看 model_ms / frame_ms）

| variant | model ms | x vs r5_v1 | frame ms | GPU 上限 FPS | 视频 FPS | finite | mask-logit cos vs r5_v1（只报告） |
|---|---:|---:|---:|---:|---:|---|---:|
| r5_v1 | 3.673 | 1.00 | 3.835 | 260.7 | 225.4 | True | 1.000000 |
| r12 | 2.721 | 1.35 | 2.793 | 358.1 | 275.7 | True | 0.999992 |
| **r14_nv4a** | **1.773** | 2.07 | 1.843 | 542.6 | 287.5 | True | 0.997203 |
| r14_mx4a | 1.798 | 2.04 | 1.867 | 535.7 | 281.0 | True | 0.988281 |

（r12+head16 在 ViT-L 上本次测得 7.069 ms，r12 的 NOTES 里记的是 7.353 ms；同一脚本，差异来自别的进程 / 时钟状态，两次都是实测。）

## 5. 踩过的坑

- **同源 attention 扩展的符号冲突**：r12 的 `attn16o.cu` 和 round-11 的 `attn16.cu` 同源，都定义 `a16::launch<KIND>`（含函数内 `static bool attr` 用来只设一次
  `cudaFuncAttributeMaxDynamicSharedMemorySize`）。同一进程先加载 r12 的模块再加载另一个同源模块时，后者的 launch 复用了前者的 `attr`，没给自己的 kernel 设 smem 属性 ->
  `CUDA error: invalid argument at line 382`。单独跑没问题（`r14_nv4a` 单独 1.768 ms），和 r12 同进程就挂。解法：`impl/attn14.cu` 把 namespace 改名 `a14`。
- **QuACK autotune 留下的 last-error**：autotune 试无效 config 的 launch 错误被 Python 吞掉，但留在 runtime 的 last-error 槽里，之后别处的 `cudaGetLastError()` 会误报。
  现在第一次（校准）调用里每个 QuACK GEMM 之后调用一次 `clear_cuda_error()`（`ln14.cu`，只在 `calib.on` 时；图回放里没有任何额外开销）。

## 6. Checklist（续做用）

- [x] step 1 quantizer + GEMM op sanity
- [x] step 1 整模型 ViT-B / ViT-L：speed12 表（§1.2）
- [ ] step 1 nsys 逐 kernel 账（ViT-L nv4a）
- [ ] step 2：自写 block-scaled mma.sync GEMM（`r14_nv4` / `r14_mx4`）
- [ ] step 2：LN+quant 生产者、fc1 quantize-out epilogue、attention 输出量化
- [ ] 最终表（ViT-L / ViT-B，r5_v1 / r12 / r14 全部变体）


## 7. 状态更新（协调者追加，2026-10-05 10:45；R14 子代理被会话额度打断后）

### 7.1 整模型速度，本轮最终测（`speed12.py`，CUDA graph 回放，4 个变体同进程交替，3 轮取中位数）

日志 `logs/speed_L_final.log`、`logs/speed_B_final.log`；结果合并进 `results_speed.json`（键 `final_L`、`final_B`）；外来 GPU 进程：开始 / 结束都为空。精度门关闭，最后一列只报告。

**ViT-L**

| variant | model ms | x vs r5_v1 | frame ms | GPU 上限 FPS | 视频 FPS | mask-logit cos vs r5_v1 |
|---|---:|---:|---:|---:|---:|---:|
| r5_v1（baseline，bf16） | 9.765 | 1.00 | 9.930 | 100.7 | 94.6 | 1.000000 |
| r12+head16（FP16 栈） | 7.084 | 1.38 | 7.755 | 128.9 | 120.4 | 0.999996 |
| **r14_nv4a（整模型 NVFP4）** | **4.019** | **2.43** | 4.100 | 243.9 | 207.5 | 0.998997 |
| **r14_mx4a（整模型 MXFP4）** | **4.038** | **2.42** | 4.118 | 242.9 | 207.6 | 0.995952 |

**ViT-B**

| variant | model ms | x vs r5_v1 | frame ms | GPU 上限 FPS | 视频 FPS | mask-logit cos vs r5_v1 |
|---|---:|---:|---:|---:|---:|---:|
| r5_v1 | 3.619 | 1.00 | 3.780 | 264.5 | 215.4 | 1.000000 |
| r12（FP16 栈） | 2.701 | 1.34 | 2.836 | 352.6 | 241.7 | 0.999992 |
| **r14_nv4a** | **1.739** | **2.08** | 1.807 | 553.5 | 243.9 | 0.997203 |
| **r14_mx4a** | **1.751** | **2.07** | 1.823 | 548.4 | 239.8 | 0.988281 |

- ViT-B 的视频 FPS 受 host / cv2 解码限制；这次测的时候机器负载很高（load average 15–21），视频 FPS 比 §1.2 那次低（那次 225 / 276 / 288 / 281），比较请看 model_ms / frame_ms / GPU 上限 FPS。
- 和 §1.2 的 step 1 数字一致：ViT-L 4.00 对 4.02 ms，ViT-B 1.77 对 1.74 ms（噪声 ±3%）。

### 7.2 step 1 的 nsys 逐 kernel 账（ViT-L `r14_nv4a`，`profile/nsys_L_nv4a_cuda_gpu_kern_sum.csv`，5 帧平均，每帧 ms）

| 部分 | ms/帧 | 占比 | 备注 |
|---|---:|---:|---|
| QuACK GEMM（proj + fc2，fp32 残差 epilogue） | 0.922 | 24% | 48 个 / 帧，平均 19.2 µs |
| attention（`a14` kind 4，fp16 累加） | 0.848 | 22% | 24 个 / 帧，35.3 µs |
| QuACK GEMM fc1（bias + GELU） | 0.486 | 13% | 20.3 µs，约 854 TFLOPS |
| QuACK GEMM qkv（bias） | 0.349 | 9% | 14.5 µs，约 891 TFLOPS |
| 激活量化 `actf_kernel` | 0.323 | 8% | 98 个 / 帧 |
| `ln_bf16` | 0.272 | 7% | 48 个 / 帧 |
| `rope14`（RoPE + 布局） | 0.193 | 5% | |
| head `dw_ln2` | 0.185 | 5% | |
| head 的 bf16 GEMM（cuBLAS cutlass_80 64x256，推测是 mask einsum） | 0.104 | 3% | 1 个 / 帧，103 µs |
| head ConvT（QuACK plain） | 0.068 | 2% | |
| patch / cat / tail / 其他 | 0.07 | 2% | |
| **kernel 合计** | **3.82** | | model_ms 4.02，差 0.2 ms 是 kernel 间隙 |

- GEMM 合计 1.83 ms（48%），QuACK FP4 在这些小 M 形状上只有约 850–890 TFLOPS，指令峰值 1964。
- 量化 + LN + RoPE 这三类 12 个 kernel/层的小 kernel 合计 0.79 ms（21%），是融合的主要空间。

### 7.3 step 2（自写 `mma.sync kind::mxf4nvf4.block_scale` GEMM，`r14_nv4` / `r14_mx4`）现状：未完成，目前比 step 1 慢

来自 `results_speed.json` 的 `step2_tuned1_*`（子代理被打断前最后一次测）：

| 模型 | nv4a（QuACK） | nv4（自写） | mx4a（QuACK） | mx4（自写） |
|---|---:|---:|---:|---:|
| ViT-L model ms | 3.964 | 4.471 | 3.981 | 4.135 |
| ViT-B model ms | 1.759 | 1.977 | 1.792 | 1.873 |

- 自写 kernel 的 GEMM 合计 2.88 ms/帧（ViT-L nv4，`profile/nsys_L_nv4_*`），QuACK 是 1.83 ms；自写 kernel 的 tile 调参结果只有约 380–730 TFLOPS。
- 自写 kernel 配套的融合量化已经写好（`impl/lnq14.cu`：LN + 量化一个 kernel，5.3 µs，对 `ln_bf16` + `actf` 的 5.7 + 3.3 µs；attention 输出量化 `quantx_kernel` 3.0 µs），但没有单独接到 QuACK 的栈上。
- 子代理在 `impl/fp4g_core.cu`、`fp4g_core_v1.cu`、`fp4g_abl.cu`、`tuned14_v2.json` 上继续调主循环时被打断（最后修改 01:54），没有产出新结果。**整模型 NVFP4 / MXFP4 的发布版本是 step 1（`r14_nv4a` / `r14_mx4a`）。**

### 7.4 还能做的（没做，按预计收益排序）

1. 融合：把 `lnq14.cu` 的 LN + 量化接到 step 1 栈上（约 −0.18 ms/帧 ViT-L）、attention 输出量化并进 attention epilogue（约 −0.07）、GELU 量化并进 fc1 epilogue（约 −0.1）、RoPE 并进 attention 预处理（最多 −0.19，实际要扣掉 attention 里多出来的那部分）。合计约 −0.4 ~ −0.5 ms/帧（−10% ~ −12%）。另外 head 里 103 µs 的 bf16 GEMM 和 `dw_ln2`（0.185 ms）也还有空间。
2. 低精度 attention（R11 候选矩阵：Sage3 FP4 kernel 27.9 µs，Spark FP8 29.4 µs，对现在的 34.8 µs；两者都要另付量化，没有融合的话整体反而更慢）：融合量化之后最多约 −0.17 ms/帧（(34.8 − 27.9) µs × 24 层）。
3. 视频 FPS 已经被解码 / host 卡住：ViT-B 约 240–290 FPS，ViT-L 的 frame_ms 4.1 ms 对应 243 FPS、视频 207 FPS。再往下压 GPU 时间，视频 FPS 看不出来，要先换解码（NVDEC，需要 pip 安装，要先问用户）。
4. 多帧合批（M=4116）可以改善 GEMM 的 wave quantization，只对吞吐有利，延迟变大。
