# Round17：MXFP8（E4M3 + UE8M0，block-scaled mma）骨干 Linear（sm_120）（负责：Agent C2）

> 全部数字都在 `infer-5090`（新容器，RTX 5090，575 W）上实测，缺项写 pending/null。参考栈 `TB1_R13` = `TB1_STACK + ",r13:full"`；候选 `MX8_<variant>` = `TB1_R13` + `r17:<variant>`（`run.py` 的简写）。
> 同一台机器上还有别的代理在用 GPU（持锁排队，但上下文会残留），所以绝对步时有 ±2–3% 的漂移（`TB1_R13` 在不同时段测得 mid 158.0 / 160.3 ms）；表里的倍数只在同一时段的同批测量之间比较。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 做了什么 | 自研 CUDA 双输出量化器（一次读取同时产出行副本和列副本，scale 直接写成 QuACK 的 128×4 swizzle）+ QuACK 0.6.5 SM120 block-scaled GEMM（`mma.sync kind::mxf8f6f4.block_scale`）+ round13 的融合 epilogue 原样复用 + 每步量化权重 + 分层策略（Linear / GEMM 种类 / 块） |
| GEMM（ViT-L plain 块，M=7203，预量化，原生布局） | 12 个 GEMM：bf16 2585 µs → MXFP8 1042 µs（2.48×，314–638 TFLOPS）；量化（8 个激活/梯度双输出 + 4 个权重）203 µs；合计 1245 µs（2.08×） |
| 单块 fwd+bwd（CUDA graph 回放，含注意力） | ViT-L plain：3.63 → 2.30 ms（全 4 个 Linear，1.58×）/ 2.63 ms（fc+proj，1.38×）/ 2.72 ms（fc，1.33×）；joint 4.21 → 2.79 ms；ViT-B plain 4.35 → 3.12 ms（1.40×） |
| 整步（三相加权，eager，同一时段） | ViT-L：43.72 → 50.06（fc，1.145×）/ 50.44（fc+proj，1.154×）/ **51.56（fc+proj+qkv_dw，1.179×）** / 52.09（all，1.191×）；ViT-B：100.67 → 110.46（1.097×）/ 108.91（1.082×）/ **110.03（1.093×）** / 112.29（all，1.115×）；见 §4.3。推荐策略相对 R0（16.2 / 37.2）：3.18× / 2.96×；相对 TB1（41.4 / 96.0）：1.25× / 1.15× |
| 推荐 | **推荐 `MX8_fc+proj+qkv_dw`（`r17:fc+proj+qkv_dw`）；稳妥备选 `MX8_fc`。** 两档：(a) 稳妥档 `r17:fc`（只有 fc1/fc2 走 MXFP8，全部块含 joint 块）：ViT-L 1.145× / ViT-B 1.097×（相对 TB1_R13），单步全局梯度余弦 40 批里 38 批 ≥0.99，最低 0.9897；(b) 性能档 `r17:fc+proj+qkv_dw`（再加 proj 的三个 GEMM，以及 qkv 的 dgrad/wgrad——qkv 的**前向保持 bf16**）：ViT-L 1.179× / ViT-B 1.093×，16 批里 13 批 ≥0.99，最低 0.9880（loss_rel ≤4e-3，Gate S 全过）。**不推荐** `all`（qkv 前向也走 MXFP8）：ViT-L 1.191× / ViT-B 1.115×，但 qkv 前向的量化噪声在部分批次把全局梯度余弦拉到 0.96–0.99（§5）。eager 下整步已被 host 限制，(b) 相对 (a) 的 GPU 收益要等 r16 的整步 graph 才兑现（块级 graph 回放 fc+proj 2.63 ms vs fc 2.72 ms，再加 qkv 的 dgrad/wgrad 预计再省 ~0.24 ms/块，未单独实测） |
| 正确性 | Gate A 全过（量化器 98 例逐位一致；GEMM+epilogue；块）；Gate B-lite（改精度准则，单步全局梯度余弦 ≥0.99，**按批统计**）：fc 38/40 批通过（最低 0.9897），fc+proj+qkv_dw 13/16（最低 0.9880），全部 Linear ViT-L 11/11、ViT-B 7/11（最低 0.9833）；加入 qkv 的**前向**会在部分批次把全局梯度余弦拉到 0.96–0.99；qkv 的 dgrad/wgrad 用 MXFP8 对余弦没有可见影响；loss_rel 全部 ≤5.3e-3（门槛 2e-2）；Gate S（50 步×2 批）所有策略全过（EMA 损失偏差 ≤4.1e-3，门槛 5e-2）。单步门槛对 MXFP8 是边缘的，见 §5 和 §7 的契约变更请求 |
| 瓶颈 | MXFP8 让整步从 GPU 受限变成**主机（launch）受限**：nsys 的 kernel 总时间 ViT-L 147.9 → 117.0 ms，但 GPU 忙碌率从 88.8% 掉到 80.5%，>20 µs 的间隙从 125 个/7.5 ms 增到 275 个/14.7 ms；单块 CUDA graph 回放已证明可捕获，剩下的收益要靠 r16 的整步 graph（见 §7） |
| 峰值显存 | ViT-L 17.48 → 17.24 GB（全部 Linear 与推荐策略 fc+proj+qkv_dw）/ 17.25（fc+proj）/ 17.07（fc）；ViT-B 15.31 → 14.80（全部 Linear 与推荐策略）/ 14.88 / 14.81 GB（激活由 bf16 变 fp8 只省下一小部分，峰值主要在别处） |
| 契约变更请求 | 见 §7 |

## 1. 目标与假设

骨干 Linear 的 12 个 GEMM（qkv/proj/fc1/fc2 × fwd/dgrad/wgrad）在 bf16 下受 FP32 累加 HMMA 限速（~230 TFLOPS），r13 之后 ViT-L 步里约 65 ms；block-scaled MXFP8 不限速（探针 314–650 TFLOPS）。净收益 = GEMM 节省 − 量化开销 − 可能的 host 开销。假设：(1) 量化器不是瓶颈（~0.2 ms/块）；(2) 精度满足 §7 的改精度准则（loss_rel ≤2e-2 且全局 grad 余弦 ≥0.99）；(3) 融合量化进生产者能再省一小部分。实测核对：(1) 成立（量化 5.1 ms/步，kernel 总时间的 4.2%）；(2) 部分成立（单步梯度余弦在批间波动，分层策略见 §5）；(3) **本轮未做**：生产者融合没有实现，因为 (4)。

(4)（新发现）整步在 eager 下已经受 host launch 限制（GPU 忙碌率 0.89 → 0.79–0.81），GPU 侧的进一步压缩（融合量化）在 eager 下看不到收益，只有 CUDA graph 之后才兑现。

## 2. 改动

| 文件 | 内容 |
|---|---|
| `impl/mx8_kernels.cu`、`impl/mx8.py` | `quant_dual`：bf16/fp32 `x[R,C]` → 行副本（`qr [R,C]` e4m3，scale 沿 C 分块，`sfr`）+ 列副本（`qc [Rpad,C]` e4m3，scale 沿 R 分块，数据仍是行主序，`sfc`），一次读取；scale 直接写成 (rm, rk, 32, 4, 4) 的 128×4 swizzle。`MX8` 容器 = CONTRACTS 的 MX8Tensor（`qr/sfr`、`qc/sfc`、`rows/cols/mpad`）+ 四种 QuACK 操作数视图 `a_row / b_row_t / b_col / a_col_t` |
| `impl/gemm_mx8.py` | 7 个 GEMM 包装：fwd（bias / γ+fp32 残差 / erf-GELU）、dgrad（plain / GELU' + db1）、wgrad（fp32 输出直接作 Parameter 梯度）；epilogue 全部是 round13 的 `@gemm_epilogue`，通过 `SFA/SFB/bs_format_*` 接收 block-scaled 操作数 |
| `impl/fused_block_mx8.py` | round13 融合块的 MXFP8 版：`Policy`、`Mx8Weights`（每步在 block 0 由 fp32 Parameter 量化出行/列两份，fwd 与 bwd 共用）、`_MXLinear`（qkv，可按 GEMM 种类 f/d/w 选择）、`_MXProjRes`、`_MXFC1`、`_MXFC2Res`；LN / 残差 / LayerScale / GELU / 注意力仍是 round13 的代码；bf16 影子只刷新仍留在 bf16 的 Linear |
| `impl/patch.py` | `ops.block` 覆盖，REQUIRES round13；变体 token（用 `+` 连接）：`qkv proj fc1 fc2 fc attn all`、`qkv_f / qkv_d / qkv_w / qkv_fd / qkv_dw / qkv_fw`（只让 qkv 的指定 GEMM 种类走 MXFP8：f=fwd，d=dgrad，w=wgrad）、`nofirst nolast nojoint blocks=a-b` |
| `run.py` | 改编自 round13/run.py：新增 `r17`、`MX8_<variant>` 预设 |
| `tests/` | `test_quant.py`（Gate A 量化器）、`test_gemm_mx8.py`（Gate A GEMM+epilogue）、`test_block_mx8.py`（Gate A 块）、`gate_b_multi.py` / `gate_s_multi.py`（多栈 B/S，同进程同批同 RngTape）、`gate_b_summary.py`、`bench_feasibility.py`、`bench_block_mx8.py`（graph 回放块计时）、`host_overhead.py` / `profile_host_block.py`（host 侧 launch 开销）、`nsys_queue.sh` / `nsys_step_categories.py`、`finalize_results.py`、`apply_r13_closure.py`（round13 收尾）等 |

**Scale 规则（逐位测试）**：NVIDIA MXFP8 训练配方——所有张量用 E4M3，scale = 2^ceil(log2(amax/448))（RCEIL，round-to-infinity），UE8M0 存储；amax 用 |x| 的位型做整数取最大值（NaN/Inf 传播成 scale 字节 255），元素乘 2^-e 后 RN-even 饱和转 e4m3。与 QuACK `to_mx(..., "rceil")` / `to_mx_dim0` 逐字节一致（`tests/test_quant.py`，98 个用例）。QuACK 的 `to_mx` 默认就是 rceil（对 torchao 的 floor 做了有意偏离）。

**量化方向**：`Y = X Wᵀ`：fwd 的 X、W 沿 K 量化；dgrad `dX = dY W`：dY 沿 N，W 沿 N（W 的“列”方向）；wgrad `dW = dYᵀ X`：dY、X 沿 token 轴（M）。所以每个激活/梯度张量需要行副本和列副本（`quant_dual` 一次读取同时产出），权重的两份每步量化一次。**数据布局一律保持行主序**：QuACK 的 SM120 block-scaled GEMM 用 `ldmatrix.m16n16.trans.b8` 读取 MN-major 的 fp8 操作数（dgrad 的 W、wgrad 的两个操作数），实测满速（§4.1，比物理转置的 K-major 副本慢 0–5%，所以不需要写转置副本）。

**M 填充**：wgrad 的收缩维是 token 数 M = B·N = 7203 / 7903 / 14406 / 15806，不是 32/128 的倍数。列副本按 Mpad = ceil(M/128)·128 分配（7296 / 7936 / 14464 / 15872，多 0.4–1.3% 的 FLOP），pad 行的数据和 scale 都由量化器写 0（pad 行输入视为 0，整块全 0 时 scale 字节为 0），所以对 GEMM 的贡献恰好为 0；行副本以及 fwd/dgrad 仍用真实的 M（TMA 越界补零，epilogue 谓词）。不需要把激活缓冲整体加长；`buf[:M].view(B,N,C)` 之类的布局问题不存在。

**参考源码**：QuACK 0.6.5（`gemm_sm120.py` 的 `MmaMXF8Op`、`epilogue/quantize_out.py` 的 SFD 量化 epilogue）。Spark `fp8gemm`（MIT）的 MX 模式只支持 K-major 双操作数、bf16 输出、`sfa[M][K/32]` 的平铺 scale 和 K%256==0，没有 epilogue，也不接受 MN-major 操作数；用于 dgrad/wgrad 需要物理转置副本且 wgrad 的收缩维 7296 不满足 K%256，所以**没有改编、没有实测**（results 里为 null）。Triton 量化器（r9）同理未测，QuACK `to_mx_compiled` 作为量化器候选实测（§4.2）。

## 3. nsys 分析过程

**命令**（全部持双锁，`profile_nsys.sh` 在 GPU 上有他人上下文时会拒绝，用 `tests/nsys_queue.sh` 重试；原始文件在 `/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/nsys/`，项目里只放 csv/summary）：

```bash
tests/nsys_queue.sh v2 vitl16:TB1_R13 vitl16:MX8_all vitl16:TB1 vitb16:MX8_all vitl16:MX8_fc+proj+qkv_dw vitb16:MX8_fc+proj+qkv_dw vitb16:TB1_R13   # 第二次调用补采
#  = common/profile_nsys.sh $OUT/step_<model>_<stack>_v2 run.py --model <model> --stack <stack> --action profile --phases mid --warmup 20
#    python common/nsys_analyze.py $OUT/step_... --iters 5 --top 25 ;  python tests/nsys_step_categories.py $OUT/step_...   (按 kernel 名分类)
```
NVTX 表（`:blk_NN`、`:fwd`、`:bwd`、`:predict_*`、`:clip_opt_sched`、`:loss`）取自 `*_nvtx_gpu_proj_sum.csv`。

**3.1 整步类别表（ViT-L mid，ms/步；kernel 时间 = 5 步平均）**

| 类别 | TB1 | TB1_R13（前） | MX8_all（后） | MX8_fc+proj+qkv_dw（推荐） |
|---|---|---|---|---|
| GEMM bf16（QuACK） | – | 49.2 | 0（见下） | 5.1（qkv fwd，bf16） |
| GEMM MXFP8（QuACK block-scaled，288 个 kernel） | – | – | 27.6 | 25.4 |
| GEMM bf16（cuBLAS/cutlass sm80：wgrad 与 head） | 72.8 | 20.8 | 6.2 | 6.2 |
| **MXFP8 量化（自研 `quant_dual`，288 个 kernel）** | – | – | **5.15** | **5.06** |
| head 卷积 | 22.2 | 22.2 | 22.1 | 22.3 |
| elementwise / copy / reduce / 其他 | 30.3 | 19.2 | 19.6 | 19.6 |
| attention（flash + RoPE） | 20.6 | 16.2 | 16.5 | 16.4 |
| 优化器 + grad clip | 10.4 | 11.6 | 10.4 | 10.7 |
| LN / 行 kernel（r13 CUDA） | 3.7 | 8.5 | 9.2 | 9.2 |
| loss / matcher | 0.2 | 0.2 | 0.2 | 0.2 |
| **kernel 总时间** | 160.1 | 147.9 | **117.0** | **120.4** |
| kernel 数 / 步 | 3607 | 3006 | 3295 | 3292 |
| GPU 忙碌率 | 0.902 | 0.888 | **0.805** | 0.794 |
| >20 µs 间隙 | 95 个 / 5.5 ms | 126 个 / 7.5 ms | **275 个 / 14.7 ms** | 329 个 / 17.6 ms |
| 同步 / 步 | 8 | 8 | 8 | 8 |

（MX8_all 一栏里 “GEMM bf16 QuACK” 的 27.6 ms 其实是 MXFP8 的 288 个 block-scaled GEMM；类别表按 kernel 名分类，QuACK 的 block-scaled kernel 名与 bf16 的相同，所以我把它们在这里手工分开。ViT-B MX8_all：kernel 112.6 ms，忙碌率 0.847，间隙 187 个/10.4 ms，量化 4.18 ms/步（144 个 kernel）。）

**3.2 NVTX（ViT-L，每块 GPU 投影时间 vs CPU 区间，ms，mid）**

| 范围 | TB1_R13 GPU / CPU | MX8_all GPU / CPU |
|---|---|---|
| blk_NN（plain 块，fwd） | 1.09–1.16 / 0.65–0.78 | 0.94–1.02 / 0.81–0.88 |
| blk_00（含每步的权重量化 + 影子刷新） | 2.27 / 1.36 | 4.20 / 4.11 |
| blk_21..23（joint） | 1.13–1.27 / 0.79–1.04 | 0.73–0.86 / 1.05–1.14 |
| fwd 总计 | 44.1 / 32.9 | 46.6 / 39.1 |
| clip_opt_sched | 17.5 / 54.5 | 16.3 / 29.4 |

TB1_R13 的块是 GPU 受限（GPU 1.1 ms > CPU 0.7 ms）；MXFP8 之后块前向的 GPU 时间本该降到 ~0.76 ms（graph 回放），但前向的 CPU 区间 0.85 ms，所以 GPU 在块里空转（块区间的 GPU 投影时间含空转）。`tests/profile_host_block.py` 对 eager 块（fwd+bwd）按函数累计 host 时间（两次独立运行、各 20 次迭代）：MXFP8 全 Linear 的被包装函数合计 1.94 ms/块，round13 的 bf16 块 1.46 ms/块（MXFP8 多 33%：12 次量化 400 µs + block-scaled 派发）；同时 GPU 时间 3.63 → 2.30 ms，所以块级 host 与 GPU 已经基本相当（1.94 ms host 对 2.30 ms GPU，不含 autograd 引擎与 attention 的 host 开销），放进整步（RoPE、head、优化器自己的 host 开销）就变成 host 受限。（整块 enqueue 总时间的两次运行相差很大：MXFP8 3.24 / 2.27 ms，bf16 1.93 / 3.17 ms，因为 launch 队列满时 enqueue 会被 GPU 阻塞，这个总数不可靠，只有按函数的累计值可用。）各函数的 host 开销（µs/调用）：QuACK epilogue GEMM 105–165、stock GEMM 95–100、量化 33、LN/rowbwd 31–52；round13 的 bf16 块里同样的 GEMM 包装是 75–200 µs，QuACK 的 Python 派发本身就贵（`tensor_key`、`_gemm_execute`、容器构造），MXFP8 只是把 GPU 时间压到了它之下。

**3.3 结论**
1. MXFP8 GEMM 总计 27.6 ms（288 个 kernel，平均 96 µs），对 bf16 的 49.2 + 约 15（wgrad 里的 cuBLAS）= ~64 ms，省 ~36 ms；量化 5.2 ms/步（288 个 kernel，4.4%）；净 kernel 时间 147.9 → 117.0 ms（−30.9 ms）。
2. 量化器时间：192 个激活/梯度双输出 kernel 共 4.17 ms（平均 21.7 µs，h / dz 两个 29.5M 元素的张量各约 64 µs，已在 DRAM 带宽上：59 MB 读 + 59 MB 写 ≈ 1.85 TB/s），96 个权重 kernel 约 1 ms。
3. 但整步 wall 只从 160.3 ms 降到 134.5 ms（bench，−25.8 ms；nsys 5 步平均 wall 164 → 142 ms），kernel 总时间与 wall 的差（间隙）从 12.4 ms 涨到 17.5 ms（nsys：busy 148 → 117 ms，wall 164 → 142 ms，空闲 16 → 25 ms）：**下一个瓶颈不是 GPU 而是 launch**。
4. 推荐策略 `fc+proj+qkv_dw`（ViT-L）：kernel 120.4 ms = MXFP8 GEMM 25.4 + 仍为 bf16 的 qkv 前向 5.1 + 量化 5.06 + 其余；wall 147.4 ms（nsys）/ 135.8 ms（bench）。

## 4. 实测结果

### 4.1 GEMM 候选矩阵（ViT-L，M=7203，12 个 GEMM；预量化操作数；CUDA graph ×5 回放，3 轮取中位数）

| GEMM | bf16 最优（QuACK/cuBLAS）µs | MXFP8 K-major 副本（物理转置，转置不计时）µs / TFLOPS | MXFP8 原生布局（MN-major，`ldmatrix.trans.b8`）µs / TFLOPS |
|---|---|---|---|
| qkv fwd | 195.5 | 84.0 / 539 | **74.4 / 609** |
| qkv dgrad | 217.5 | **77.6 / 584** | 82.0 / 553 |
| qkv wgrad（K=Mpad=7296） | 215.0 | **107.6 / 421** | 113.1 / 401 |
| proj fwd | 73.9 | 27.3 / 553 | **26.8 / 565** |
| proj dgrad | 75.1 | **26.7 / 566** | 30.1 / 501 |
| proj wgrad | 86.6 | **46.8 / 323** | 48.1 / 314 |
| fc1 fwd | 266.8 | 114.9 / 526 | **109.6 / 552** |
| fc1 dgrad | 287.7 | **103.2 / 585** | 108.7 / 556 |
| fc1 wgrad | 307.1 | **109.6 / 551** | 118.1 / 511 |
| fc2 fwd | 289.5 | 94.5 / 640 | 94.7 / 638 |
| fc2 dgrad | 266.1 | **98.7 / 612** | 119.3 / 506 |
| fc2 wgrad | 304.0 | **110.3 / 548** | 117.1 / 516 |
| **12 个之和** | **2584.6** | **1001.3** | **1042.1（2.48×）** |

bf16 一栏是各 GEMM 在 QuACK 与 cuBLAS（wgrad，fp32 输出）之间取的最快者。MXFP8 的相对 L2 误差对 fp32 乘积 3.75–3.77%（高斯数据，E4M3 三位尾数，两个操作数的量化误差叠加）。原生布局只比 K-major 副本慢 41 µs/块（4%），而 K-major 需要量化器额外写转置副本，所以选原生布局。wgrad 的 proj / qkv 受并行度/波量化限制（输出只有 64 / 192 个 128×128 tile，170 个 SM），314–421 TFLOPS，是 12 个 GEMM 里最低的；`R17_WGRAD_SPLITK=auto`（QuACK 串行 split-K，确定性）的开关保留，未做整步对照。

### 4.2 量化器候选矩阵（ViT-L 块的真实张量，M=7203；µs，CUDA graph ×5，bf16 输入；热循环，L2 常驻——超过 L2 的张量由 DRAM 带宽决定）

| 张量 [M×C] | 自研 dual（行+列） | 自研 row-only | 自研 col-only | QuACK `to_mx_compiled`（行，含 swizzle 打包） |
|---|---|---|---|---|
| ln1 / o / ln2 / g1 / g2 [7203×1024] | 7.5 | 5.0–5.3 | 5.5–5.8 | 7.9 |
| dqkv [7203×3072] | 19.6 | 11.1 | 13.3 | 15.8 |
| h / dz [7203×4096] | 63.8 / 65.0（输入 59 MB + 两份输出 59 MB 超过 96 MB L2，DRAM 受限，1.85 TB/s） | 14.5 / 14.4（L2 常驻） | 17.1 / 17.0（L2 常驻） | 20.2 |
| 权重 qkv / proj / fc1 / fc2（fp32 或 bf16 输入，dual） | 4.2 / 2.5 / 5.0 / 5.0 | – | – | – |
| **每块合计（8 个激活 dual + 4 个权重 dual）** | **203** | | | |

自研 kernel：每个 CTA 32 行×128 列，128 线程，每线程 4 行×8 列寄存器驻留，行 scale 用 4 线程 shuffle，列 amax 过一个 4 KB smem 交换，54–72 个寄存器、0 溢出（§6）。QuACK `to_mx_compiled` 只出行副本（列副本还要 `to_mx_dim0` + 另一次 swizzle 打包，未计），比自研 row-only 慢 1.4–1.5×，比自研 dual 在小张量上略慢、在大张量上因为不写列副本而快；因为反向需要两份副本，选自研 dual。r9 的 Triton 量化器和 Spark `fp8quant.cu` 未测（接口是 K-major 行量化且不出 128×4 swizzle / 列副本，改造量大于自研）。

### 4.3 整步（两个模型，三相加权；`common/bench.time_steps`，预热 20，50×3；early/mid/late 加权 0.2/0.6/0.2；同一时段测量）


**vitl16** (ViT-L B=7; phases early/mid/late, weights 0.2/0.6/0.2)

| stack | early ms | mid ms | late ms | weighted img/s | vs TB1_R13 | vs TB1 (41.4/96.0) | vs R0 (16.2/37.2) | fwd / bwd / opt / loss (mid, ms) | peak mem GB |
|---|---|---|---|---|---|---|---|---|---|
| TB1_R13 | 160.8 | 160.3 | 158.8 | 43.72 | 1.000x | 1.06x | 2.70x | 43.5 / 97.7 / 13.5 / 4.1 | 17.48 |
| MX8_fc | 139.9 | 140.3 | 138.2 | 50.06 | 1.145x | 1.21x | 3.09x | 38.6 / 82.5 / 13.6 / 4.2 | 17.07 |
| MX8_fc+proj | 139.5 | 139.0 | 137.5 | 50.44 | 1.154x | 1.22x | 3.11x | 39.1 / 80.8 / 13.4 / 4.2 | 17.25 |
| MX8_fc+proj+qkv_dw | 136.8 | 135.8 | 134.6 | 51.56 | 1.179x | 1.25x | 3.18x | 41.3 / 75.9 / 13.2 / 4.1 | 17.24 |
| MX8_all | 135.3 | 134.5 | 133.3 | 52.09 | 1.191x | 1.26x | 3.22x | 39.9 / 75.9 / 13.2 / 4.1 | 17.24 |

**vitb16** (ViT-B B=14; phases early/mid/late, weights 0.2/0.6/0.2)

| stack | early ms | mid ms | late ms | weighted img/s | vs TB1_R13 | vs TB1 (41.4/96.0) | vs R0 (16.2/37.2) | fwd / bwd / opt / loss (mid, ms) | peak mem GB |
|---|---|---|---|---|---|---|---|---|---|
| TB1_R13 | 139.4 | 139.9 | 136.2 | 100.67 | 1.000x | 1.05x | 2.71x | 37.4 / 86.8 / 7.2 / 7.3 | 15.31 |
| MX8_fc | 128.4 | 126.7 | 125.3 | 110.46 | 1.097x | 1.15x | 2.97x | 34.7 / 78.4 / 6.7 / 6.0 | 14.81 |
| MX8_fc+proj | 129.9 | 128.6 | 127.0 | 108.91 | 1.082x | 1.13x | 2.93x | 35.4 / 78.2 / 7.4 / 6.6 | 14.88 |
| MX8_fc+proj+qkv_dw | 128.3 | 127.6 | 125.1 | 110.03 | 1.093x | 1.15x | 2.96x | 36.3 / 75.9 / 7.5 / 6.8 | 14.80 |
| MX8_all | 126.0 | 124.8 | 123.1 | 112.29 | 1.115x | 1.17x | 3.02x | 34.2 / 75.6 / 7.3 / 6.6 | 14.80 |


### 4.4 单块 fwd/bwd（CUDA graph 回放，同一注意力；含每步的权重量化）

| 块 | r13_full ms | MXFP8 全 4 个 Linear | fc+proj | fc |
|---|---|---|---|---|
| ViT-L plain | 3.63（fwd 1.13 / bwd 2.50） | 2.30（0.76 / 1.54）**1.58×** | 2.63（0.88 / 1.75）1.38× | 2.72（0.90 / 1.82）1.33× |
| ViT-L joint | 4.21 | 2.79 **1.51×** | 3.15 1.34× | 3.24 1.30× |
| ViT-B plain | 4.35 | 3.12 **1.40×** | 3.48 1.25× | 3.50 1.24× |
| ViT-B joint | 5.26 | 3.93 **1.34×** | 4.31 1.22× | 4.35 1.21× |

所有变体在 `torch.cuda.graph` 里实际捕获并回放（plain/joint、两个模型），所以 MXFP8 块是 CUDA-graph 可捕获的（契约 §5）：量化 kernel 和 QuACK 调用都在当前流上，JIT/autotune 在预热里完成，每步新分配的 fp8/scale 张量在 graph 私有池里。

## 5. 正确性

### Gate A（算子对 fp32 / 对反量化 fp32）
- **量化器**（`tests/test_quant.py`，`profile/gateA_quant.json`）：对 QuACK `to_mx` / `to_mx_dim0`（rceil）逐字节比较行/列数据与 128×4 swizzle 的 scale，98 个用例：{bf16, fp32} × M∈{7203, 7903, 1029, 14406, 15806, 128, 33}（含 M 非 128 倍数的尾块）× {normal, spiky（尖峰通道 ×3e3、整行 ×80、单点 4e4）, scale8, tiny(1e-30), large(1e4), zeros, pow2（块最大值恰为 448·2^k 的 RCEIL 边界）}，**0 个不一致**；NaN/Inf 的 scale 字节（255）一致（数据字节的 NaN 编码不同，GEMM 里都是 NaN，无影响）。重写分配（每个方向一次分配）之后重跑，仍然 0 不一致。
- **GEMM + epilogue**（`tests/test_gemm_mx8.py`，`profile/gateA_gemm_quick.json`）：fwd(bias) / fwd(γ+fp32 残差) / fwd(GELU) / dgrad / dgrad(GELU'+db1) / wgrad，ViT-L M=7203，normal + spiky：对**反量化后的 fp32 乘积**相对 L2 ≤1.7e-3（bf16 输出舍入量级，kernel 正确）；对**原始 bf16 的 fp32 乘积**（量化误差）fwd/dgrad 3.6–3.9%、wgrad 3.8–5.1%（spiky 的 wgrad 个别偏低 1–2%，是尖峰通道主导范数）。其余形状（joint M=7903、ViT-B M=14406/15806）没有单独跑 GEMM 矩阵（`test_gemm_mx8.py` 不带 `--quick` 的完整矩阵没有运行），而是由下面的块测试覆盖：块测试在这些 M 上执行全部 12 个 GEMM 与 epilogue，并与 fp32 块比较。
- **块**（`tests/test_block_mx8.py`，`profile/gateA_block_mx8.json`；真实 DINOv3 `SelfAttentionBlock`、LayerScale、k-masked qkv bias、尖峰残差流、joint mask）：ViT-L/ViT-B × plain/joint × normal/spiky 共 8 个用例，全 4 个 Linear / 仅 fc：out 余弦 ≥0.99993、dx 余弦 ≥0.9999，全局梯度余弦 0.9984–0.9997（round13 的 bf16 块 0.99999），单个参数最低 0.9954（bf16 块 ≥0.99999）。qkv 按 GEMM 种类（f/d/w/fd/dw/fw，ViT-L plain spiky）的块级检查：全局梯度余弦 ≥0.99994，单个参数 ≥0.9996。

### Gate B-lite（改精度准则，Amendment 1 §9.4）

`tests/gate_b_multi.py`：同一进程内用同一权重、同一 batch、同一 RngTape（R0 记录，其余回放，`assert_complete` 通过）依次跑 R0（fp32）、TB1（bf16 参考）和若干候选各一步，比较全部梯度/更新。准则：loss_rel ≤2e-2 且全局 grad 余弦 ≥0.99，**同时对 R0 和对 TB1** 成立；逐参数余弦只记录（要求 `≥ cos(TB1,R0) − 0.05`，违例参数列在 JSON 的 `per_param_offenders_vs_r0`，不单独卡）。每次进程调用的“batch”不同（loader 在进程间不是逐位可复现的，同一进程内所有栈共享同一批），所以 3 个 seed 不是固定的 3 批；下表按“批”计数（来源：`profile/gateB_*_mx8*.json`，共 4 轮：v1 的 3 批、policy bisect 的 3 批、seeds 3–10 的 8 批、推荐策略的 6 批、qkv 种类实验的 4 批，视候选而定）。


**vitl16**

| policy | batches | pass | grad cos vs TB1: min / mean | loss_rel max (vs TB1) | grad_norm_rel max |
|---|---|---|---|---|---|
| MX8_fc | 20 | 19 / 20 | 0.9901 / 0.9970 | 2.9e-03 | 0.024 |
| MX8_fc+proj | 15 | 14 / 15 | 0.9881 / 0.9958 | 2.3e-03 | 0.038 |
| MX8_fc+proj+qkv_dw | 10 | 8 / 10 | 0.9880 / 0.9950 | 4.0e-03 | 0.038 |
| MX8_all | 11 | 11 / 11 | 0.9905 / 0.9946 | 5.3e-03 | 0.013 |
| MX8_attn | 3 | 2 / 3 | 0.9783 / 0.9899 | 6.7e-04 | 0.002 |
| MX8_fc+qkv | 3 | 2 / 3 | 0.9854 / 0.9924 | 1.6e-03 | 0.008 |
| MX8_all+nojoint | 3 | 2 / 3 | 0.9789 / 0.9913 | 1.3e-03 | 0.004 |
| MX8_all+nofirst | 3 | 2 / 3 | 0.9607 / 0.9835 | 1.6e-03 | 0.005 |
| MX8_all+nofirst+nolast+nojoint | 3 | 2 / 3 | 0.9621 / 0.9859 | 1.4e-03 | 0.004 |

**vitb16**

| policy | batches | pass | grad cos vs TB1: min / mean | loss_rel max (vs TB1) | grad_norm_rel max |
|---|---|---|---|---|---|
| MX8_fc | 20 | 19 / 20 | 0.9914 / 0.9960 | 2.4e-03 | 0.016 |
| MX8_fc+proj | 11 | 11 / 11 | 0.9928 / 0.9955 | 2.9e-03 | 0.014 |
| MX8_fc+proj+qkv_dw | 6 | 5 / 6 | 0.9907 / 0.9938 | 1.5e-03 | 0.016 |
| MX8_all | 11 | 7 / 11 | 0.9833 / 0.9913 | 4.8e-03 | 0.014 |
| MX8_attn | 3 | 3 / 3 | 0.9921 / 0.9947 | 2.1e-03 | 0.007 |
| MX8_fc+qkv | 3 | 3 / 3 | 0.9926 / 0.9943 | 2.8e-03 | 0.013 |
| MX8_all+nojoint | 3 | 3 / 3 | 0.9948 / 0.9960 | 4.2e-04 | 0.007 |
| MX8_all+nofirst | 3 | 3 / 3 | 0.9907 / 0.9937 | 2.8e-03 | 0.011 |
| MX8_all+nofirst+nolast+nojoint | 3 | 3 / 3 | 0.9910 / 0.9952 | 2.0e-04 | 0.016 |


**更新余弦与损失**：所有候选、所有批：update 余弦 1.0000（Adam 第一步的更新≈符号），loss_rel 最大 5.3e-3（全部 Linear，ViT-L）/ 4.8e-3（ViT-B），远低于 2e-2；全局 grad_norm_rel ≤0.038。

**读法**：(1) 只有 fc1/fc2：40 批里 38 批通过，两个未通过的批全局余弦分别是 0.9901 对 TB1 / 0.9899 对 R0（ViT-L）、0.9914 对 TB1 / 0.9897 对 R0（ViT-B），都在门槛的 3e-3 内；(2) 加 proj 和 qkv 的 dgrad/wgrad：16 批里 13 批通过，最低 0.9880；(3) 加 qkv 前向（`all`、`attn`、`fc+qkv`）：ViT-B `all` 11 批只有 7 批过（最低 0.9833），ViT-L 在某一批里所有含 qkv 前向的策略都掉到 0.96–0.985（`all+nofirst` 0.9607、`all+nofirst+nolast+nojoint` 0.9621、`all+nojoint` 0.9789、`attn` 0.9783、`fc+qkv` 0.9854），而同一批里 fc / fc+proj 是 0.9995 / 0.9946；(4) 保留首块、末块或 joint 块为 bf16（`nofirst / nolast / nojoint`）不能修复这些批次，噪声不是集中在首末块；同一批里每个块的梯度余弦都只有 0.93–0.97，`queries.weight`、`mask_head` 也只有 0.92–0.96，也就是噪声不集中在某一层，而是整条链上的。

**qkv 按 GEMM 种类的二分**（ViT-L，4 个新批；基线 `fc+proj`；`qkv_f/d/w` = 只有 qkv 的前向/dgrad/wgrad 走 MXFP8）：

| policy (all contain fc+proj) | batch 11 | batch 12 | batch 13 | batch 14 | mean |
|---|---|---|---|---|---|
| fc+proj | 0.9881 | 0.9976 | 0.9930 | 0.9972 | 0.9940 |
| fc+proj + qkv_f | 0.9832 | 0.9906 | 0.9902 | 0.9954 | 0.9899 |
| fc+proj + qkv_d | 0.9881 | 0.9975 | 0.9930 | 0.9972 | 0.9939 |
| fc+proj + qkv_w | 0.9881 | 0.9976 | 0.9930 | 0.9972 | 0.9940 |
| fc+proj + qkv_fd | 0.9832 | 0.9906 | 0.9902 | 0.9955 | 0.9899 |
| fc+proj + qkv_dw | 0.9880 | 0.9975 | 0.9930 | 0.9972 | 0.9939 |
| fc+proj + qkv_fw | 0.9832 | 0.9906 | 0.9902 | 0.9954 | 0.9899 |


**结论**：qkv 的 dgrad 和 wgrad 用 MXFP8 对全局梯度余弦**没有可见影响**（与基线相差 ≤1e-4），而 qkv 的前向（LN1 输出和 qkv 权重的量化）使余弦系统性下降 0.002–0.007。这意味着：前向激活的量化噪声才是这套策略里精度的主要代价，反向 MXFP8 几乎是免费的；这也是 `fc+proj+qkv_dw` 比 `all` 更稳、而 GEMM 节省只少 ~0.12 ms/块的原因（qkv 前向 bf16 195 µs → MXFP8 74 µs）。未验证但值得做的推论：proj / fc 的前向也可能是主要噪声源，“前向 bf16、反向 MXFP8”的策略应该更接近 bf16（GEMM 只省 2/3 的 FLOP），本轮没有实现（`_MXProjRes/_MXFC1/_MXFC2Res` 还没有按种类拆分）。

### Gate S（50 步轨迹，2 个真实 batch，`gate_c(interval=10, max_rel=0.05)`，无 NaN/Inf）

`tests/gate_s_multi.py`：R0、TB1、TB1_R13 和各 MXFP8 策略在同一进程、同两批上各跑 50 步（真实掩码调度，固定 mid / late 相位）。**第一次运行我用了缓存的 R0/TB1 曲线（来自另一进程的另一批数据）与新进程的候选比较，得到 3% 的“偏差”，这是无效比较（连 TB1_R13 对 TB1 都 3%），已删除（文件改名 `*_INVALID_mixed_batches.json` 留作记录），并把缓存键改成所有栈都带同一个 tag 后重跑。**


**vitl16**: 50 steps, 2 batches, gate_c(interval=10, max_rel=0.05); max relative EMA-loss deviation over checkpoints 10..50, grad-norm ratio range

| pair | schedule | fixed mid | fixed late |
|---|---|---|---|
| MX8_all vs R0 | 0.0041 / [0.917,1.018] / PASS | 0.0025 / [0.999,1.011] / PASS | 0.0026 / [0.994,1.011] / PASS |
| MX8_all vs TB1 | 0.0041 / [0.917,1.018] / PASS | 0.0023 / [0.970,1.010] / PASS | 0.0028 / [0.995,1.014] / PASS |
| MX8_fc+proj vs R0 | 0.0024 / [0.997,1.010] / PASS | 0.0022 / [0.998,1.006] / PASS | 0.0016 / [0.998,1.006] / PASS |
| MX8_fc+proj vs TB1 | 0.0023 / [0.998,1.009] / PASS | 0.0019 / [0.969,1.005] / PASS | 0.0018 / [0.999,1.009] / PASS |
| MX8_fc vs R0 | 0.0014 / [0.998,1.005] / PASS | 0.0012 / [0.997,1.004] / PASS | 0.0014 / [0.999,1.007] / PASS |
| MX8_fc vs TB1 | 0.0013 / [0.997,1.005] / PASS | 0.0010 / [0.969,1.002] / PASS | 0.0016 / [0.999,1.007] / PASS |
| MX8_fc+proj+qkv_dw vs R0 | 0.0016 / [0.998,1.026] / PASS | 0.0007 / [0.997,1.022] / PASS | - |
| MX8_fc+proj+qkv_dw vs TB1 | 0.0007 / [0.998,1.024] / PASS | 0.0004 / [0.998,1.020] / PASS | - |
| MX8_fc+proj+qkv_dw vs R0 | - | - | 0.0016 / [0.997,1.006] / PASS |
| MX8_fc+proj+qkv_dw vs TB1 | - | - | 0.0013 / [0.995,1.004] / PASS |

**vitb16**: 50 steps, 2 batches, gate_c(interval=10, max_rel=0.05); max relative EMA-loss deviation over checkpoints 10..50, grad-norm ratio range

| pair | schedule | fixed mid | fixed late |
|---|---|---|---|
| MX8_all vs R0 | 0.0007 / [0.999,1.008] / PASS | 0.0011 / [0.991,1.360] / PASS | 0.0015 / [0.993,1.281] / PASS |
| MX8_all vs TB1 | 0.0006 / [0.999,1.012] / PASS | 0.0005 / [0.990,1.003] / PASS | 0.0013 / [0.994,1.279] / PASS |
| MX8_fc+proj vs R0 | 0.0006 / [0.993,1.127] / PASS | 0.0007 / [0.919,1.004] / PASS | 0.0006 / [0.951,1.007] / PASS |
| MX8_fc+proj vs TB1 | 0.0005 / [0.994,1.126] / PASS | 0.0017 / [0.672,0.999] / PASS | 0.0006 / [0.949,1.008] / PASS |
| MX8_fc vs R0 | 0.0009 / [0.996,1.079] / PASS | 0.0012 / [0.915,1.003] / PASS | 0.0005 / [0.950,1.002] / PASS |
| MX8_fc vs TB1 | 0.0008 / [0.999,1.079] / PASS | 0.0026 / [0.670,1.004] / PASS | 0.0007 / [0.949,1.002] / PASS |
| MX8_fc+proj+qkv_dw vs R0 | 0.0012 / [0.944,1.002] / PASS | 0.0007 / [0.992,1.007] / PASS | - |
| MX8_fc+proj+qkv_dw vs TB1 | 0.0012 / [0.945,0.999] / PASS | 0.0008 / [0.992,1.008] / PASS | - |
| MX8_fc+proj+qkv_dw vs R0 | - | - | 0.0007 / [0.991,1.003] / PASS |
| MX8_fc+proj+qkv_dw vs TB1 | - | - | 0.0008 / [0.992,1.003] / PASS |


结果：所有 MXFP8 策略对 R0 和对 TB1 的 EMA 损失最大相对偏差 ≤4.1e-3（ViT-L 全部 Linear）/ ≤2.6e-3（ViT-B），门槛 5e-2；梯度范数比（检查点 10…50 上的单步比值）的范围：ViT-L 全部 Linear [0.917, 1.018]，ViT-B [0.67, 1.36]（门槛 [0.1,10]；对照：bf16 TB1 对 R0 为 [0.995, 1.031]，TB1_R13 对 TB1 为 [0.99, 1.00]），也就是 MXFP8 的单步梯度范数在个别检查点上会偏 ±30%，但损失曲线不受影响；全部有限，全部 PASS。推荐策略 `fc+proj+qkv_dw`：schedule + 固定 mid + 固定 late（late 最后补跑），全 PASS。


## 6. 静态分析与 roofline

| kernel | regs | spill/stack B | smem B | 备注 |
|---|---|---|---|---|
| `quant_dual_kernel<bf16/fp32, row+col>` | 72 | 0 | 5248 | 128 线程 CTA，32 行×128 列；硬件 e4m3x2 饱和转换 `F2FP.SATFINITE.E4M3.F32.PACK_AB` |
| `quant_dual_kernel<…, row-only>` | 54 | 0 | 0 | |
| `quant_dual_kernel<…, col-only>` | 63–64 | 0 | 5248 | |

`cuobjdump --dump-resource-usage` / `-sass`（ncu 不可用）。GEMM 由 QuACK（CuTe DSL）运行时生成，不做 SASS 计数，用实测 TFLOPS 对 roofline 评价。

**Roofline（ViT-L plain 块 fwd+bwd 的 12 个 GEMM）**：FLOP = 3 × 2 × 7203 × 1024² × 12 = 543.8 GFLOP（wgrad 因 Mpad 多 1.3%）；bf16 在 230 TFLOPS 下限 2.36 ms，实测 2.58 ms（91%）；MXFP8 在探针峰值 ~650 TFLOPS 下限 0.84 ms，实测 1.04 ms（80%，wgrad 的 proj/qkv 波量化和 MN-major 读取拖低）。量化器：h / dz（29.5M 元素）双输出 = 59 MB 读 + 59 MB 写，GDDR7 ~1.79 TB/s 下限 66 µs，实测 64–65 µs（在 DRAM roofline 上）；小张量（≤15 MB，L2 常驻）7.5 µs，即 3.9 TB/s 的 L2 吞吐。

**失败 / 未做 / 教训（如实）**
- **Gate S 第一次结果无效**：我复用了缓存的 R0/TB1 曲线（另一进程另一批数据）去比候选，得到 ~3% 的“偏差”（连 TB1_R13 对 TB1 都 3%）；已把文件改名 `gateS_*_mx8_INVALID_mixed_batches.json` 留档，缓存键改为所有栈共用同一个 tag 后重跑，所有数字来自重跑。
- **qkv 前向走 MXFP8 的策略（`all`、`attn`、`fc+qkv`）精度不稳**（ViT-B `all` 7/11 批通过，ViT-L 某批降到 0.96–0.985），所以不推荐；保留首块/末块/joint 块为 bf16（`nofirst/nolast/nojoint`）修复不了。
- 若干 bench / nsys 被 `assert_gpu_exclusive` 拒绝（别的进程占着 GPU 上下文），用 `bench_queue.sh` / `nsys_queue.sh` 的重试包装重跑；r13 的 TB1 整步 nsys 就是这样丢的，本轮补采了 ViT-L 的 `step_vitl16_TB1_v2`；ViT-B 的 TB1 整步 nsys 没有补采。
- **没做**：Spark `fp8gemm` 的改编（只有 K-major、无 epilogue、K%256，见 §2）；Triton 量化器候选；生产者融合量化（GELU epilogue 出 h、LN 输出、γ⊙dX、权重量化进优化器）；wgrad split-K 的整步对照；`test_gemm_mx8.py` 的完整形状矩阵（只跑了 `--quick`，其余形状由块测试覆盖）；proj/fc 前向 bf16、反向 MXFP8 的策略（`_MXProjRes/_MXFC1/_MXFC2Res` 还没有按种类拆分）；e2e（200 步真实 loader）；max batch。这些字段在 results.json 里为 null。
- **未确认的猜测**：量化器在 h/dz 上 dual（64 µs）比分开跑 row-only + col-only（L2 常驻时合计 32 µs）慢，但那是热循环下的数字；真实步里 h 刚由 fc1 epilogue 写出，部分仍在 L2，nsys 平均 21.7 µs/次（192 次共 4.17 ms）说明实际在两者之间，没有单独测。

## 7. 结论与下一瓶颈

**结论**
- **做成了**：自研双输出 MXFP8 量化器（逐位等于 QuACK rceil）+ QuACK block-scaled GEMM（原生 MN-major 操作数，满速）+ round13 的融合 epilogue，整块 fwd+bwd 在 graph 回放下 ViT-L plain 3.63 → 2.30 ms（全 4 个 Linear，1.58×）/ 2.63（fc+proj，1.38×）/ 2.72（fc，1.33×）；整步（eager，三相加权）ViT-L 43.72 → 50.06 / 50.44 / 51.56 / 52.09 img/s（fc / fc+proj / fc+proj+qkv_dw / all，1.145–1.191×），ViT-B 100.67 → 110.46 / 108.91 / 110.03 / 112.29（1.082–1.115×）。相对 R0（16.2 / 37.2）3.0–3.2×，相对 TB1（41.4 / 96.0）1.15–1.26×。
- **量化开销**：每块 203 µs（8 个激活/梯度双输出 + 4 个权重），整步 5.1 ms（ViT-L，kernel 总时间的 4.2–4.4%）/ 4.1 ms（ViT-B）；h / dz 两个最大的张量各 ~64 µs（DRAM 受限），占一半。**未做**生产者融合（GELU epilogue 里直接出量化的 h、LN 输出、γ⊙dX、权重量化进优化器）：eager 下整步已是 host 受限，GPU 侧再省几 ms 看不到；这 5.1 ms 要在 graph 之后才有意义。
- **精度**：单步梯度余弦对 bf16 参考是边缘的——fc 38/40 批、fc+proj+qkv_dw 13/16 批、全部 Linear ViT-B 7/11 批 ≥0.99；所有批 loss_rel ≤5.3e-3、update 余弦 1.0000；Gate S 所有策略全过（EMA 损失偏差 ≤4.1e-3）。qkv 的**前向**是精度代价的主要来源，qkv 的 dgrad/wgrad 免费。
- **峰值显存**：ViT-L 17.48 → 17.07–17.25 GB，ViT-B 15.31 → 14.80–14.88 GB（只省 0.2–0.5 GB，峰值不在块的激活上）。

**什么限制了剩下的收益**（ViT-L 全部 Linear：kernel 117.0 ms，wall 134.5 ms）
1. **host launch**：GPU 忙碌率 0.888 → 0.805，>20 µs 间隙 126 → 275 个（7.5 → 14.7 ms）；MXFP8 把块的 GPU 时间压到 2.3 ms，已经低于 eager 下块的 host enqueue（~2.3 ms，QuACK epilogue/GEMM 的 Python 派发 100–165 µs/次 + 12 次量化 33 µs/次 + autograd）。每块 +12 次量化 kernel 和更贵的 block-scaled 派发是新增的 host 开销。整步 kernel 总时间到 117 ms 的潜力只有 CUDA graph 能兑现：块级 graph 回放已验证可捕获（全部变体、plain/joint、两个模型）；r16 做整步 graph 后预期 wall ≈ kernel 总时间 + 约 5–7 ms 间隙（估计，未测）。
2. head 卷积 22.1 ms（ViT-B 30.4 ms，27%）、elementwise 19.6 ms（ViT-B 28.9 ms）、attention 16.4 ms、优化器 10.4 ms、LN/行 kernel 9.2 ms、量化 5.1 ms、bf16 的 head GEMM 6.2 ms：现在骨干 GEMM（MXFP8 27.6 ms，23.6%）已经不是最大块，ViT-B 上 head 卷积 + elementwise 占一半以上。

**下一步（按预期收益）**
1. 整步 CUDA graph（r16）里验证 MXFP8 块：预期把 ViT-L 从 ~135 ms 推向 ~122 ms；这时才值得做 2 和 3。
2. 把 `fc+proj+qkv_dw` 里没拆的部分按 GEMM 种类拆开，试“前向 bf16 / 反向 MXFP8”（proj、fc1、fc2 的 `kinds`）以及只对 fc 的前向保持 MXFP8：用 Gate B-lite 的批次分布（而不是 3 批）选策略。
3. 生产者融合量化（按大小）：fc1 的 GELU epilogue 直接出 h 的行副本（QuACK 的 SFD 量化 epilogue 对 aux 输出只支持行方向，列副本仍需单独一次读 z 重算 GELU）、dgrad 的 GELU' epilogue 出 dz、LN 输出、`γ⊙dX`；权重量化接到 r15 的 `register_weight_writer`（每步量化 12.6M×24 个权重 ~1 ms）。预期合计 2–3 ms/步（graph 之后）。
4. proj / qkv 的 wgrad（314–421 TFLOPS，输出 tile 数不足 170 个 SM）：试 split-K（`R17_WGRAD_SPLITK=auto`，未整步对照）。

**契约变更请求**
1. CONTRACTS §4 的 `MX8Tensor` 建议改为本轮的 `MX8`：`qr/sfr`（行副本，K-major，scale 沿 C）+ `qc/sfc`（列副本，行主序，scale 沿 token 轴，rows 填充到 128 的倍数且 pad 行为 0），scale 为 QuACK/cuBLAS 的 128×4 swizzle `(rm, rk, 32, 4, 4)` UE8M0，rceil 公式已对 QuACK `to_mx` 逐位验证。
2. 门 B 的改精度准则“全局 grad 余弦 ≥0.99”在单步上对 MXFP8 是边缘的（见 §5，批间波动 0.988–0.9995）；建议改成多批统计准则（例如 ≥8 批、中位数 ≥0.995、最低 ≥0.98）或加 Level C/D 判据，而不是单步 3 个 seed。
3. `common/bench.assert_gpu_exclusive` 对“别的代理留下的空闲 CUDA 上下文”会直接中止测量/nsys（本轮多次发生，r13 的 TB1 整步 nsys 就是这样丢的）；建议所有 GPU 进程都用同一个锁并在持锁区间外释放上下文，或让该检查只看锁持有者。
4. `gate_s` 的 R0/TB1 曲线不可跨进程缓存（批数据在进程间不同）——`check.curve` 文档里应写明。

