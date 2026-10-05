# Round13：BF16 GEMM 融合整块（sm_120）（负责：Agent C）

> 本文件最初由 `tests/assemble_notes.py` 从 `tests/NOTES.template.md` 生成；§3、§5 的 Gate B/S 和 §7 由 C2 在 2026-10-03 直接写入（不要再运行 assemble_notes.py，会覆盖）。数字全部实测，缺项写 pending/null。
> 全部数字都在**新容器**（RTX 5090，575 W 功耗上限，`infer-5090` = connect.bjb2.seetacloud.com:54793）上重测；旧容器（600 W）的早期数据只放在 `profile/old_machine/`。
> 参考栈 TB1 = Agent A 的 T-BF16：`r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16`；候选 `TB1_R13` = TB1 + `round13_gemm_fusion_sm120:full`。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 整步（TB1 → TB1+R13，三相加权） | ViT-L B=7：40.79 → 44.35 img/s（**1.087×**）；ViT-B B=14：95.90 → 103.20 img/s（**1.076×**）；详见 §4.5 |
| 单块 GPU 时间（CUDA graph 回放，同一 attention） | ViT-L plain 块 fwd+bwd 4.22 → 3.65 ms（1.157×），ViT-B plain 5.13 → 4.36 ms（1.177×） |
| 每块 kernel（不含 attention） | 前向 6（LN1、qkv、proj+epilogue、LN2、fc1+epilogue、fc2+epilogue），反向的 GEMM 8 + 行 kernel 3 + 规约 finalize 若干，见 §3 |
| GEMM 选择 | fwd / dgrad：QuACK SM120（带自定义 epilogue）；wgrad：按形状在 cuBLAS 与 QuACK（含 split-K）之间选；细节 §4.1 |
| 正确性 | Gate A、B、S 见 §5；Gate B 的逐参数余弦门槛按 bf16 自噪声校准 |
| 契约变更请求 | 见 §7（逐参数 grad cos ≥ 0.99 对 bf16-vs-bf16 同精度换 kernel 不可达，建议改为校准准则） |

## 1. 目标与假设

**目标**：把 DINOv3 训练块 `x1 = x + γ1·proj(attn(LN1 x))`、`x2 = x1 + γ2·fc2(GELU_erf(fc1(LN2 x1)))`（fp32 残差流）做成 `ops.block` 覆盖：每个 Linear 的 fwd / dgrad / wgrad 用最快的 BF16 GEMM，并把 bias、LayerScale、残差、GELU / GELU'、bias 梯度、LN 前反向、γ 梯度等访存型 kernel 融合进 GEMM epilogue 或少数几个行 kernel。

**假设（以实测核对）**

1. GEMM 选择本身有收益：QuACK SM120 GEMM 在 fwd 上 223–234 TFLOPS，cuBLAS 176–219（§4.1）。
2. 融合收益来自去掉整遍显存扫描：fp32 残差乘加、GELU、bias 梯度列规约、`g = γ⊙dX` 的 bf16 化等在 T-BF16 里是十几个独立 kernel/块（§3 实测）。
3. BF16 GEMM 受 FP32 累加 HMMA 的 ~230 TFLOPS 限速；融合后块时间下界约为 GEMM 时间，再往下要靠 MXFP8（r17）。
4. 5090 有 170 个 SM；wgrad 的输出只有 256 个 128×128 块（K=M=7203 很长），2 个不整波，所以 wgrad 受波量化限制（§4.1）。

## 2. 改动

| 文件 | 内容 |
|---|---|
| `impl/block_kernels.cu/.py` | CUDA C++（sm_120）：`ln_fwd`（fp32→bf16 LN + 均值/rstd）、`rowbwd`（LN 反向 + 残差梯度加 + `g=γ⊙dX`(bf16) + dW/dB/dγ/dbias 列部分和，3 个模板实例）、`colsum_bf16`（qkv bias 梯度）、`finalize`（定序两级规约，逐位确定） |
| `impl/gemm_ops.py` | GEMM 层：QuACK 自定义 epilogue（`@gemm_epilogue`）、cuBLAS 回退、按 (kind, linear[, 隐藏维]) 的后端策略表 `POLICY` / `WGRAD_AUTO` |
| `impl/fused_block.py` | 6 个 `torch.autograd.Function` 组成的块；`Shadows`（bf16 权重影子，每次前向在 block 0 刷新，或由 `ctx.meta['shadow_writer']` / r11 `bf16_shadow` 接管） |
| `impl/patch.py` | `ops.block` 覆盖（变体 `full` / `wgrad_quack` / `torch_gemm`），保留旧 `ops.block` 作为不支持块的回退（`fused_block.eligible`） |
| `impl/spark_hgemm_ext.py` + `impl/third_party/spark_hgemm/` | Spark hgemm（MIT，aidxhxr/5090-blackwell-kernels@a401bfa9e363，原样复制）的 JIT 构建，仅作候选实测 |
| `impl/third_party/r5_lt_epilogue/` | r5 `lt_epilogue.cu`（项目内部）原样复制，仅作 cuBLASLt epilogue 候选实测 |
| `tests/` | 候选矩阵、Gate A、Gate B/校准、nsys 驱动与分类、静态分析、CPU 协议测试（`tests/test_round13_cpu.py`，5 项通过） |
| `run.py` | 整步计时 / gate B / gate S / nsys 入口，改编自 `round11_bf16_mixed/run.py`（新增 `r13` 简称和 `TB1*` 预设） |

### 融合块设计

前向（不含 attention）：

| 步骤 | kernel | 融合内容 |
|---|---|---|
| `_LN1` | 1 个 CUDA kernel | fp32 x → bf16 LN1 输出 + 均值/rstd；x 作为别名输出返回，使残差路径梯度在 `_LN1.backward` 里与 LN 梯度融合相加（省一次 autograd 加法） |
| `_Linear`(qkv) | 1 个 QuACK GEMM | bias epilogue，直接写 `[B,N,3d]` 打包 bf16 |
| `ctx.ops.attn_core` | 由 r11/r12 提供 | 契约：qkv `[B,N,3d]` 进，`[B,N,d]` 出 |
| `_ProjRes` | 1 个 QuACK GEMM + 1 个 LN2 kernel | epilogue：`y=acc+b`（bf16，保存给 dγ）、`x1 = x + γ1⊙y`（fp32）；LN2 读 x1 |
| `_FC1` | 1 个 QuACK GEMM | epilogue：`z=acc+b`（bf16，保存给 GELU'）、`h=GELU_erf(z)`（bf16） |
| `_FC2Res` | 1 个 QuACK GEMM | 同 proj，γ2 |

反向：

| 步骤 | kernel | 融合内容 |
|---|---|---|
| `_FC2Res.bwd` | 1 个 `rowbwd` + GEMM(dgrad) + GEMM(wgrad) | `dy2=γ2⊙g`(bf16)、dγ2、db2 部分和；dgrad epilogue 里 `dz=(dy2·W2)⊙GELU'(z)` 并规约 db1（fc1 的 bias 梯度）；wgrad 得到 fp32 直接作为 Parameter 梯度 |
| `_FC1.bwd` | 2 个 GEMM | dgrad、wgrad |
| `_ProjRes.bwd` | 1 个 `rowbwd` + 2 个 GEMM | LN2 反向 + 残差梯度加 + `dyp=γ1⊙dx1`(bf16) + dγ1、dbp、dLN2 部分和 |
| attention 反向 | autograd（r11/r12） | — |
| `_Linear.bwd`(qkv) | 2 个 GEMM + `colsum_bf16` | qkv bias 梯度（k 部分的 bias 掩码由块外的 autograd 乘法处理） |
| `_LN1.bwd` | 1 个 `rowbwd` | `dx = dxr + LN_bwd(dln1)` |

每个 `rowbwd`/`colsum` 后各跟一个 `finalize`（对 ≤170 个 CTA 部分和做定序求和，逐位确定，无原子）。γ、LN 权重、偏置的梯度一次得到，不再有独立的 `sum(0)`。

精度：残差流、LN 统计、规约、梯度部分和全部 fp32；GEMM 操作数 bf16、fp32 累加；Parameter 与 `state_dict` 不变（harness 的 `apply_patches` 检查通过），权重经 bf16 影子进入 GEMM；wgrad 输出 fp32，直接作为 Parameter 梯度返回。

可捕获性（CONTRACTS §5）：所有 kernel 在 current stream 发射；QuACK JIT/autotune 在首个预热步完成；块 fwd+bwd 在 `bench_block.py` 中实际用 `torch.cuda.graph` 捕获并回放（所有变体、plain/joint、两个模型）。

### 候选取舍（实测依据在 §4）

| 候选 | 结论 |
|---|---|
| torch / cuBLAS（`torch.mm`、`F.linear` 的 Lt bias epilogue） | wgrad 的 fc1/fc2（ViT-L）和 proj（ViT-B）最快；fwd/dgrad 比 QuACK 慢 4–25% |
| r5 `lt_epilogue`（cuBLASLt BIAS / GELU_AUX_BIAS） | erf 版 = Lt bias + 单独 Triton GELU，与 torch 持平；tanh 版（融合 GELU）快 14% 但语义不是 erf，且无 z/h 双输出，不采用 |
| QuACK 0.6.5 stock `gemm_act` / `gemm_dact` / `gemm_add` | 只支持 tanh GELU、无 γ、无 fp32 残差流；用其 `@gemm_epilogue` frontend 写自定义 epilogue（erf、γ、fp32 残差、aux 输出、bias 梯度规约）后开销仅 +5–12 µs/GEMM |
| Spark hgemm v4/v6（TMA + stream-K，MIT） | 只有行主序 NN、bf16 输出；dgrad 上 v6 比 QuACK 快 4–9%，但 stream-K 的 `epoch` 是 host 递增后写进 kernel 参数，**CUDA graph 回放不安全**（旧 epoch 的 flag 会被当成有效），且 workspace 按需 `cudaMalloc`，违反契约 §5，不采用；wgrad 要求 K=M%64==0 且 A 转置，不支持 |
| CUTLASS 4.2 C++ sm120 | `sm120_mma_builder.inl` 的非 block-scaled collective 只接受 F8F6F4（static_assert），没有 BF16 sm120 mainloop；示例 79/87 都是 block-scaled/FP8。BF16 只能走 sm80 风格 cp.async kernel，预期不超过 cuBLAS，未实测（不假装有数字） |

## 3. nsys 分析过程

**命令**（全部持双锁，原始 `.nsys-rep/.sqlite` 在 `/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/nsys/`）：

```bash
# 单块：每个变体 5 次 fwd+bwd，NVTX 分 fwd/bwd；tests/nsys_categorize.py 按 NVTX 范围分类
common/profile_nsys.sh $OUT/blk_<variant>_a tests/profile_block.py --variant {autocast_default|r5_ln_fuse|r13_full} --model vitl16
python tests/nsys_categorize.py $OUT/blk_<variant>_a --json profile/nsys_block_<variant>.json
# 整步：ViT-L / ViT-B，mid 相，预热 20 步、采 5 步
common/profile_nsys.sh $OUT/step_<model>_<stack>_a run.py --model <model> --stack {TB1|TB1_R13} --action profile --phases mid --warmup 20
```

**3.1 单块（ViT-L plain，同一 flash attention + triton RoPE，CUDA graph 之外的 nsys 5 次 fwd+bwd，kernel 时间求和）**

| 变体 | fwd kernels | fwd GPU µs | bwd kernels | bwd GPU µs | fwd+bwd µs |
|---|---|---|---|---|---|
| autocast_default（torch autocast + r11） | 24 | 1370 | 43 | 2900 | 4270 |
| r5_ln_fuse（TB1 的块） | 20 | 1337 | 37 | 2790 | 4127 |
| r13_full | 10 | 1146 | 23 | 2447 | 3593 |

r13_full 的 fwd 4 个 GEMM 共 867 µs（含 epilogue），attention+RoPE 201 µs，LN 2 个 kernel 21 µs；bwd 8 个 GEMM 共 1763 µs（4 个 QuACK dgrad/wgrad 共 805 µs、dgelu epilogue GEMM 288 µs，2 个 cuBLAS wgrad 共 594 µs，另有 1 个 75 µs 的小 GEMM），attention 468 µs，行 kernel 8 个共 172 µs（3 个 `rowbwd` 141 µs + 4 个 `finalize` 19 µs + colsum 11 µs）。相对 r5_ln_fuse，块的 kernel 数从 57 降到 33，elementwise/copy/reduce 从 35 个 512 µs 降到 5 个 100 µs；GEMM 本身时间只下降 ~136 µs（2766 → 2630 µs）（QuACK 比 cutlass_80 快一些），整块 3593 vs 4127 µs（kernel 求和，1.149×），与 CUDA graph 回放的 §4.4（1.157×）一致。

**3.2 round13 块的 kernel 清单（ViT-L plain，一次 fwd + bwd）**
fwd：`ln_fwd`×2（LN1、LN2，各约 10.6 µs）、QuACK GEMM×4（bias_epi 201 µs、res_epi×2 共 386 µs、gelu_epi 280 µs）、flash_fwd 177 µs、`_rotate` 24 µs、1 个 multi_tensor 刷新 bf16 影子（只在 block 0）。bwd：`rowbwd`×3（fc2 的 γ 预处理 / LN2 + proj 预处理 / LN1，共 141 µs）、`finalize`×4、QuACK dgrad/dgelu GEMM、cuBLAS wgrad（fc1/fc2）、QuACK wgrad（qkv/proj）、flash_bwd 418 µs + dot_do_o 12 µs + convert_dq 9 µs、`_rotate` bwd 28 µs、cat 35 µs、`colsum_bf16` 11 µs。

**3.3 整步（TB1_R13，mid 相，预热 20 步、采 5 步；原始文件在 tmp 的 nsys 目录，类别表 `profile/nsys_step_cat_*.json`）**

命令：`common/profile_nsys.sh $OUT/step_<model>_TB1_R13_a run.py --model <model> --stack TB1_R13 --action profile --phases mid --warmup 20`，再 `common/nsys_analyze.py`；类别表用 `round17_mxfp8_sm120/tests/nsys_step_categories.py`（按 kernel 名分类）。

| 类别 | ViT-L ms/步 | % | ViT-B ms/步 | % |
|---|---|---|---|---|
| GEMM bf16（QuACK，fwd/dgrad/部分 wgrad） | 49.1 | 33.2 | 29.7 | 23.4 |
| head 卷积（cuDNN/cutlass conv） | 22.2 | 15.0 | 30.3 | 23.9 |
| GEMM bf16（cuBLAS/cutlass sm80：fc wgrad 与 head 的 GEMM） | 20.7 | 14.0 | 11.2 | 8.8 |
| elementwise/copy/reduce/其他 | 19.3 | 13.1 | 28.9 | 22.8 |
| attention（flash + RoPE） | 16.1 | 10.9 | 12.8 | 10.1 |
| 优化器 + grad clip | 11.6 | 7.8 | 4.0 | 3.2 |
| LN / 行 kernel（r13 CUDA） | 8.5 | 5.7 | 9.6 | 7.6 |
| loss / matcher | 0.2 | 0.2 | 0.5 | 0.4 |
| **kernel 总时间** | **147.8** | 100 | **127.0** | 100 |

ViT-L：3010 个 kernel/步，GPU 忙 89.9%，8 次同步/步，>20 µs 的间隙 109.8 个/步共 6.26 ms；ViT-B：2042 个 kernel/步，GPU 忙 90.5%，8 次同步。TB1 的整步 nsys 在 r13 里没有采到：`nsys_step_vitl16_TB1_a.log` 显示 `assert_gpu_exclusive` 拒绝启动（当时另一个代理的进程在占用 GPU，`GPU has other compute processes`），`nsys_step_*_TB1.txt` 因此是空的；本轮对比改用 bench 的 mid 步时（171.8 → 158.0 ms，ViT-B 146.1 → 135.8 ms）和单块 nsys §3.1，TB1 的整步 nsys 由 round17 的 nsys 一并补采（见 round17 NOTES §3）。

**3.4 分析结论**
1. r13 去掉的是访存型 kernel 与 autocast 的 cast：每块 57 → 33 个 kernel，整步 kernel 总时间 mid 171.8 → 158.0 ms（−13.8 ms，8.0%），其中 GEMM 本身只贡献一小部分（QuACK 比 cutlass_80 快 4–25%），主要是 elementwise/reduce 与 LN 的合并。
2. 剩下的时间：骨干 GEMM（QuACK 49 + cuBLAS/sm80 的 wgrad 约 15）约 64 ms，ViT-L 步里 43%；它们已经在 bf16 FP32 累加限速的 ~200–230 TFLOPS 上，只有换到 block-scaled MXFP8（r17）才能再动。
3. head 卷积 22 ms（ViT-B 30 ms，24%）是第二大块，attention 16 ms，优化器 11.6 ms；每步仍有 8 次同步和 ~110 个 >20 µs 间隙（6 ms，3.8%）。

## 4. 实测结果

### 4.1 GEMM 候选矩阵（12 个 GEMM × 2 模型 × plain/joint M）

形状：ViT-L d=1024，M=7203（plain）/ 7903（joint）；ViT-B d=768，M=14406 / 15806；qkv N=3d,K=d；proj d→d；fc1 d→4d；fc2 4d→d。
布局：fwd `X·Wᵀ`（B 为 K-major 视图）、dgrad `dY·W`、wgrad `dYᵀ·X`（fp32 输出，所有候选统一）。计时：CUDA graph 5 次调用，3 轮交替 × 30 次取中位数，同进程同锁，roofline = 230 TFLOPS。

**vitl16, M=7203 (plain)** (us / TFLOPS, % of the 230 TF roofline; Spark v4 omitted: always slower than v6)

| GEMM | torch (cuBLAS) | QuACK | QuACK split-K | Spark v6 | best |
|---|---|---|---|---|---|
| qkv fwd | 257.4 / 176 (77%) | 193.7 / 234 (102%) | - | 202.6 / 224 (97%) | **quack** |
| qkv dgrad | 224.5 / 202 (88%) | 218.6 / 207 (90%) | - | 202.2 / 224 (97%) | **spark6** |
| qkv wgrad | 231.4 / 196 (85%) | 215.2 / 211 (92%) | 215.2 / 211 (92%) | - | **quack** |
| proj fwd | 80.4 / 188 (82%) | 74.9 / 202 (88%) | - | 72.5 / 208 (91%) | **spark6** |
| proj dgrad | 80.1 / 188 (82%) | 75.1 / 201 (87%) | - | 72.4 / 209 (91%) | **spark6** |
| proj wgrad | 87.4 / 173 (75%) | 87.8 / 172 (75%) | 87.9 / 172 (75%) | - | **torch** |
| fc1 fwd | 311.7 / 194 (84%) | 266.7 / 227 (99%) | - | 269.2 / 224 (98%) | **quack** |
| fc1 dgrad | 297.4 / 203 (88%) | 289.4 / 209 (91%) | - | 267.4 / 226 (98%) | **spark6** |
| fc1 wgrad | 307.2 / 197 (86%) | 339.0 / 178 (77%) | 342.5 / 176 (77%) | - | **torch** |
| fc2 fwd | 305.4 / 198 (86%) | 289.6 / 209 (91%) | - | 267.2 / 226 (98%) | **spark6** |
| fc2 dgrad | 308.8 / 196 (85%) | 267.0 / 226 (98%) | - | 269.0 / 225 (98%) | **quack** |
| fc2 wgrad | 305.5 / 198 (86%) | 342.3 / 177 (77%) | 342.5 / 176 (77%) | - | **torch** |

**vitl16, M=7903 (joint)** (us / TFLOPS, % of the 230 TF roofline; Spark v4 omitted: always slower than v6)

| GEMM | torch (cuBLAS) | QuACK | QuACK split-K | Spark v6 | best |
|---|---|---|---|---|---|
| qkv fwd | 255.9 / 194 (84%) | 216.7 / 229 (100%) | - | 219.3 / 227 (99%) | **quack** |
| qkv dgrad | 222.8 / 223 (97%) | 217.8 / 228 (99%) | - | 218.1 / 228 (99%) | **quack** |
| qkv wgrad | 250.7 / 198 (86%) | 234.3 / 212 (92%) | 234.2 / 212 (92%) | - | **quack_sk** |
| proj fwd | 80.4 / 206 (90%) | 75.1 / 221 (96%) | - | 78.0 / 212 (92%) | **quack** |
| proj dgrad | 80.1 / 207 (90%) | 75.2 / 221 (96%) | - | 77.8 / 213 (93%) | **quack** |
| proj wgrad | 101.5 / 163 (71%) | 95.6 / 173 (75%) | 95.7 / 173 (75%) | - | **quack** |
| fc1 fwd | 309.6 / 214 (93%) | 290.5 / 228 (99%) | - | 292.1 / 227 (99%) | **quack** |
| fc1 dgrad | 299.7 / 221 (96%) | 290.0 / 229 (99%) | - | 290.4 / 228 (99%) | **quack** |
| fc1 wgrad | 333.1 / 199 (87%) | 373.4 / 178 (77%) | 373.3 / 178 (77%) | - | **torch** |
| fc2 fwd | 300.5 / 221 (96%) | 290.5 / 228 (99%) | - | 290.6 / 228 (99%) | **quack** |
| fc2 dgrad | 307.9 / 215 (94%) | 291.7 / 227 (99%) | - | 292.6 / 227 (98%) | **quack** |
| fc2 wgrad | 332.0 / 200 (87%) | 375.2 / 177 (77%) | 375.5 / 177 (77%) | - | **torch** |

**vitb16, M=14406 (plain)** (us / TFLOPS, % of the 230 TF roofline; Spark v4 omitted: always slower than v6)

| GEMM | torch (cuBLAS) | QuACK | QuACK split-K | Spark v6 | best |
|---|---|---|---|---|---|
| qkv fwd | 241.9 / 211 (92%) | 219.8 / 232 (101%) | - | 229.1 / 222 (97%) | **quack** |
| qkv dgrad | 225.2 / 226 (98%) | 218.7 / 233 (101%) | - | 226.6 / 225 (98%) | **quack** |
| qkv wgrad | 267.0 / 191 (83%) | 340.9 / 150 (65%) | 262.2 / 194 (85%) | - | **quack_sk** |
| proj fwd | 83.7 / 203 (88%) | 75.6 / 225 (98%) | - | 81.9 / 208 (90%) | **quack** |
| proj dgrad | 82.8 / 205 (89%) | 75.9 / 224 (97%) | - | 81.7 / 208 (90%) | **quack** |
| proj wgrad | 84.6 / 201 (87%) | 169.3 / 100 (44%) | 90.0 / 189 (82%) | - | **torch** |
| fc1 fwd | 318.7 / 213 (93%) | 304.5 / 223 (97%) | - | 303.0 / 224 (98%) | **spark6** |
| fc1 dgrad | 317.0 / 214 (93%) | 292.4 / 233 (101%) | - | 316.0 / 215 (94%) | **quack** |
| fc1 wgrad | 325.1 / 209 (91%) | 343.2 / 198 (86%) | 335.6 / 203 (88%) | - | **torch** |
| fc2 fwd | 310.2 / 219 (95%) | 292.4 / 233 (101%) | - | 314.5 / 216 (94%) | **quack** |
| fc2 dgrad | 316.7 / 215 (93%) | 293.7 / 231 (101%) | - | 308.0 / 221 (96%) | **quack** |
| fc2 wgrad | 344.6 / 197 (86%) | 343.6 / 198 (86%) | 335.8 / 202 (88%) | - | **quack_sk** |

**vitb16, M=15806 (joint)** (us / TFLOPS, % of the 230 TF roofline; Spark v4 omitted: always slower than v6)

| GEMM | torch (cuBLAS) | QuACK | QuACK split-K | Spark v6 | best |
|---|---|---|---|---|---|
| qkv fwd | 278.9 / 201 (87%) | 253.3 / 221 (96%) | - | 250.1 / 224 (97%) | **spark6** |
| qkv dgrad | 262.4 / 213 (93%) | 248.2 / 225 (98%) | - | 264.7 / 211 (92%) | **quack** |
| qkv wgrad | 292.7 / 191 (83%) | 372.3 / 150 (65%) | 286.8 / 195 (85%) | - | **quack_sk** |
| proj fwd | 102.9 / 181 (79%) | 87.8 / 212 (92%) | - | 88.9 / 210 (91%) | **quack** |
| proj dgrad | 102.3 / 182 (79%) | 87.7 / 213 (92%) | - | 88.7 / 210 (91%) | **quack** |
| proj wgrad | 91.3 / 204 (89%) | 185.6 / 100 (44%) | 98.2 / 190 (83%) | - | **torch** |
| fc1 fwd | 358.9 / 208 (90%) | 322.6 / 231 (101%) | - | 332.9 / 224 (97%) | **quack** |
| fc1 dgrad | 354.9 / 210 (91%) | 329.2 / 227 (99%) | - | 351.6 / 212 (92%) | **quack** |
| fc1 wgrad | 355.5 / 210 (91%) | 375.2 / 199 (86%) | 364.0 / 205 (89%) | - | **torch** |
| fc2 fwd | 356.0 / 209 (91%) | 330.1 / 226 (98%) | - | 351.3 / 212 (92%) | **quack** |
| fc2 dgrad | 355.0 / 210 (91%) | 323.8 / 230 (100%) | - | 337.6 / 221 (96%) | **quack** |
| fc2 wgrad | 377.7 / 197 (86%) | 375.1 / 199 (86%) | 363.8 / 205 (89%) | - | **quack_sk** |

读表要点：fwd 上 QuACK 的 223–234 TF 已在 roofline 上；dgrad 略低（207–233）；wgrad 最差（cuBLAS 196–209，QuACK 172–212）。ViT-L 的 fc1/fc2 wgrad 是 4096×1024 输出 = 256 个 128×128 块在 170 个 SM 上，实测 305–342 µs，而 230 TF 下限是 263 µs；Spark 的 stream-K 能拿到这部分，但它既不支持 wgrad 布局也不是 graph 安全的（§2）。

### 4.2 带 epilogue 的候选（fp32 残差流、erf-GELU、bias 梯度）
| model | op | candidate | us | TFLOPS | cos vs fp32 (erf) |
|---|---|---|---|---|---|
| vitl16 | fc1_gelu | torch F.linear + F.gelu (2 kernels) | 345.8 | 175 | 0.999997 |
| vitl16 | fc1_gelu | r5 lt_epilogue erf (Lt BIAS + Triton GELU) | 350.4 | 172 | 0.999997 |
| vitl16 | fc1_gelu | r5 lt_epilogue tanh (Lt GELU_AUX_BIAS, tanh approx) | 299.7 | 202 | 0.999999 |
| vitl16 | fc1_gelu | QuACK gemm_act gelu_tanh_approx (stock) | 273.0 | 221 | 0.999999 |
| vitl16 | fc1_gelu | Spark hgemm v6 bias+gelu(tanh) (h only) | 287.4 | 210 | 0.999999 |
| vitl16 | fc1_gelu | round13 QuACK custom epilogue (erf, z+h) | 275.0 | 220 | 0.999999 |
| vitl16 | fc1_gelu | round13 QuACK plain GEMM (reference, no epilogue) | 262.6 | 230 | - |
| vitl16 | res_proj | torch F.linear + mul + add (3 kernels, autocast sem.) | 126.6 | 119 | 1.000000 |
| vitl16 | res_proj | QuACK gemm_add stock (C+bias, no gamma: lower bound) | 78.0 | 194 | - |
| vitl16 | res_proj | round13 QuACK custom epilogue (bias+gamma+residual, y) | 80.2 | 188 | 1.000000 |
| vitl16 | res_proj | round13 QuACK plain GEMM (reference) | 75.1 | 201 | - |
| vitl16 | res_proj | Spark hgemm v6 bias+residual (bf16 stream; info only) | 75.7 | 200 | - |
| vitl16 | res_fc2 | torch F.linear + mul + add (3 kernels, autocast sem.) | 356.4 | 170 | 1.000000 |
| vitl16 | res_fc2 | QuACK gemm_add stock (C+bias, no gamma: lower bound) | 294.5 | 205 | - |
| vitl16 | res_fc2 | round13 QuACK custom epilogue (bias+gamma+residual, y) | 307.2 | 197 | 1.000000 |
| vitl16 | res_fc2 | round13 QuACK plain GEMM (reference) | 289.6 | 209 | - |
| vitl16 | res_fc2 | Spark hgemm v6 bias+residual (bf16 stream; info only) | 269.0 | 225 | - |
| vitl16 | fc2_dgelu | torch mm + gelu_backward + sum (3 kernels) | 563.9 | 107 | 0.999997 |
| vitl16 | fc2_dgelu | r5 mm + Triton dgelu (2 kernels, no bias grad) | 385.9 | 157 | 0.999997 |
| vitl16 | fc2_dgelu | QuACK gemm_dact gelu_tanh_approx (stock, +postact) | 280.8 | 215 | 0.999999 |
| vitl16 | fc2_dgelu | round13 QuACK custom epilogue (erf, + db1) | 295.2 | 205 | 0.999999 |
| vitl16 | fc2_dgelu | round13 QuACK plain GEMM (reference) | 266.9 | 226 | - |
| vitb16 | fc1_gelu | torch F.linear + F.gelu (2 kernels) | 413.5 | 164 | 0.999997 |
| vitb16 | fc1_gelu | r5 lt_epilogue erf (Lt BIAS + Triton GELU) | 411.6 | 165 | 0.999997 |
| vitb16 | fc1_gelu | r5 lt_epilogue tanh (Lt GELU_AUX_BIAS, tanh approx) | 346.6 | 196 | 0.999998 |
| vitb16 | fc1_gelu | QuACK gemm_act gelu_tanh_approx (stock) | 303.3 | 224 | 0.999999 |
| vitb16 | fc1_gelu | Spark hgemm v6 bias+gelu(tanh) (h only) | 333.7 | 204 | 0.999998 |
| vitb16 | fc1_gelu | round13 QuACK custom epilogue (erf, z+h) | 312.4 | 218 | 0.999999 |
| vitb16 | fc1_gelu | round13 QuACK plain GEMM (reference, no epilogue) | 305.9 | 222 | - |
| vitb16 | res_proj | torch F.linear + mul + add (3 kernels, autocast sem.) | 161.6 | 105 | 1.000000 |
| vitb16 | res_proj | QuACK gemm_add stock (C+bias, no gamma: lower bound) | 84.8 | 200 | - |
| vitb16 | res_proj | round13 QuACK custom epilogue (bias+gamma+residual, y) | 97.0 | 175 | 1.000000 |
| vitb16 | res_proj | round13 QuACK plain GEMM (reference) | 75.0 | 227 | - |
| vitb16 | res_proj | Spark hgemm v6 bias+residual (bf16 stream; info only) | 86.2 | 197 | - |
| vitb16 | res_fc2 | torch F.linear + mul + add (3 kernels, autocast sem.) | 392.6 | 173 | 1.000000 |
| vitb16 | res_fc2 | QuACK gemm_add stock (C+bias, no gamma: lower bound) | 300.6 | 226 | - |
| vitb16 | res_fc2 | round13 QuACK custom epilogue (bias+gamma+residual, y) | 313.1 | 217 | 1.000000 |
| vitb16 | res_fc2 | round13 QuACK plain GEMM (reference) | 291.9 | 233 | - |
| vitb16 | res_fc2 | Spark hgemm v6 bias+residual (bf16 stream; info only) | 325.7 | 209 | - |
| vitb16 | fc2_dgelu | torch mm + gelu_backward + sum (3 kernels) | 741.1 | 92 | 0.999997 |
| vitb16 | fc2_dgelu | r5 mm + Triton dgelu (2 kernels, no bias grad) | 460.1 | 148 | 0.999997 |
| vitb16 | fc2_dgelu | QuACK gemm_dact gelu_tanh_approx (stock, +postact) | 323.6 | 210 | 0.999999 |
| vitb16 | fc2_dgelu | round13 QuACK custom epilogue (erf, + db1) | 327.3 | 208 | 0.999999 |
| vitb16 | fc2_dgelu | round13 QuACK plain GEMM (reference) | 293.5 | 232 | - |

要点：torch 的 `linear + mul + add` 3 个 kernel 比自定义 epilogue GEMM 慢 58%（proj）/ 16%（fc2）；fc2 dgrad + GELU' 的 torch 3-kernel 路径是 564 µs，自定义 epilogue 295 µs（ViT-L）。erf 版 epilogue 比 stock tanh 版 `gemm_dact` 慢 14 µs，换来与源码 `nn.GELU()` 完全相同的语义，并多出 bias 梯度规约。

### 4.3 LayerNorm 候选（fp32 x 进 bf16 出；前反向；含残差/γ）
形状 ViT-L M=7203 D=1024、ViT-B M=14406 D=768（M=7903/15806 见 `profile/ln_matrix.json`）。热循环（L2 常驻），所以 GB/s 不是 DRAM 带宽。
| shape | op | candidate | us | cos vs fp32 |
|---|---|---|---|---|
| vitl16 M=7203 | fwd | torch(layer_norm+cast) | 22.9 | 0.9999998 |
| vitl16 M=7203 | fwd | r5_triton | 8.4 | 0.9999998 |
| vitl16 M=7203 | fwd | quack_layernorm_fwd(bf16 out) | 9.9 | 0.9999998 |
| vitl16 M=7203 | fwd | round13_cuda | 10.3 | 0.9999998 |
| vitl16 M=7203 | bwd_ln | torch(autograd composite) | 154.4 | 1.0000001 |
| vitl16 M=7203 | bwd_ln | r5_triton(bwd; branch grad fp32 [M,D]) | 68.3 | 1.0000001 |
| vitl16 M=7203 | bwd_ln | round13_cuda | 34.6 | 1.0000001 |
| vitl16 M=7203 | bwd_ln | quack_layernorm_bwd(core only, no dres/gamma) | 72.2 | 1.0000001 |
| vitl16 M=7203 | bwd_ln_gam | torch(autograd composite) | 297.6 | 1.0000001 |
| vitl16 M=7203 | bwd_ln_gam | r5_triton(bwd; branch grad fp32 [M,D]) | 87.3 | 1.0000001 |
| vitl16 M=7203 | bwd_ln_gam | round13_cuda | 69.5 | 1.0000001 |
| vitl16 M=7203 | bwd_gam | torch(mul+cast+sums) | 117.7 | - |
| vitl16 M=7203 | bwd_gam | round13_cuda | 14.4 | - |
| vitl16 M=7203 | colsum_qkv | torch(float().sum(0)) | 111.4 | - |
| vitl16 M=7203 | colsum_qkv | torch(sum(0,dtype=fp32)) | 16.9 | - |
| vitl16 M=7203 | colsum_qkv | round13_cuda | 17.7 | - |
| vitb16 M=14406 | fwd | torch(layer_norm+cast) | 48.8 | 1.0000001 |
| vitb16 M=14406 | fwd | r5_triton | 12.9 | 1.0000001 |
| vitb16 M=14406 | fwd | quack_layernorm_fwd(bf16 out) | 12.1 | 1.0000001 |
| vitb16 M=14406 | fwd | round13_cuda | 12.3 | 1.0000001 |
| vitb16 M=14406 | bwd_ln | torch(autograd composite) | 316.9 | 1.0000001 |
| vitb16 M=14406 | bwd_ln | r5_triton(bwd; branch grad fp32 [M,D]) | 104.8 | 1.0000001 |
| vitb16 M=14406 | bwd_ln | round13_cuda | 82.3 | 1.0000001 |
| vitb16 M=14406 | bwd_ln | quack_layernorm_bwd(core only, no dres/gamma) | 126.2 | 1.0000000 |
| vitb16 M=14406 | bwd_ln_gam | torch(autograd composite) | 527.4 | 1.0000001 |
| vitb16 M=14406 | bwd_ln_gam | r5_triton(bwd; branch grad fp32 [M,D]) | 145.6 | 1.0000001 |
| vitb16 M=14406 | bwd_ln_gam | round13_cuda | 122.9 | 1.0000001 |
| vitb16 M=14406 | bwd_gam | torch(mul+cast+sums) | 236.2 | - |
| vitb16 M=14406 | bwd_gam | round13_cuda | 17.2 | - |
| vitb16 M=14406 | colsum_qkv | torch(float().sum(0)) | 191.5 | - |
| vitb16 M=14406 | colsum_qkv | torch(sum(0,dtype=fp32)) | 21.2 | - |
| vitb16 M=14406 | colsum_qkv | round13_cuda | 18.3 | - |

选择：前向 4 个融合候选都在 8–10 µs（torch 两个 kernel 23 µs），选自研 CUDA（与反向共用统计量约定）；反向 `rowbwd` 比 r5 Triton 和 QuACK `layernorm_bwd` 快 2×（且多做了残差梯度加、`γ⊙dX`、dγ/dbias 部分和），LN2-反向 + proj 预处理（K=4）69.5 µs，r5 为 87 µs 且不含 bf16 `g`、dγ。

### 4.4 单块 fwd/bwd（CUDA graph 回放，同一 attention）
| block (real shapes) | variant | fwd ms | bwd ms | fwd+bwd ms | vs T-BF16 block (r5 ln_fuse) |
|---|---|---|---|---|---|
| vitl16_plain | autocast_default | 1.415 | 2.992 | 4.408 | 0.958x |
| vitl16_plain | r5_ln_fuse | 1.379 | 2.844 | 4.223 | 1.000x |
| vitl16_plain | r13_full | 1.135 | 2.515 | 3.650 | 1.157x |
| vitl16_plain | r13_wgrad_quack | 1.150 | 2.573 | 3.723 | 1.134x |
| vitl16_plain | r13_torch_gemm | 2.047 | 3.321 | 5.368 | 0.787x |
| vitl16_joint | autocast_default | 1.572 | 3.435 | 5.007 | 0.956x |
| vitl16_joint | r5_ln_fuse | 1.525 | 3.262 | 4.787 | 1.000x |
| vitl16_joint | r13_full | 1.331 | 2.896 | 4.227 | 1.132x |
| vitl16_joint | r13_wgrad_quack | 1.331 | 2.984 | 4.315 | 1.109x |
| vitl16_joint | r13_torch_gemm | 2.264 | 3.800 | 6.063 | 0.789x |
| vitb16_plain | autocast_default | 1.687 | 3.869 | 5.556 | 0.924x |
| vitb16_plain | r5_ln_fuse | 1.613 | 3.519 | 5.133 | 1.000x |
| vitb16_plain | r13_full | 1.352 | 3.010 | 4.362 | 1.177x |
| vitb16_plain | r13_wgrad_quack | 1.351 | 3.188 | 4.539 | 1.131x |
| vitb16_plain | r13_torch_gemm | 2.627 | 4.252 | 6.879 | 0.746x |
| vitb16_joint | autocast_default | 2.008 | 4.659 | 6.667 | 0.927x |
| vitb16_joint | r5_ln_fuse | 1.921 | 4.257 | 6.178 | 1.000x |
| vitb16_joint | r13_full | 1.619 | 3.658 | 5.277 | 1.171x |
| vitb16_joint | r13_wgrad_quack | 1.618 | 3.850 | 5.468 | 1.130x |
| vitb16_joint | r13_torch_gemm | 3.056 | 5.051 | 8.107 | 0.762x |

### 4.5 整步（两个模型，三相加权，T-BF16 对比）
计时：`common/bench.time_steps`，预热 20，测 50 × 3，early/mid/late 加权 0.2/0.6/0.2；`AC` 和 `r11_autocast_r13_full` 是不带 r1/r3/r5/r7 等其它优化的最小 T-BF16（只用于看 r13 单独的增益，与 TB1 不可直接比）。

**vitl16** (batch 7; phases early/mid/late, weights 0.2/0.6/0.2)

| stack | early ms | mid ms | late ms | weighted img/s | vs TB1 | fwd / bwd / opt / loss (mid, ms) | peak mem GB |
|---|---|---|---|---|---|---|---|
| TB1 | 172.6 | 171.8 | 170.2 | 40.79 | 1.000x | 48.4 / 106.3 / 12.2 / 3.8 | 17.5 |
| TB1_R13 | 158.5 | 158.0 | 156.8 | 44.35 | 1.087x | 43.2 / 97.9 / 12.2 / 3.7 | 17.5 |
| TB1_R13_wgq | 160.2 | 159.5 | 158.2 | 43.92 | 1.077x | 43.2 / 99.5 / 12.2 / 3.7 | 17.5 |
| TB1_R13_torch | 199.3 | 198.3 | 197.1 | 35.30 | 0.865x | 64.2 / 117.3 / 12.3 / 3.8 | 17.5 |
| AC | 240.8 | 241.4 | 241.4 | 29.01 | 0.711x | 67.3 / 128.4 / 21.5 / 21.8 | 19.5 |
| r11_autocast_r13_full | 226.8 | 222.3 | 222.6 | 31.35 | 0.769x | 61.4 / 116.3 / 21.5 / 21.7 | 19.5 |

**vitb16** (batch 14; phases early/mid/late, weights 0.2/0.6/0.2)

| stack | early ms | mid ms | late ms | weighted img/s | vs TB1 | fwd / bwd / opt / loss (mid, ms) | peak mem GB |
|---|---|---|---|---|---|---|---|
| TB1 | 147.7 | 146.1 | 143.8 | 95.90 | 1.000x | 40.3 / 92.6 / 6.4 / 5.9 | 15.3 |
| TB1_R13 | 137.0 | 135.8 | 133.7 | 103.20 | 1.076x | 36.7 / 86.6 / 6.1 / 5.9 | 15.3 |

### 4.6 MXFP8 探针（只是数据点，BF16 块不使用）
QuACK 0.6.5 block-scaled GEMM（E4M3 + UE8M0，32 元素一组，操作数已量化，不含量化时间），ViT-L 形状；wgrad 的收缩维 M=7203 不是 32 的倍数，探针里把 M 零填充到 128 的倍数。
| GEMM (ViT-L, M=7203) | bf16 QuACK us | MXFP8 QuACK us | MXFP8 TFLOPS | speedup | rel L2 err vs fp32 |
|---|---|---|---|---|---|
| qkv fwd | 193.8 | 75.0 | 605 | 2.58x | 0.0375 |
| qkv dgrad | 223.1 | 78.3 | 579 | 2.85x | 0.0375 |
| qkv wgrad | 215.2 | 107.6 | 421 | 2.00x | 0.0377 |
| proj fwd | 75.1 | 28.9 | 522 | 2.59x | 0.0375 |
| proj dgrad | 74.3 | 26.4 | 572 | 2.82x | 0.0375 |
| proj wgrad | 87.0 | 46.1 | 327 | 1.89x | 0.0377 |
| fc1 fwd | 277.1 | 110.9 | 545 | 2.50x | 0.0375 |
| fc1 dgrad | 300.5 | 98.3 | 615 | 3.06x | 0.0375 |
| fc1 wgrad | 344.0 | 110.2 | 548 | 3.12x | 0.0377 |
| fc2 fwd | 289.7 | 93.4 | 647 | 3.10x | 0.0375 |
| fc2 dgrad | 266.8 | 96.0 | 629 | 2.78x | 0.0375 |
| fc2 wgrad | 343.9 | 110.6 | 546 | 3.11x | 0.0377 |

## 5. 正确性

### Gate A（算子对 fp32 参考，含尖峰残差、尾块、joint mask）
- elementwise kernels (`gateA_kernels.json`): 20 cases (D=1024/768 x M=7203/7903/14406/15806/129 x plain/spiky), 360 checks, 0 failures; worst cos of any fp32 output 0.999997616; bitwise deterministic.
- fused block vs fp32 block (real DINOv3 `SelfAttentionBlock`, LayerScale, k-masked qkv bias, spiky residual, joint mask): error of every tensor (out, dx, all 12 Parameter gradients) vs the fp32 block, fused vs torch-autocast (same-precision reference):

| case | worst fused-vs-fp32 cos | worst autocast-vs-fp32 cos | worst fused-vs-autocast cos | max rel L2 fused / autocast |
|---|---|---|---|---|
| vitl16_plain_normal | 0.999969 | 0.999969 | 0.999987 | 7.90e-03 / 7.86e-03 |
| vitl16_plain_spiky | 0.999993 | 0.999990 | 0.999992 | 4.26e-03 / 4.56e-03 |
| vitl16_joint_normal | 0.999972 | 0.999970 | 0.999987 | 7.54e-03 / 7.70e-03 |
| vitl16_joint_spiky | 0.999993 | 0.999990 | 0.999990 | 4.23e-03 / 7.03e-03 |
| vitb16_plain_normal | 0.999976 | 0.999977 | 0.999988 | 6.89e-03 / 6.80e-03 |
| vitb16_plain_spiky | 0.999993 | 0.999988 | 0.999990 | 3.87e-03 / 4.98e-03 |
| vitb16_joint_normal | 0.999978 | 0.999979 | 0.999990 | 6.61e-03 / 6.49e-03 |
| vitb16_joint_spiky | 0.999992 | 0.999990 | 0.999991 | 4.88e-03 / 1.56e-02 |

### Gate B（单步，3 个 seed；C2 在 2026-10-03 用 `round17_mxfp8_sm120/tests/gate_b_multi.py` 重跑）
同一进程内用同一权重、同一 batch、同一 RngTape（R0 记录，其余回放；`assert_complete` 全部通过）依次跑 R0（fp32）、TB1、TB1+R13 各一步，逐参数比较梯度。准则是 Amendment 1 §9.4：全局指标 loss_rel ≤1e-2、全局 grad 余弦 ≥0.99、grad_norm_rel ≤0.02、update 余弦 ≥0.99（对 TB1）；逐参数 `cos(cand,R0) >= min(0.99, cos(TB1,R0) − 0.02)`。

**vitl16 TB1_R13** (same); per seed, vs R0 (fp32) and vs the bf16 reference

| seed | loss_rel vs R0 | grad cos (global) vs R0 | grad_norm_rel vs R0 | loss_rel vs ref | grad cos (global) vs ref | grad_norm_rel vs ref | update cos (global) vs ref | per-param cos min vs ref | per-param offenders (calibrated) | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.4e-05 | 0.99781 | 0.0057 | 1.2e-04 | 0.99998 | 0.0004 | 1.00000 | 0.8607 | 0 | True |
| 1 | 3.3e-05 | 0.99997 | 0.0000 | 2.1e-05 | 0.99973 | 0.0007 | 1.00000 | 0.8224 | 0 | True |
| 2 | 4.1e-05 | 0.99972 | 0.0015 | 2.1e-04 | 0.99986 | 0.0012 | 1.00000 | 0.8290 | 0 | True |

TB1 vs R0 (bf16 self-noise): 0: loss_rel 4.2e-05, gcos 0.99781, pmin 0.891; 1: loss_rel 5.4e-05, gcos 0.99973, pmin 0.919; 2: loss_rel 1.7e-04, gcos 0.99974, pmin 0.801

**vitb16 TB1_R13** (same); per seed, vs R0 (fp32) and vs the bf16 reference

| seed | loss_rel vs R0 | grad cos (global) vs R0 | grad_norm_rel vs R0 | loss_rel vs ref | grad cos (global) vs ref | grad_norm_rel vs ref | update cos (global) vs ref | per-param cos min vs ref | per-param offenders (calibrated) | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.1e-05 | 0.99867 | 0.0147 | 5.6e-05 | 0.99998 | 0.0004 | 1.00000 | 0.9974 | cls_token: cos(cand,R0) 0.836 vs cos(TB1,R0) 0.867, cos(cand,TB1) 0.9976 | False |
| 1 | 2.4e-04 | 0.99930 | 0.0007 | 2.8e-05 | 0.99941 | 0.0028 | 1.00000 | 0.9846 | 0 | True |
| 2 | 4.8e-05 | 0.99937 | 0.0008 | 1.1e-04 | 0.99982 | 0.0006 | 1.00000 | 0.9892 | blocks.0.norm1.weight: cos(cand,R0) 0.493 vs cos(TB1,R0) 0.519, cos(cand,TB1) 0.9979 | False |

TB1 vs R0 (bf16 self-noise): 0: loss_rel 4.5e-05, gcos 0.99866, pmin 0.733; 1: loss_rel 2.1e-04, gcos 0.99878, pmin 0.919; 2: loss_rel 1.6e-04, gcos 0.99929, pmin 0.519

结果：全局指标 6/6 全过（对 TB1：loss_rel ≤2.1e-4，全局 grad 余弦 ≥0.9994，grad_norm_rel ≤0.003，update 余弦 1.0000）。逐参数校准准则：ViT-L 3/3 通过；ViT-B 1/3 通过，另两个 seed 各有 1 个“越线”参数：seed 0 的 `cls_token`（cand/R0 = 0.836，TB1/R0 = 0.867）和 seed 2 的 `blocks.0.norm1.weight`（0.493 对 0.519），差值 0.031 和 0.027，略大于 0.02 的边距。这两个参数本身对 bf16 噪声极敏感（TB1 自己对 R0 的余弦只有 0.867 / 0.519），而候选与 TB1 的余弦是 0.9976 / 0.9979，也就是候选落在 TB1 自己的噪声邻域里，不是回归。几何上这是准则的问题：当 TB1 对 R0 的夹角已是 30°–60° 时，一个与 TB1 夹角仅 4° 的点，对 R0 的余弦天然可以差 0.03。建议协调者把校准准则改成 `cos(cand,R0) >= min(0.99, cos(TB1,R0) − 0.02)` **或** `cos(cand,TB1) >= 0.99`（后者对任何同精度换 kernel 都是更直接的“没有比 TB1 噪声更大”的判据）。按改后的准则 6/6 全过。

**校准说明**：`common/check.py` 的逐参数 `grad_cos_min >= 0.99` 对两个 bf16 栈互比不可达：layer-0 的 LayerNorm 权重/偏置、qkv 权重、`storage_tokens` 的梯度对 bf16 舍入噪声极其敏感（梯度经 24 层放大）。`tests/calibrate_gateb.py` 用同一权重/批/随机数带跑 R0（fp32）、TB1、候选三份，逐参数比较：如果 TB1 自己对 fp32 的余弦就低于 0.99，则候选只需不比 TB1 更差。校准准则：`cos(cand, R0) >= min(0.99, cos(TB1, R0) − 0.02)`。全局指标（loss_rel、全局 grad 余弦、grad_norm_rel、update 余弦）沿用 check.py 阈值。需要协调者批准把校准准则作为同精度换 kernel 的 B 门槛（§7）。

### Gate S（50 步轨迹，2 个真实 batch；C2 重跑，`tests/gate_s_multi.py`）
R0、TB1、TB1+R13 各跑 50 步（真实掩码调度，以及固定 mid / late 相位），`gate_c(interval=10, max_rel=0.05)`，全部有限、无 NaN/Inf。

**vitl16**: 50 steps, 2 batches, gate_c(interval=10, max_rel=0.05); max relative EMA-loss deviation over checkpoints 10..50, grad-norm ratio range

| pair | schedule | fixed mid | fixed late |
|---|---|---|---|
| TB1_R13 vs R0 | 0.0006 / [1.000,1.003] / PASS | 0.0002 / [1.000,1.001] / PASS | 0.0006 / [1.000,1.004] / PASS |
| TB1_R13 vs TB1 | 0.0002 / [0.999,1.000] / PASS | 0.0002 / [0.999,1.000] / PASS | 0.0002 / [1.000,1.001] / PASS |
| TB1 vs R0 | 0.0007 / [1.000,1.003] / PASS | 0.0003 / [1.000,1.002] / PASS | 0.0007 / [1.000,1.004] / PASS |

**vitb16**: 50 steps, 2 batches, gate_c(interval=10, max_rel=0.05); max relative EMA-loss deviation over checkpoints 10..50, grad-norm ratio range

| pair | schedule | fixed mid | fixed late |
|---|---|---|---|
| TB1_R13 vs R0 | 0.0002 / [0.999,1.000] / PASS | 0.0006 / [0.989,1.000] / PASS | 0.0001 / [1.000,1.000] / PASS |
| TB1_R13 vs TB1 | 0.0005 / [0.997,1.001] / PASS | 0.0004 / [0.994,1.001] / PASS | 0.0002 / [0.990,1.000] / PASS |
| TB1 vs R0 | 0.0006 / [0.998,1.003] / PASS | 0.0002 / [0.995,1.000] / PASS | 0.0002 / [0.999,1.009] / PASS |

结果：TB1+R13 对 R0 和对 TB1 的 EMA 损失最大相对偏差 ≤6e-4（门槛 5e-2），梯度范数比 [0.989, 1.004]，6 个对比 × 3 个相位全部 PASS。TB1 自己对 R0 的偏差是 2e-4–7e-4，同一量级，说明 R13 没有增加 bf16 噪声。

## 6. 静态分析与 roofline

`common/kernel_static.sh` 依赖 ripgrep（新容器没有），且只给整个二进制的汇总，所以用 `tests/static_analysis.py`（`cuobjdump --dump-resource-usage` + `-sass` 分函数计数）；ncu 不可用。
| kernel | regs | spill/stack B | instr | LDG | STG | BAR | SHFL | FFMA | MUFU |
|---|---|---|---|---|---|---|---|---|---|
| rowbwd D=1024 GAM | 48 | 0 | 328 | 6 | 16 | 4 | 0 | 8 | 1 |
| rowbwd D=1024 LN | 80 | 0 | 496 | 10 | 16 | 5 | 20 | 40 | 1 |
| rowbwd D=1024 LN GAM | 119 | 0 | 736 | 14 | 32 | 9 | 20 | 48 | 1 |
| rowbwd D=768 GAM | 56 | 0 | 328 | 6 | 16 | 4 | 0 | 8 | 1 |
| rowbwd D=768 LN | 80 | 0 | 512 | 10 | 16 | 5 | 20 | 40 | 1 |
| rowbwd D=768 LN GAM | 117 | 0 | 752 | 14 | 32 | 9 | 20 | 48 | 1 |
| rowbwd D=384 GAM | 45 | 0 | 464 | 11 | 19 | 4 | 0 | 20 | 2 |
| rowbwd D=384 LN | 56 | 0 | 456 | 6 | 15 | 5 | 20 | 20 | 1 |
| rowbwd D=384 LN GAM | 64 | 0 | 640 | 8 | 30 | 9 | 20 | 24 | 1 |
| ln_fwd D=1024 | 60 | 0 | 296 | 24 | 10 | 0 | 10 | 49 | 1 |
| ln_fwd D=768 | 58 | 0 | 240 | 18 | 8 | 0 | 10 | 37 | 1 |
| ln_fwd D=384 | 42 | 0 | 152 | 9 | 5 | 0 | 10 | 19 | 1 |
| finalize_kernelPKfPfii | 40 | 0 | 448 | 57 | 1 | 0 | 0 | 0 | 0 |
| colsum_bf16_kernelPK13__ | 24 | 0 | 80 | 1 | 8 | 0 | 0 | 0 | 0 |

要点：K=4 的 `rowbwd`（LN2 反向 + proj 预处理）119 个寄存器、无溢出（第一版“一个 warp 一行、32 列累加器放寄存器”的写法 255 个寄存器并溢出、慢 2.6×，改成“一行由 D/256 个 warp 协作 + 每线程只保留 K×8 个列累加器 + 具名 barrier 交换行和”后无溢出）；这些 kernel 是纯访存/FMA kernel，没有 HMMA/QMMA。GEMM 由 QuACK（CuTe DSL）生成，kernel 在运行时 JIT，未做 SASS 计数；用实测 TFLOPS 对 230 TF roofline 评价。

**Roofline（块级）**：ViT-L plain 块 fwd+bwd 的 GEMM FLOP = 3 × 2 × 7203 × 1024² × 12 = 543.8 GFLOP，230 TF 下 2.36 ms；实测整块（含 attention、elementwise）3.65 ms。(pending)

## 7. 结论与下一瓶颈

**结论**
- 整步（三相加权）ViT-L B=7：40.79 → 44.35 img/s（1.087×，mid 171.8 → 158.0 ms），ViT-B B=14：95.90 → 103.20 img/s（1.076×，146.1 → 135.8 ms）；单块 fwd+bwd（graph 回放）1.157× / 1.177×。峰值显存不变（17.5 / 15.3 GB）。
- 正确性：Gate A 全过（kernel 与块对 fp32 的余弦 ≥0.99997 / 0.99997，与 autocast 同量级）；Gate B 全局指标 6/6 全过，逐参数校准准则 ViT-L 3/3、ViT-B 1/3，其余 2 个 seed 各有 1 个对 bf16 噪声极敏感的参数以 0.027–0.031 的差值略越过 0.02 边距，而候选与 TB1 的余弦是 0.998（准则问题，见 §5，建议加上“cos(cand,TB1) ≥ 0.99 即可”的备选条件）；Gate S 6 个对比 × 3 相位全部 PASS（max_rel ≤6e-4）。
- GEMM 选择 / 融合设计 / 候选取舍见 §2、§4，Spark hgemm 因 stream-K 的 host epoch 不可 graph 回放、按需 cudaMalloc 而不采用。

**下一瓶颈**
1. 骨干 GEMM 仍占 ViT-L 步的 43%（~64 ms），已在 bf16 限速上；下一步是 MXFP8（round17）：探针里 12 个 GEMM 比 bf16 快 1.9–3.1×。
2. head（4 次 predict，ViT-L 22 ms、ViT-B 30 ms）、attention（16 ms）、优化器（11.6 ms）和 ~110 个 >20 µs 的间隙是其余的主要部分，分别属于 r14、r12、r15/r16。
3. 块内剩余的行 kernel（~8.5 ms/步）和 5 个 elementwise kernel 已经很小；继续融合的收益 <2 ms/步。
