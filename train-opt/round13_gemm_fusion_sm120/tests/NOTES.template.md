# Round13：BF16 GEMM 融合整块（sm_120）（负责：Agent C）

> 本文件由 `tests/assemble_notes.py` 从 `tests/NOTES.template.md` 和 `profile/` 里的实测 JSON/CSV 生成，数字全部实测，缺项写 pending/null。
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

**3.1 单块（ViT-L plain，同一 flash attention + triton RoPE）**

{{NSYS_BLOCK}}

**3.2 round13 块的 kernel 清单**
{{NSYS_BLOCK_KERNELS}}

**3.3 整步（TB1 vs TB1+R13）**
{{NSYS_STEP}}

**3.4 分析结论**

{{NSYS_CONCLUSIONS}}

## 4. 实测结果

### 4.1 GEMM 候选矩阵（12 个 GEMM × 2 模型 × plain/joint M）

形状：ViT-L d=1024，M=7203（plain）/ 7903（joint）；ViT-B d=768，M=14406 / 15806；qkv N=3d,K=d；proj d→d；fc1 d→4d；fc2 4d→d。
布局：fwd `X·Wᵀ`（B 为 K-major 视图）、dgrad `dY·W`、wgrad `dYᵀ·X`（fp32 输出，所有候选统一）。计时：CUDA graph 5 次调用，3 轮交替 × 30 次取中位数，同进程同锁，roofline = 230 TFLOPS。
{{GEMM_TABLES}}

读表要点：fwd 上 QuACK 的 223–234 TF 已在 roofline 上；dgrad 略低（207–233）；wgrad 最差（cuBLAS 196–209，QuACK 172–212）。ViT-L 的 fc1/fc2 wgrad 是 4096×1024 输出 = 256 个 128×128 块在 170 个 SM 上，实测 305–342 µs，而 230 TF 下限是 263 µs；Spark 的 stream-K 能拿到这部分，但它既不支持 wgrad 布局也不是 graph 安全的（§2）。

### 4.2 带 epilogue 的候选（fp32 残差流、erf-GELU、bias 梯度）
{{EPILOGUE_TABLE}}

要点：torch 的 `linear + mul + add` 3 个 kernel 比自定义 epilogue GEMM 慢 58%（proj）/ 16%（fc2）；fc2 dgrad + GELU' 的 torch 3-kernel 路径是 564 µs，自定义 epilogue 295 µs（ViT-L）。erf 版 epilogue 比 stock tanh 版 `gemm_dact` 慢 14 µs，换来与源码 `nn.GELU()` 完全相同的语义，并多出 bias 梯度规约。

### 4.3 LayerNorm 候选（fp32 x 进 bf16 出；前反向；含残差/γ）
形状 ViT-L M=7203 D=1024、ViT-B M=14406 D=768（M=7903/15806 见 `profile/ln_matrix.json`）。热循环（L2 常驻），所以 GB/s 不是 DRAM 带宽。
{{LN_TABLE}}

选择：前向 4 个融合候选都在 8–10 µs（torch 两个 kernel 23 µs），选自研 CUDA（与反向共用统计量约定）；反向 `rowbwd` 比 r5 Triton 和 QuACK `layernorm_bwd` 快 2×（且多做了残差梯度加、`γ⊙dX`、dγ/dbias 部分和），LN2-反向 + proj 预处理（K=4）69.5 µs，r5 为 87 µs 且不含 bf16 `g`、dγ。

### 4.4 单块 fwd/bwd（CUDA graph 回放，同一 attention）
{{BLOCK_TABLE}}

### 4.5 整步（两个模型，三相加权，T-BF16 对比）
计时：`common/bench.time_steps`，预热 20，测 50 × 3，early/mid/late 加权 0.2/0.6/0.2；`AC` 和 `r11_autocast_r13_full` 是不带 r1/r3/r5/r7 等其它优化的最小 T-BF16（只用于看 r13 单独的增益，与 TB1 不可直接比）。
{{STEP_TABLE}}

### 4.6 MXFP8 探针（只是数据点，BF16 块不使用）
QuACK 0.6.5 block-scaled GEMM（E4M3 + UE8M0，32 元素一组，操作数已量化，不含量化时间），ViT-L 形状；wgrad 的收缩维 M=7203 不是 32 的倍数，探针里把 M 零填充到 128 的倍数。
{{MXFP8_TABLE}}

## 5. 正确性

### Gate A（算子对 fp32 参考，含尖峰残差、尾块、joint mask）
{{GATE_A}}

### Gate B（单步，3 个 seed，对 TB1）
{{GATE_B}}

**校准说明**：`common/check.py` 的逐参数 `grad_cos_min >= 0.99` 对两个 bf16 栈互比不可达：layer-0 的 LayerNorm 权重/偏置、qkv 权重、`storage_tokens` 的梯度对 bf16 舍入噪声极其敏感（梯度经 24 层放大）。`tests/calibrate_gateb.py` 用同一权重/批/随机数带跑 R0（fp32）、TB1、候选三份，逐参数比较：如果 TB1 自己对 fp32 的余弦就低于 0.99，则候选只需不比 TB1 更差。校准准则：`cos(cand, R0) >= min(0.99, cos(TB1, R0) − 0.02)`。全局指标（loss_rel、全局 grad 余弦、grad_norm_rel、update 余弦）沿用 check.py 阈值。需要协调者批准把校准准则作为同精度换 kernel 的 B 门槛（§7）。

### Gate S（50 步轨迹，2 个真实 batch）
{{GATE_S}}

## 6. 静态分析与 roofline

`common/kernel_static.sh` 依赖 ripgrep（新容器没有），且只给整个二进制的汇总，所以用 `tests/static_analysis.py`（`cuobjdump --dump-resource-usage` + `-sass` 分函数计数）；ncu 不可用。
{{STATIC_TABLE}}

要点：K=4 的 `rowbwd`（LN2 反向 + proj 预处理）119 个寄存器、无溢出（第一版“一个 warp 一行、32 列累加器放寄存器”的写法 255 个寄存器并溢出、慢 2.6×，改成“一行由 D/256 个 warp 协作 + 每线程只保留 K×8 个列累加器 + 具名 barrier 交换行和”后无溢出）；这些 kernel 是纯访存/FMA kernel，没有 HMMA/QMMA。GEMM 由 QuACK（CuTe DSL）生成，kernel 在运行时 JIT，未做 SASS 计数；用实测 TFLOPS 对 230 TF roofline 评价。

**Roofline（块级）**：ViT-L plain 块 fwd+bwd 的 GEMM FLOP = 3 × 2 × 7203 × 1024² × 12 = 543.8 GFLOP，230 TF 下 2.36 ms；实测整块（含 attention、elementwise）3.65 ms。{{ROOFLINE_NOTE}}

## 7. 结论与下一瓶颈

{{CONCLUSION}}
