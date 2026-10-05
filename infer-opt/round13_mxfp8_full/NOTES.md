# Round 13：整模型 MXFP8（E4M3 + UE8M0 1×32）

> 日期 2026-10-05。用户决定：**精度门全部关闭**（checkpoint 是 bf16 训练的，不期望低精度过门），本轮只量速度。不跑门 M/H、不做精度扫描；
> 只记录 op 级 sanity（对 fp32 乘积、量化器逐位）和 speed12.py 打印的 mask-logit cos。所有数字都是本机实测，GPU 任务持双锁。
> R0–R12 没有改动，只读 import。本轮目录 `round13_mxfp8_full/`，临时文件 `/root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full/`。

## 0. 结论（最终表，`speed12.py`，CUDA graph 回放，3 轮交替取中位数；`results/speed_final.json`、`logs/speed_vit{l,b}_final.log`）

| 模型 | variant | model_ms | frame_ms | fps_gpu | fps_video | finite | mask-logit cos vs r5_v1 |
|---|---|---:|---:|---:|---:|---|---:|
| ViT-L | r5_v1 | 9.711 | 9.886 | 101.2 | 97.3 | True | 1.000000 |
| ViT-L | r12+head16 | 7.077 | 7.479 | 133.7 | 125.5 | True | 0.999996 |
| ViT-L | r13_mx8a（第一步：cuBLASLt + 独立 epilogue/量化 kernel） | 4.976 | 5.145 | 194.3 | 182.8 | True | 0.999820 |
| ViT-L | r13_mx8（第二步：自写 mgx 全部四类 + ConvT） | 4.946 | 5.035 | 198.6 | 181.9 | True | 0.999888 |
| ViT-L | **r13_mx8best**（逐类取优：qkv 走 cuBLASLt，其余自写） | **4.904**（1.98× vs r5_v1，1.44× vs r12+head16） | **4.983** | **200.7** | **183.4** | True | 0.999892 |
| ViT-B | r5_v1 | 3.680 | 3.844 | 260.2 | 226.2 | True | 1.000000 |
| ViT-B | r12 | 2.715 | 2.791 | 358.3 | 272.6 | True | 0.999992 |
| ViT-B | r13_mx8a | 2.089 | 2.173 | 460.1 | 279.7 | True | 0.999556 |
| ViT-B | r13_mx8 | 1.948 | 2.019 | 495.3 | 285.0 | True | 0.999693 |
| ViT-B | **r13_mx8best** | **1.939**（1.90× vs r5_v1，1.40× vs r12） | **2.010** | **497.6** | 272.4（host / 解码受限，噪声 ±5%） | True | 0.999505 |

读法与可信度：
- **同一变体在不同时刻的 model_ms 会漂 ±3%**（另一个 agent 在同一块 GPU 上跑时 clock / 带宽不同，`other gpu procs` 非空时更明显）。只有同一次运行里交替测的数才能互相比较。本轮 r13_mx8 / mx8best / mx8a 三者之间 ViT-L 上只差 1–2%（最终表），在别的运行里 mx8best 比 mx8 快过 3.9%（4.711 vs 4.903 ms，`logs/speed_vitl_s2d.log`）、比 mx8a 快 4.6%（`s2e`）；ViT-B 上自写比 cuBLASLt 版稳定快 7%。结论是"自写 ≥ cuBLASLt 版，差距在 1–7% 量级"，不是倍数。
- ViT-B 的 fps_video 被 host / cv2 解码限制（模型 1.94 ms，单帧实际 ≈3.6 ms）；比较模型速度看 model_ms / frame_ms / fps_gpu。
- mask-logit cos 只记录，不是门（MXFP8 的量化误差本身 rel L2 ≈ 2.7e-2 / GEMM，整模型 cos 0.9995–0.9999）。

最快配置（每个 GEMM 类）：

| GEMM | 实现 | 说明 |
|---|---|---|
| LN → 量化 | `ln_mx8`（本轮 `mx13.cu`） | fp32 残差 → LN → MX8 e4m3 + cuBLAS 分块 scale，一遍 |
| qkv | cuBLASLt `torch._scaled_mm`（bf16 输出）+ `epi_rope`（bias + RoPE + fp16 `[3,B,N,H,64]` 布局） | ViT-L 上略优于自写；ViT-B 持平 |
| attention | r12 `attn16o.cu`（fp16，原样）+ `quant_rows`（fp16 输出 → MX8） | |
| proj | **自写 mgx**，epilogue = bias + γ + fp32 残差 | |
| fc1 | **自写 mgx**，epilogue = bias + GELU + **MX8 量化输出**（直接写 fc2 的 e4m3 字节和分块 scale） | 省掉 bf16 中间写读和 `gelu_quant` 遍 |
| fc2 | **自写 mgx**，epilogue = bias + γ + fp32 残差 | |
| head ConvT1 / ConvT2 | **自写 mgx**（bf16 输出，`dw_ln` 不变），输入由 `quant_rows` 量化 | |

## 1. 第一步：`r13_mx8a`（cuBLASLt `_scaled_mm` + 独立量化 / epilogue kernel）

`r13_mx8a` = r12 的骨干四类 GEMM（全部层）和 head 的两个 ConvT GEMM 换成 MXFP8，其余（patch-embed GEMM、attention fp16 kernel、dw_ln、tail、post_a2）不动。

每个 block 12 个 kernel（`impl/stack13.py::block8a`）：

| # | kernel | 内容 |
|---|---|---|
| 1 | `ln_mx8` | fp32 残差 → LN → MX8（q + cuBLAS 分块 scale），`impl/mx13.cu`（ln16 的结构 + r17 的 RCEIL 规则） |
| 2 | `_scaled_mm` | qkv GEMM，cuBLASLt block-scaled MXFP8，bf16 输出，无 epilogue |
| 3 | `epi_rope` | bf16 y + bias + RoPE → fp16 `[3,B,N,H,64]`（r12 attention 要的物理布局） |
| 4 | attention | r12 `attn16o.cu`，原样 |
| 5 | `quant_rows` | attention 输出（fp16）→ MX8 |
| 6 | `_scaled_mm` | proj GEMM |
| 7 | `epi_res` | bias + γ + fp32 残差 |
| 8 | `ln_mx8` | LN2 → MX8 |
| 9 | `_scaled_mm` | fc1 GEMM |
| 10 | `gelu_quant` | bf16 y + bias → GELU(erf) → MX8 |
| 11 | `_scaled_mm` | fc2 GEMM |
| 12 | `epi_res` | bias + γ + fp32 残差 |

权重在建模时一次性量化（`MXW`，K-major，scale 沿 K）；激活量化用持久缓冲（`act_buf`，各层共用一个、scale 的 padding 行恒为 0，graph 捕获安全）。head：`quant_rows`(bf16) → `_scaled_mm` → 原 `dw_ln`。
入口：`plug13.py`（`PREFIXES=("r13",)`）→ `impl/stack13.py::make_variant`，返回 `(model_fn, r11_model.Pipeline11)`。

首个整模型数字（第一次 `speed12.py`，写入时的原始记录，`results/speed_s1.json`）：ViT-L r13_mx8a 5.088 ms（r5_v1 9.677，r12+head16 7.063），ViT-B 2.097 ms（r5_v1 3.673，r12 2.722）。

### 1.1 op 级 sanity（`tests/test13a.py`，不是门；`results/test13a_d{1024,768}.json`）

- 量化器对 `round10 quant_ref.mx8_quantize` **逐位一致**：`quant_rows`（fp32 / fp16 / bf16 输入，K=d / K=4d，M=2058 / 2258，ViT-L 与 ViT-B）q 与 scale 字节 100% 相同；`gelu_quant` 100%；`ln_mx8` q 字节 0.999999（fp32 LN 求和顺序不同导致约 1e-6 个字节差 1 ulp），scale 100%。
- cuBLASLt GEMM（M=2058 / 2258，qkv/proj/fc1/fc2，随机数据含离群通道）对"反量化操作数的 fp32 乘积"：rel L2 **1.66e-3**（= bf16 输出舍入），cos 0.9999986，输出有限。对原始 fp32 操作数 rel L2 3.7e-2（MX8 量化误差本身，只记录）。
- `epi_rope` vs torch：rel L2 2.1e-4（fp16 输出舍入）；`epi_res` max abs 4.8e-7。

### 1.2 `r13_mx8b`（残差加法并进下一个 LN：`res_ln_mx8`）：没有收益，不采用

把 proj/fc2 的 `epi_res` 与随后的 `ln_mx8` 合成一个 kernel（每块 10 kernel）。ViT-L 同场（`logs/speed_vitl_s1b.log`）：model_ms r5_v1 9.608 / mx8a 4.933 / mx8b 4.983，frame_ms 9.832 / 5.075 / 5.053，mask-logit cos 与 mx8a 完全相同（0.999820，说明融合正确）。在噪声内，不采用；变体仍保留在 `stack13.py`（`r13_mx8b`）。

## 2. 第二步：自写 Blackwell MXFP8 block-scaled GEMM `mgx`

### 2.1 做了什么

`impl/mgx_core.cu`：`mg8_core.cu` 的结构（cp.async 多级流水 + XOR 交错 smem + ldmatrix，cubin-per-config）上换 MMA：
`mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`（e4m3 + 每行每 32 个 k 一个 UE8M0 scale，fp32 累加，scale 由 tensor core 应用）。

- **scale 操作数**：cuBLAS 分块布局 `[R/128, K/128, 32, 4, 4]` 里，一行的 4 个相邻 k 块的 scale 恰好是一个 32 位字。每 128 个 k 把 A、B 各一个 512 B atom 用 cp.async 搬进 smem 小环（环深 `ceil((ST-1)/KPW)+1`，BKB=64 时一个字服务两个 k-tile，高/低字节用 `>>16` 选取），lane (g,c) 取 A 的第 `g+8*(c&1)` 行 / B 的第 g 列的字（Spark 文档记录的 thread-id=0 取数规则），k32 步对应的字节号是指令立即数。任意 `BM, BN`（16 的倍数；跨 atom 的 tile 自动多搬一个 atom）。
- **epilogue（全部融合）**：
  - `EPI 2`（qkv）：bias + RoPE(fp32) + fp16 `[3,B,Ntok,H,64]`（可选 smem 暂存后 16 B 行写）；
  - `EPI 1`（proj / fc2）：`out_f32 = R + γ·(acc + bias)`；
  - `EPI 0`（fc1）：bias + GELU(erf) → **MX8 量化输出**：一个 1×32 块是同一行的 4 个相邻 nj × 4 个 lane，amax 用 2 次 shuffle，RCEIL scale、cvt.rn.satfinite.e4m3，经 smem 暂存后 16 B 行写 fp8 字节，lane 0 写分块 scale 字节（直接是 fc2 的操作数）；
  - `EPI 4`（ConvT）：纯 GEMM → bf16。
- **PDL**：prologue 先预取权重 tile（及其 scale atom），再 `griddepcontrol.wait`，再取激活；prologue 后 `launch_dependents`。`mx13.cu` 的所有 kernel 也用 `cudaLaunchKernelEx` + PDL 属性（`MX13_PDL=0` / `MGX_PDL=0` 关闭，用于串行化的 nsys）。实测收益很小（ViT-L 约 −0.04 ms，见 §2.5）。
- 其它文件：`impl/mgx.py`（构建 / 启动 / 包装函数；cubin 经 `impl/mgxl.cu` = round10 `mg16l.cu` 的拷贝加载，CUDA graph 安全）、`tests/tune_mx.py`（tile 扫描）、`tests/test_mgx.py`（op 级 sanity + 微基准）、`tests/bench_big.py`（大 GEMM mainloop 探针）、`impl/tuned_mx.json`（最终 tile 表）。

### 2.2 op 级 sanity（`tests/test_mgx.py`；对"反量化操作数的 fp32 乘积"+ torch 里的 epilogue；`results/test_mgx_d{1024,768}.json`）

| ViT-L / M=2058 | rel L2 | cos | max abs | 备注 |
|---|---:|---:|---:|---|
| qkv（bias+RoPE+布局，fp16 输出） | 2.07e-4 | 1.0000000 | 3.8e-3 | = fp16 输出舍入 |
| proj（γ + fp32 残差） | 3.9e-8 | 1.0 | 7e-7 | fp32 输出，纯累加顺序差 |
| fc2（γ + fp32 残差，K=4096） | 1.6e-7 | 1.0 | 4.8e-6 | 同上 |
| fc1（GELU + MX8 量化输出） | 2.65e-2 | 0.99965 | 0.125 | 量化输出的 e4m3 字节与 `quant_ref.mx8_quantize(GELU(...))` 一致 99.99993%，scale 字节 100%；2.65e-2 就是 MX8 量化本身的误差 |
| plain（bf16 输出） | 1.66e-3 | 0.9999986 | 9e-3 | = bf16 输出舍入 |

ViT-B 同样（qkv 2.08e-4，proj 3.2e-8，fc2 1.1e-7，fc1 q 字节一致 99.9999%，plain 1.66e-3），M=2258 同样，全部有限。

### 2.3 tile 扫描与单核结果（`tests/tune_mx.py`，只扫 M=2058 / 2258 与 ConvT 形状）

- 空间：BM ∈ {64,80,96,112,128,160,192,256}，BN ∈ {64,128,192,256}，BKB ∈ {64,128}，ST ∈ {2,3,4}，warp 布局 9 种，MINB ∈ {1,2,3}，ORDER/UNR，STG。第一轮每个（模型, 类）随机 40 个，第二轮 300 个（3473 个 cubin，nvcc 335 s 并行），最后对 ViT-L qkv 做了 4000 个配置的近乎穷举（`logs/tune_*.log`）。阶段 1 单次 graph 回放计时取 top-8~10，阶段 2 在每个 M 上交替 3 轮重测取中位。M=2258 只在 M=2058 的 top 里重选（只影响 3 个联合层）。
- 结果（µs，graph 内连续 10 次的均值；TFLOPS 按 2MNK）：

| 形状 | 自写 mgx（含融合 epilogue） | 对照：cuBLASLt GEMM 本体（qkv 为本轮模型内 nsys；其余取 round10 §3.3 的单核数）+ 独立 epilogue 遍（本轮模型内 nsys 均值） | tile |
|---|---:|---:|---|
| ViT-L qkv 2058 | 32.8（395） | 26.3 + 6.7（epi_rope）= 33.0 | 80x256x64 st2 w1x8 minb2 stg |
| ViT-L proj 2058 | 14.8（291） | ≈10.5 + 7.3（epi_res）= 17.8 | 128x64x64 st2 w2x4 minb1 |
| ViT-L fc1 2058 | 39.1（441） | ≈34.4 + 14.5（gelu_quant）= 48.9 | 112x128x64 st3 w1x4 minb2 |
| ViT-L fc2 2058 | 37.1（466） | ≈32.0 + 7.3 = 39.3 | 112x128x128 st2 w1x8 minb1 |
| ViT-L ConvT1 (M=2048) | 37.3（460） | 37.5（cuBLASLt） | 96x128x64 st3 w2x4 |
| ViT-L ConvT2 (M=8192) | 130.6（526） | 141.9（cuBLASLt） | 256x128x128 st2 w2x4 |
| ViT-B qkv / proj / fc1 / fc2 2058 | 23.3 / 10.9 / 23.5 / 24.3 | | `impl/tuned_mx.json` |

- **mainloop 探针**（`tests/bench_big.py`，8192×8192×4096，无 wave / prologue 影响）：cuBLASLt 624 TFLOPS，mgx 最好 589（94%；BKB=128 128x128 st3 w2x4）。说明 mainloop 本身已接近 cuBLASLt；小形状上的差距主要是 wave 量化（例如 qkv 在 128x128 下 17×24=408 个 tile、170 个 SM）和 prologue/epilogue 占比，cuBLASLt 用 128x128x128 TMA 单 CTA/SM，3 波 × 9.1 µs ≈ 27 µs。qkv 在 plain epilogue 下 160x128 / 4 warp 可到 28.9 µs（cuBLASLt 27.5），带 RoPE epilogue 后掉到 ≥32.8 µs。

### 2.4 逐类取优（模型内，`logs/speed_vitl_s2d.log`、`s2e`、`speed_vitb_explore.log`，5 轮，`+lt=<类>` = 该类留在 cuBLASLt + 独立 epilogue 遍）

| 变体 | ViT-L s2d | ViT-L s2e（另一时刻） | ViT-B |
|---|---:|---:|---:|
| r13_mx8a（全 cuBLASLt） | 4.979 | 5.138 | 2.098 |
| r13_mx8（全自写） | 4.903 | — | 1.953 |
| +lt=qkv | **4.711** | **4.901** | **1.937** |
| +lt=fc2 | 4.730 | — | 1.974 |
| +lt=proj | 4.814 | — | 2.002 |
| +lt=qkv.fc2 / qkv.proj / qkv.fc2.proj | — | 4.919 / 4.948 / 4.966 | — |
| +lthead（ConvT 走 cuBLASLt） | 4.912 | — | 1.948 |

只有 qkv 留在 cuBLASLt 一致地不亏（ViT-L −1~4%，ViT-B 持平）；proj / fc2 / fc1 / ConvT 自写都不比 cuBLASLt 版差。最终配置 `r13_mx8best` = `r13_mx8+lt=qkv`。

### 2.5 PDL 的收益

仅 GEMM 加 PDL：ViT-L r13_mx8 4.883 → 4.842 ms；再给 `mx13.cu` 的 LN/量化/epilogue kernel 加 PDL：4.831（`logs/speed_vitl_s2a/s2b/s2c.log`），都在噪声边缘。nsys 里 kernel 间平均空隙只有 0.45 µs（非 PDL）/0.19 µs（PDL，kernel 时间互相重叠），所以 launch 间隙不是瓶颈。

## 3. 每帧 kernel 时间分解（nsys，`MX13_PDL=0 MGX_PDL=0`，串行，20 次 graph 回放，`tests/prof13.sh` + `tests/ksum13.py` + `tests/gemm_split.py`）

PDL 打开时 kernel 时间互相重叠（kernel 之和 6.17 ms 对 span 4.65 ms），所以分解用串行 profile。

**ViT-L `r13_mx8best`**：kernel 总和 **4.613 ms/帧**，231 launch（nsys 回放 span 4.717 ms；speed12 的 model_ms 4.904，二者差 ≈4% 是 clock / 另一租户波动）。

| 部分 | kernel | ms/帧 | % | 次/帧 | 平均 µs |
|---|---|---:|---:|---:|---:|
| GEMM（自写）fc1 | `gemm_kernel` 32x19 / 32x21 | 0.804 + 0.117 | 20 | 24 | 38.3 / 39.0 |
| GEMM（自写）fc2 | `gemm_kernel` 8x19 / 8x21 | 0.787 + ≈0.114 | 19.5 | 24 | 37.5 / ≈38 |
| GEMM（自写）proj | `gemm_kernel` 16x17 / 8x21 | 0.307 + ≈0.049 | 7.7 | 24 | 14.6 / ≈15.5 |
| GEMM（自写）ConvT2 / ConvT1 | `gemm_kernel` 32x32 / 32x22 | 0.124 + 0.038 | 3.5 | 1 / 1 | 124.3 / 38.3 |
| GEMM（cuBLASLt）qkv | `cutlass3x_sm120_..block_scaled..128x128x128` | 0.632 | 13.7 | 24 | 26.3 |
| epilogue 遍 | `epi_rope_k`（qkv 的 bias + RoPE + 布局） | 0.161 | 3.5 | 24 | 6.7 |
| attention（r12） | `a16::fwd_kernel<4>` | 0.839 | 18.2 | 24 | 35.0 |
| LN → MX8 | `ln_mx8_kernel` | 0.216 | 4.7 | 48 | 4.5 |
| attention 输出 → MX8 | `quant_rows_k<half>` | 0.055 | 1.2 | 24 | 2.3 |
| head | `dw_ln2`（2 次）0.186 + mask-head bmm 0.104 + mask-head Linear 0.026 + tail/量化/gelu 0.016 | 0.332 | 7.2 | | |
| patch-embed + cat | `patch_gemm_kernel` 0.028 + cat 0.009 | 0.037 | 0.8 | | |

（自写 GEMM 组合计 2.341 ms，模型内每类的平均时间与 tile 表里的单核时间一致：38.3 / 37.5 / 14.6 / 124.3 vs 39.1 / 37.1 / 14.8 / 130.6。）

**ViT-B `r13_mx8best`**：kernel 总和 **1.907 ms/帧**，123 launch：自写 GEMM 0.835（38 次）、attention 0.419、cuBLASLt qkv 0.196（12 × 16.3 µs）、`dw_ln2` 0.139、`ln_mx8` 0.085、mask-head bmm 0.079、`epi_rope` 0.068、`quant_rows` 0.030、patch 0.023、其余 0.03。

对照第一步 `r13_mx8a`（ViT-L，同方法的 nsys，kernel 总和 4.894 ms）：cuBLASLt GEMM 2.564（98 次，26.2 µs）+ `gelu_quant` 0.349 + `epi_res` 0.348 + `epi_rope` 0.159 + `quant_rows` 0.054 + `ln_mx8` 0.213 + attention 0.837 + head 0.33。自写 kernel 融合掉了 `gelu_quant` / `epi_res`（共 0.70 ms），GEMM 时间（含 epilogue）从 2.564 增加到 2.973 ms（+0.41），净省 ≈0.28 ms kernel 时间（4.894 → 4.613）。

## 4. 剩余时间在哪、没做的

ViT-L 4.61 ms/帧（kernel 和）：GEMM 2.97 ms（64%，其中 qkv 的 cuBLASLt 0.63）、attention 0.84（18%，原样 fp16 kernel，本轮不动）、LN / 量化 / RoPE 遍 0.43（9%）、head 非 GEMM 部分 0.33（7%）。

没做 / 可以继续：
1. **GEMM 小形状效率**：大 GEMM 上 mgx 达 589 TFLOPS（峰值 981，cuBLASLt 624），但 M=2058 的实际形状只有 290–470 TFLOPS，主要是 wave 量化与 prologue/epilogue 占比。若用 persistent / stream-K 式调度或 TMA + warp specialization（cuBLASLt 的做法），保守估计还能省 0.3–0.6 ms/帧。本轮 tile 扫描没有覆盖 persistent 调度。
2. **qkv 自写 kernel**：RoPE epilogue 把 plain 28.9 µs 拉到 32.8 µs，模型内不如 cuBLASLt + 独立 `epi_rope`；可把 sin/cos 在 mainloop 前预取、或在 epilogue 里少做一次 smem 往返。
3. attention（0.84 ms）按指示没碰；其输出量化（`quant_rows`，0.055 ms）可并进 attention epilogue。
4. mask einsum bmm（0.104 ms）和 `dw_ln2`（0.186 ms）仍是 bf16；einsum 换 MXFP8 预计省 ~0.06 ms。
5. 精度：按用户决定全部不测；mask-logit cos 0.9995–0.9999（只记录）。没有跑 IoU / 置信度扫描，也没有检查 graph==eager。
6. M=2258（3 个联合层）的 tile 只在 M=2058 的 top 里重选，没有独立扫描。

## 5. 文件与复现

- `plug13.py`（speed12 插件，`PREFIXES=("r13",)`）、`impl/stack13.py`（变体：`r13_mx8a`、`r13_mx8b`、`r13_mx8[+lt=qkv.fc2][+lthead][+nohead]`、`r13_mx8best`）、`impl/mx13.{py,cu}`（量化器 / LN / epilogue 遍 / cuBLASLt 包装）、`impl/mgx_core.cu` + `impl/mgx.py` + `impl/mgxl.cu` + `impl/tuned_mx.json`（自写 GEMM）。
- `tests/test13a.py`、`tests/test_mgx.py`、`tests/tune_mx.py`、`tests/bench_big.py`、`tests/prof13.{py,sh}`、`tests/ksum13.py`、`tests/gemm_split.py`；`results/`（speed_s1 / s2 / final.json、test 结果）、`logs/`。
- 速度：`cd /root/lightly-train/infer-opt/round12_lowp_stack && python speed12.py --model vitl16 --variants r5_v1,r12+head16,r13_mx8best --plugin /root/lightly-train/infer-opt/round13_mxfp8_full:plug13 --tag t --out <本轮 results/xxx.json>`（注意 `--out`：speed12 默认写 round12 目录）。
- nsys：`MX13_PDL=0 MGX_PDL=0 tests/prof13.sh vitl16 r13_mx8best <tag> 20`，再 `python3 tests/ksum13.py /root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full/prof_<tag> 20`。
