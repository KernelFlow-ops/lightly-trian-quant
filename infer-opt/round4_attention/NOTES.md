# Round 4：TileLang attention（sm_120）

文件：`impl/attn_tl.py`（kernel 生成）、`impl/ops.py`（`attention`、`attention_qkv`、`attention_rope`、`attention_rope_qkv`、`attention_3bnhd`）、`impl/tuned_cfg.json`、`test_attn.py`、`test_3bnhd.py`、`test_model_tl.py`、`tune_attn.py` / `sweep_all.sh` / `tune/`、`bench_attn.py`、`run.py`、`static_attn.py` / `static/`、`results.json`。

## 设计
online softmax，scale*log2e 折进 exp2 参数，fp32 累加；Q tile 常驻 smem，K/V 按 BLOCK_N 流水。TileLang 在 sm_120 上自动生成 TMA（UTMALDG/UTMASTG）+ mbarrier 流水、warp 专用化（launch_bounds 384 / 640 线程 = 计算 warp + 1 个 producer warpgroup）、mma.sync（HMMA）。
- 尾块：N=1029/1129 不是 tile 倍数。Q 越界行由 TileLang copy 保护、不写出；K 尾块把 j>=N 的 score 置 -inf。用 NaN 污染的越界内存测过，不产生 NaN。
- **重要 bug（TileLang 官方示例里就有）**：示例每个 tile 把 scores_max 重置为 -inf，只取本 tile 的最大值；后续 tile 最大值远小于前面时 exp2(prev-cur) 溢出。合成数据没问题，但真实 DINOv3 register token 的分数极尖（q·k 相差数百），整层出 NaN（整模型第一次测试全 NaN，已定位并修复：保留运行最大值，reduce_max(clear=False) 直接并入）。test_attn.py 已加入尖峰（scale 8）用例。
- 布局：`bhnd`（[B,H,N,64]，简报接口）；`qkv`（[B,N,3,H,64]，QKV GEMM 原始输出，输出直接是 [B,N,H*64]，无转置/拷贝）；`3bnhd`（[3,B,N,H,64]，blockfuse 的 K2 物理布局，输出同上，与 qkv 布局结果逐位相同）。
- **融合 RoPE**：加载 Q/K tile 时对 token>=prefix 的行在 fp32 里旋转，cast bf16 后进 smem；利用 DINOv3 的 sin/cos 前后半段相同，每个线程处理 (j, j+32) 一对，只读半张表。sin/cos 为 `backbone.rope_embed` 的 [1024,64] fp32。

## wave 分析（B=2, H=16）
BLOCK_M=64：2x16x17=544 CTA；128：288；256：160。128x32/256 线程时每 SM 只放 1 个 CTA（168 寄存器 x 384 线程占满寄存器堆），288/170=1.69 wave，第二个 wave 只有 69% 满；256（512 线程，96 寄存器）为 160 CTA，一个 wave、94% SM 占用，但 tile 浪费 1280 vs 1029 行（24%）。实测两者接近；最优随 H/N 变化：H=16,N=1029 plain 用 128x32/256 线程，N=1129 及所有 rope 版用 256x{32,64}/512 线程。调参空间 bm{64,128,256} x bn{32,64} x 线程{128,256,512}（`tune/*.json`，最优在 `impl/tuned_cfg.json`）。

## 正确性
`test_attn.py`（对 SDPA-bf16 和 fp32 SDPA）：B=2，H=16/12，N=1029/1129，随机 scale 2 与尖峰 scale 8，四种接口 cos >= 0.9999990，相对 fp32 的 max_abs 与 SDPA-bf16 自身相当（如 0.0181 vs 0.0181）。融合 RoPE 在尖峰合成数据上 max_abs 最多为 SDPA 误差的 2.6 倍（fp32 fma 与 torch 分步计算的舍入顺序不同），整模型 gate 仍通过。`test_model_tl.py`：TileLang eager 与 CUDA graph 捕获后逐位相同。

## microbench（GPU 时间，CUDA graph replay，B=2；us / TFLOPS）
| H,N | sdpa-flash | sdpa-cudnn | sdpa-efficient | TL plain [B,H,N,D] | TL plain_qkv | TL+RoPE 融合（qkv） | flash + RoPE 4 算子（R1 做法） |
|---|---|---|---|---|---|---|---|
| 16,1029 | 59.1 / 147 | 62.2 / 139 | 119.2 / 73 | 56.8 / 153 | 60.1 / 144 | **71.8 / 121** | 111.3 |
| 16,1129 | 59.8 / 174 | 65.5 / 159 | 128.4 / 81 | 61.3 / 170 | 63.1 / 166 | **77.3 / 135** | 113.1 |
| 12,1029 | 59.0 / 110 | 61.1 / 107 | 94.7 / 69 | 56.1 / 116 | 58.0 / 112 | 70.8 / 92 | 100.8 |
| 12,1129 | 59.7 / 131 | 64.0 / 122 | 100.3 / 78 | 60.6 / 129 | 62.2 / 126 | 76.1 / 103 | 100.9 |

TileLang 纯 attention 与 torch flash 持平（正负 5%），不更快，约占实测 bf16 峰值（232 TFLOPS）的 60-75%。**收益来自融合 RoPE**：比 RoPE 4 个算子 + flash 快 35-40%（约 40 us/层）。cudnn 比 flash 慢约 5%，efficient 慢约 2 倍。

## 整模型（SDPA 换成 TileLang；wall p50 最佳 repeat，ms）
| | ViT-L eager | ViT-L graph | kernels | conf 误差 / gate | ViT-B graph | kernels | gate |
|---|---|---|---|---|---|---|---|
| sdpa-cudnn（R1/R2） | 12.94 | 12.41 | 440 | 2.11% 通过 | 4.80 | 240 | 通过 |
| sdpa-flash | 13.05 | 12.39 | 440 | 5.65% 不通过 | 4.77 | 240 | 通过 |
| TileLang plain（tl） | 12.95 | 12.33 | 440 | 0.49% 通过 | 4.74 | 240 | 通过 |
| **TileLang + 融合 RoPE（tl_rope）** | **11.76** | **11.29** | **320** | 3.14% 通过 | **4.35** | **180** | 通过 |

融合 RoPE 使 ViT-L model 12.41 -> 11.29 ms（-9%），ViT-B 4.80 -> 4.35 ms（-9%），kernel 少 120 / 60 个。TileLang 与 flash 数值上同类（同为 bf16 P），但 ViT-L frame 7 的 gate 落在通过一侧（该 frame 的 conf 对 bf16 扰动很敏感，原因见 R1 NOTES；tl_rope 3.14% 比 tl 大，属同类波动）。

## 静态分析（`static/*.ptxas.txt`、`.sass`，代替 ncu）
| kernel | 配置 | 寄存器 | spill | SASS |
|---|---|---|---|---|
| plain_qkv 16x1029 | 128x32, 256 线程 | 168 | 0 | HMMA 32, LDSM 20, UTMALDG 9, UTMASTG 1, SYNCS 53, MUFU 52 |
| plain_qkv 16x1129 | 256x64, 512 线程 | 96 | 0 | HMMA 64, LDSM 36, MUFU 68 |
| rope_qkv 16x1029/1129 | 256x32, 512 线程 | 96 | 0 | HMMA 32, LDG 41, LDS 28, STS 41, F2FP 36, UTMALDG 8 |

无 LDGSTS（用 TMA 而非 cp.async），无 spill。rope 版多出的 LDG/LDS/STS 就是 tile 旋转（sin/cos 从 L2 读、fp32 计算），是它比 plain 慢 12-14 us 的来源。

## 如何替换 fused_block（blockfuse）里的 SDPA
- 输入：K2 输出 qkv，物理布局 [3,B,N,H,64] bf16 连续 -> `ops.attention_3bnhd(qkv)`，输出 [B,N,H*64] 直接喂 proj GEMM，无转置。K2 已在 epilogue 做完 RoPE 时用这个 plain 版（约 60 us，与 flash 持平，无收益但也无损）。
- 若 K2 不做 RoPE：`ops.attention_3bnhd(qkv, rope=(sin[1024,64] fp32, cos[1024,64] fp32, prefix))`，prefix = N-1024（普通层 5，joint 层 105）。比 K2 里做 RoPE 再调 flash 多 12-14 us，但省掉 4 个 RoPE 算子（约 40 us）；是否优于 K2 epilogue 融合需要集成后比较。
- kernel 在当前 stream launch、无 host 同步、形状/配置 lru 缓存，已验证可 graph 捕获。首次调用会 JIT 编译（每个形状约 5-8 s），集成时需在 capture 前 warmup。

## 遗留
- TileLang 纯 attention 没有比 flash 更快；再快需要手写 mma.sync 的 FA，预计只省几个 us/层，优先级低于 GEMM。
- 未尝试 fp8 / 更低精度。
