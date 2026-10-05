# Round 0：PyTorch eager 基线与瓶颈分析

## 测试设置
| 项 | 内容 |
|---|---|
| 模型 | ViT-L/16 与 ViT-B/16 的 DINOv3-EoMT **语义分割**模型（真实权重 `/root/autodl-tmp/vit{l,b}-best.pth`），类别 {0: background, 1: head}，100 个 query，3 个 joint 层 |
| 输入 | 真实视频 `/root/autodl-tmp/1.mp4`（1900×1440，60fps），均匀抽 8 帧；frame 0 用于计时，8 帧都用于精度 |
| 处理流程 | 短边缩放到 512 → 675×512 → 2 个 512×512 crop（batch=2）→ 后处理输出 1440×1900 |
| 调用路径 | 与 `scripts/infer_dinov3_eomt_head_video.py::predict_head` 完全一致（见 `common/model_factory.py` 的 `base_prep / base_model / base_post`） |
| 计时 | CUDA event + wall-clock，每次迭代同步；warmup 20、100 次迭代 × 3 轮，取中位数；持有 GPU 锁 |
| 精度门槛（用户接受 ±5%） | 与 fp32 eager 相比，8 帧合并的 head IoU ≥ 0.95，且每帧 confidence 相对误差 ≤ 5% |

## 结果（wall p50，单位 ms）
完整数据见 `results.json`；旧的 docs 照片（1920×1080）数据见 `results_docs1080p.json`。

| 模型 | variant | prep | model | post | e2e | e2e FPS | kernels/model fwd | 合并 head IoU | conf 相对误差 max | 门槛 |
|---|---|---|---|---|---|---|---|---|---|---|
| vitl16 | **bf16-autocast（主基线）** | 5.77 | 17.05 | 35.36 | **60.65** | **16.5** | 1302 | 0.9931 | 0.0111 | ✅ |
| vitl16 | bf16-pure | 5.56 | 15.10 | 35.09 | 57.83 | 17.3 | 1040 | 0.9897 | **0.0492** | ✅（接近上限） |
| vitl16 | fp32 | 5.35 | 40.84 | 34.77 | 81.09 | 12.3 | 933 | 1.0000 | 0.0000 | ✅ |
| vitb16 | **bf16-autocast（主基线）** | 5.76 | 9.62 | 34.84 | **51.71** | **19.3** | 682 | 0.9926 | 0.0038 | ✅ |
| vitb16 | bf16-pure | 5.86 | 8.09 | 34.10 | 51.61 | 19.4 | 540 | 0.9933 | 0.0028 | ✅ |
| vitb16 | fp32 | 6.06 | 15.26 | 35.25 | 57.91 | 17.3 | 489 | 1.0000 | 0.0000 | ✅ |

> ⚠️ **ViT-L 的 bf16-pure 置信度误差 4.92%，已经贴近 5% 的上限**；autocast 下只有 1.1%。`.to(bf16)` 会把 RoPE 的 periods 以及 LayerNorm 的计算也变成 bf16。**后续所有轮次必须保持 RoPE 的 sin/cos 用 fp32 计算、LN 的统计量用 fp32**，只有权重和激活用 bf16。

## 5090 实测峰值（`results/peaks.json`，用来代替 ncu 的 SOL 基准）
| 项 | 实测值 |
|---|---|
| bf16 GEMM 8192³ | **232 TFLOPS** |
| bf16 GEMM 4096³ | 210 TFLOPS |
| 模型形状（M=1029） | qkv 126 / proj 131 / fc1 168 / fc2 168 TFLOPS |
| FP8 e4m3 `_scaled_mm` 8192³ | **548 TFLOPS**（2.4× bf16） |
| FP8 fc1 形状 | 372 TFLOPS |
| DRAM 拷贝 | 1.53 TB/s |
| L2 拷贝（16 MB） | 4.09 TB/s |
| eager launch | 4.97 µs/op |
| CUDA Graph 内每个 kernel | **0.75 µs** |

- ViT-L 每个 crop 约 785 GFLOP，2 个 crop 约 1.57 TFLOP，按 232 TFLOPS 算**模型下界约 6.8 ms**。
- ViT-B 每个 crop 约 249 GFLOP，下界约 2.1 ms。

## nsys 分析（ViT-L，bf16-autocast，每帧）
报告：`profile/vitl16_bf16autocast_e2e_summary.md`（由 `common/nsys_analyze.py` 生成）。按阶段拆分如下（每个 kernel 按发起它的 CUDA API 调用的 CPU 时间归属到阶段）：

| 阶段 | CPU 时间 | GPU 时间跨度 | kernel 数 | kernel 时间 | memcpy | 跨度内 GPU 忙碌率 |
|---|---|---|---|---|---|---|
| prep | 7.39 | 1.22 | 8 | 0.13 | 3 次 / 0.75 | — |
| model | 20.61 | 21.52 | **1302** | **15.35** | 0 | **71%** |
| post | 36.01 | 30.44 | 14 | 0.64 | 2 次 / 1.44 | **7%** |

（开着 nsys 时 CPU 时间会偏大，绝对值以上面的 wall p50 为准。）

ViT-B 的对应数据：model 阶段 682 个 kernel，kernel 时间 6.04 ms，GPU 跨度 11.99 ms，**忙碌率只有 50%**；post 阶段的忙碌率同样是 7%。

### model 阶段的 kernel 时间构成（ViT-L，15.35 ms）
| 类别 | 时间/帧 | 次数/帧 | 说明 |
|---|---|---|---|
| cutlass sm80 bf16 GEMM（128×128、256×128、128×64 三种 tile） | ~7.2 ms | 96 | torch/cuBLAS 在 sm_120 上仍然走 **Ampere 的 mma.sync 路径** |
| flash attention（pytorch_flash fwd） | 1.43 ms | 24 | 59.5 µs/次 |
| **autocast 的 bfloat16_copy** | **1.61 ms** | **309** | 每帧都把 fp32 权重重新 cast 一遍 |
| **RoPE 的 CatArrayBatchedCopy** | 0.91 ms | 96 | rope_apply 里的 cat |
| LayerNorm | 0.58 ms | 51 | |
| 其它逐元素（add/mul/neg/cast…） | ~2 ms | 数百次 | 平均每次 5 µs，都是启动开销主导的小 kernel |
| ConvTranspose2d（走 cuDNN dgrad） | 0.32 ms | 1 | head 第二级 |

CUDA API：`cudaLaunchKernel` 1220 次/帧，共 5.05 ms；`cudaMemcpyAsync` 5 次共 3.76 ms（**pageable** 内存上的 H2D/D2H）。

## 结论：三个瓶颈，严重程度依次递减
1. **后处理（35 ms，占 e2e 的 58%）几乎全在 CPU 上**：`.float().cpu()` 把 2×1440×1900 的 fp32（21.9 MB，pageable）拷回 host，然后用 numpy 做 argmax、类别映射和 mean。GPU 忙碌率只有 7%。→ Round 7：GPU 上融合 einsum、untile、resize、argmax 和置信度规约，只回传 uint8 mask 和一个标量。
2. **模型阶段受启动开销限制**：ViT-L 有 1302 个 kernel，GPU 忙碌率 71%；ViT-B 只有 50%。而且 kernel 时间里约 25% 是无效开销：autocast cast 1.6 ms、RoPE cat 0.9 ms，外加大量小的逐元素 kernel。→ Round 1（纯 bf16、预计算、权重折叠）+ Round 2（CUDA Graph，launch 从 5 µs 降到 0.75 µs）+ Round 3（算子融合）。
3. **前处理 5.8 ms 受 CPU 限制**：cvtColor → PIL → tensor → pageable H2D，GPU 上只有约 0.9 ms。→ Round 7 的 P1：pinned 内存 + 一个融合 kernel 完成颜色转换、normalize、antialias resize 和切块。
4. **GEMM 本身**：即使消除了所有开销，ViT-L 的模型阶段也有约 7.2 ms 的 GEMM + 1.4 ms 的 attention。torch 的 GEMM 在模型形状上只有 126~168 TFLOPS，距 232 TFLOPS 的峰值还有差距（qkv/proj 的 wave quantization）。→ Round 5（sm_120 GEMM 调优）+ Round 6（FP8 峰值 548 TFLOPS，2.4×）。

## 精度基线
- bf16-autocast 与 fp32 相比：ViT-L 的合并 head IoU 0.9931、置信度相对误差 1.1%；ViT-B 为 0.9926、0.4%。都**通过**门槛。
- 这就是"bf16 本身带来的误差"的参照量：后续各轮的误差如果在这个量级，可以认为没有额外损失。
- 纯 bf16（`.to(bf16)`）下，ViT-L 的置信度误差上升到 4.9%。所以数值敏感的部分（RoPE、LN 统计量、softmax、后处理）必须保持 fp32。FP8/FP4（Round 6）剩下的误差预算很紧，需要逐层评估敏感度。
