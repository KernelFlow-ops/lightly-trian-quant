# DINOv3 + EoMT 语义分割推理优化（RTX 5090 / sm_120）

基线是 `scripts/infer_dinov3_eomt_head_video.py`（PyTorch eager + bf16 autocast）；测试数据是真实视频 `/root/autodl-tmp/1.mp4`（1900×1440，126 帧），模型权重为 `vitl-best.pth` / `vitb-best.pth`。
方案见 [PLAN.md](PLAN.md)（以顶部"修订 v2"为准）。全部结果汇总见 [results/summary.md](results/summary.md)，用 `python common/summarize.py` 重新生成。

## 当前推荐（宽图，整段视频稳态）

| 模型 | 原脚本（同轮） | 优化方案 | 加速 | 单帧服务延迟 | 精度（与 fp32 相比，±5% 门槛） |
|---|---|---|---|---|---|
| ViT-L/16 | 15.1 FPS | **94.7 FPS** | **6.3×** | 10.24 ms | head IoU 0.9946，conf 误差 0.13% ✅ |
| ViT-B/16 | 16.5 FPS | **214.1 FPS** | **13.0×** | 4.00 ms | head IoU 0.9943，conf 误差 0.21% ✅ |

优化方案 = Round 5 调优的融合 block（fp32 残差流）+ Round 7 融合前后处理和 head + Round 9 整帧 CUDA Graph + Round 8 多 stream 流水线。数字来自每实现 3×126 帧的同轮配对测试；相比 Round 3 `fused`，吞吐分别提高 2.4% / 2.8%。在 126 帧上与原脚本逐帧对比，mask 像素一致率最低 0.99998，合并 head IoU 约 0.992。

**直接使用**：
```bash
cd /root/lightly-train/infer-opt
python round9_integration/video_infer.py --model /root/autodl-tmp/vitl-best.pth \
    --video-dir /root/autodl-tmp/1.mp4 --out-dir /root/autodl-tmp/infer_out
```
默认双 crop 宽图使用 `r5_v1`，竖图或其他 crop 数自动回退 `lightly` 并打印实际实现；`--model-impl` 可选择其他双 crop 宽图实现。`--fp8` 只与 `--model-impl lightly` 搭配，且仍要求双 crop 宽图。

## 各轮一览（ViT-L，单位 ms）

| 轮次 | 内容 | 关键结果 |
|---|---|---|
| [round0_baseline](round0_baseline/NOTES.md) | eager 基线、实测峰值、nsys 瓶颈分析 | 每帧 e2e 60.7 = prep 5.8 + model 17.1（1302 个 kernel，GPU 忙碌率 71%）+ **post 35.4（GPU 忙碌率 7%，时间花在 CPU 的 numpy argmax 上）** |
| [round1_torch_cleanup](round1_torch_cleanup/NOTES.md) | bf16 权重常驻、预计算 RoPE、折叠 LayerScale、选择 SDPA 后端 | model 12.9，kernel 数 440；**flash SDPA 在 ViT-L 上不过门槛，改用 cuDNN** |
| [round2_cuda_graph](round2_cuda_graph/NOTES.md) | CUDA Graph，与 torch.compile 对比 | graph 12.6（model 段 CPU 时间 7.75 → 0.82）；compile max-autotune 10.3 |
| [round3_fusion](round3_fusion/NOTES.md) | TileLang GEMM epilogue 融合（qkv+RoPE、fc1+GELU、proj/fc2+残差）+ add+LN | 每层 40 → 7 个 kernel，model 11.27；**残差流必须用 fp32**（第 0 层起就有 1.5e5 的 massive activation） |
| [round4_attention](round4_attention/NOTES.md) | TileLang FlashAttention（sm_120），可选融合 RoPE | 纯 attention 与 flash 持平；融合 RoPE 比 flash + RoPE 算子快 35~40% |
| [round5_gemm_sm120](round5_gemm_sm120/NOTES.md) | sm_120 GEMM 调优、split-K/stream-K、LN prologue、PDL | tuned v1 在整模型交错 graph 对照中快 4.3% / 6.7%（L/B），并提高整段视频吞吐；v2/PDL 未获净收益 |
| [round6_low_precision](round6_low_precision/NOTES.md) | 逐行 FP8、真实 MXFP8、NVFP4 模型级筛选 | ViT-B MXFP8 相对同 units 的 row-FP8 快约 3%，但精度余量变小；NVFP4 覆盖空间很小，默认不量化 |
| [round7_head_postprocess](round7_head_postprocess/NOTES.md) | GPU 前处理、融合后处理、融合 head | prep 5.8 → 0.7，post 35.4 → 0.22，e2e 60.7 → 17.4 |
| [round8_pipeline](round8_pipeline/NOTES.md) | 解码直接写 pinned ring、H2D/compute/D2H 三个 stream 重叠 | 15.5 → 57.3 FPS |
| [round9_integration](round9_integration/NOTES.md) | 整帧单图 CUDA Graph + 可插拔 model | 同轮测试：ViT-L 94.7 FPS、ViT-B 214.1 FPS；宽图默认 Round 5 v1，竖图回退 eager backbone |

## 性能分析方法
- 这个容器**不能用 ncu**（ERR_NVGPUCTRPERM，宿主机不开放性能计数器）。替代做法：
  - nsys 时间线：`common/profile_nsys.sh`，再用 `common/nsys_analyze.py` 按阶段归因 kernel 并统计 GPU 忙碌率
  - 解析 roofline：`common/roofline.py`，对比实测峰值 `results/peaks.json`（bf16 232 TFLOPS、FP8 548 TFLOPS、DRAM 1.53 TB/s）
  - 静态分析：`common/kernel_static.sh`，看寄存器、spill、SASS 指令
- ncu 的命令模板已经写好（`common/profile_ncu.sh`），换到有计数器权限的机器上可以直接运行。
- 所有计时都持有同一把 GPU 锁（`common/bench.py`），避免多个 worker 并发计时互相污染。Round 2 早期的数据就被污染过，已经移到 `old_contaminated/` 并重新测量。

## 精度门槛与注意事项
- 门槛（用户接受 ±5%）：8 个评测帧合并的 head IoU ≥ 0.95，且每帧置信度相对误差 ≤ 5%（`common/model_factory.py::gate`）。
- **ViT-L 的第 7 个评测帧很脆弱**（head 只有 1139 像素，置信度只有 0.48），任何 bf16 扰动都可能让它的相对误差在 0.5%~5.7% 之间波动。所以每个新变体都要重新测，并且要看 126 帧逐帧对比的结果。
- 必须保持 fp32 的部分：残差流、LN 统计量、RoPE、softmax、后处理。整个模型 `.to(bf16)` 会让 ViT-L 贴近门槛，bf16 残差则直接不过门槛。
- 已知限制：只支持 2 个类别加 void、单路视频、最多 8 个 crop；前后处理的宽/竖图切块已对照基线。竖图目前只验证了一帧真实视频，相对原脚本的 head IoU 为 0.982；融合 backbone 在该帧未达 0.95，因此入口自动使用 `lightly`。Round 5 的 tile 配置只在 batch=2（每帧 2 个 crop）上调过，其他 crop 数同样回退；竖图完整视频的 fp32 gate 仍待测。

## 第二阶段（进行中，2026-10-04 起）：Blackwell 低精度 + 自研算子

方案见 [PLAN_v2.md](PLAN_v2.md)。在 R9 推荐栈（`r5_v1`）之上新增 round10–14，R0–R9 不改动。

- 精度分两档：默认 safe（8 帧门 + 126 帧全片都过）；可选 fast（只要求 8 帧门）。
- 新的数值路径：FP16 输入 + FP16 累加（分块提升到 FP32）、MXFP8、NVFP4/MXFP4；attention 试 FP16 累加 / INT8+FP8 / FP4。
- 可选开关，默认关：多帧合批吞吐模式、单次宽图推理。
- GPU 锁：第二阶段所有 GPU 任务用 `common/lock2.py::gpu_lock2()`，同时持 infer-opt 和 train-opt 的锁。

| 轮次 | 内容 | 状态 |
|---|---|---|
| round10_lowp_probe | GEMM 侧峰值、速度矩阵、精度仿真、go/no-go | 进行中 |
| round11_attn_tier0 | bf16 等价小项（patch-embed GEMM、dw_ln、post_a）+ attention 候选矩阵 | 进行中 |
| round12_gemm_lowp | 生产版低精度 GEMM + 生产者融合量化 | 待开始 |
| round13_attn_lowp | 生产版低精度 attention | 待开始 |
| round14_final | 集成、系统层（解码 / 合批 / host 开销）、发布 | 待开始 |

### 第二阶段进展（2026-10-05）

用户在 2026-10-05 调整了方向：低精度方案不再用精度门卡（权重是 bf16 训练的），直接比较 MXFP8 / NVFP4 / MXFP4 整模型与 baseline 的速度。PLAN_v2 里的 R12/R13 合并成了 round12，R13/R14 改为整模型低精度测速。

| 轮次 | 内容 | 状态 / 结果 |
|---|---|---|
| [round10_lowp_probe](round10_lowp_probe/NOTES.md) | 张量核峰值、GEMM 速度矩阵、精度仿真 | 完成。fp16 累加指令速率是 bf16 的 1.99×，MXFP8/INT8 3.97×，FP4 7.95× |
| [round11_attn_tier0](round11_attn_tier0/NOTES.md) | patch-embed GEMM、dw_ln、post_a、尾部融合 + attention 候选矩阵 | 完成。r11：ViT-L +3.4%，ViT-B +6.7% |
| [round12_lowp_stack](round12_lowp_stack/NOTES.md) | FP16 整模型栈 r12（fp16 累加 GEMM + fp16 累加 attention）+ 测速脚本 speed12.py | 完成。ViT-L 96.3 → 124.3 FPS，ViT-B 227 → 273 FPS；门 M/H/G 全过 |
| round13_mxfp8_full | 整模型 MXFP8 测速 | 进行中 |
| round14_fp4_full | 整模型 NVFP4 / MXFP4 测速 | 进行中 |

### 第二阶段结果：整模型低精度 vs baseline（2026-10-05，精度门关闭，只比速度）

上表里 round13 / round14 的"进行中"已经完成，以这一节为准。所有数字是 `round12_lowp_stack/speed12.py` 的 CUDA graph 回放，候选交替运行、3 轮取中位数，持双锁。

三列倍率都是"变体 ÷ 同场 baseline"：
- **模型段倍率** = baseline 的 model_fn 耗时 ÷ 变体的 model_fn 耗时（只含模型，不含 prep / post）；
- **GPU 倍率** = GPU 上限 FPS 之比（整帧 graph：prep + 模型 + post，不含解码）；
- **视频倍率** = 含 cv2 解码的整段视频 FPS 之比（126 帧 × 3 遍，inflight 2）。

| 模型 | 方案 | 变体 | 模型段 ms | 模型段倍率 | GPU 上限 FPS | GPU 倍率 | 视频 FPS | 视频倍率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ViT-L | baseline（bf16，R9） | `r5_v1` | 9.77 | 1.00× | 100.7 | 1.00× | 94.6 | 1.00× |
| ViT-L | FP16 累加 | `r12+head16` | 7.08 | 1.38× | 128.9 | 1.28× | 120.4 | 1.27× |
| ViT-L | **MXFP8** | `r13_mx8best` | 4.90 | 1.98× | 200.7 | 1.98× | 183.4 | 1.88× |
| ViT-L | **NVFP4** | `r14_nv4a` | **4.02** | **2.43×** | 243.9 | **2.42×** | 207.5 | **2.19×** |
| ViT-L | **MXFP4** | `r14_mx4a` | **4.04** | **2.42×** | 242.9 | **2.41×** | 207.6 | **2.19×** |
| ViT-B | baseline | `r5_v1` | 3.62 | 1.00× | 264.5 | 1.00× | 215.4 | 1.00× |
| ViT-B | FP16 累加 | `r12` | 2.70 | 1.34× | 352.6 | 1.33× | 241.7 | 1.12× |
| ViT-B | **MXFP8** | `r13_mx8best` | 1.94 | 1.90× | 497.6 | 1.91× | 272.4 | 1.20× |
| ViT-B | **NVFP4** | `r14_nv4a` | **1.74** | **2.08×** | **553.5** | **2.09×** | 243.9 | 1.13× |
| ViT-B | **MXFP4** | `r14_mx4a` | **1.75** | **2.07×** | 548.4 | 2.07× | 239.8 | 1.11× |

- MXFP8 一行来自 round13 的子代理在另一场的测量，倍率用那一场的 baseline（ViT-L 9.711 ms / 101.2 FPS / 97.3 FPS；ViT-B 3.680 ms / 260.2 FPS / 226.2 FPS）；其余各行用本场（round14）的 baseline。两场 baseline 差 < 2%。
- ViT-B 的视频倍率只有 1.1–1.2×：视频 FPS 被 host / cv2 解码卡在约 240–290，GPU 倍率才反映模型本身。本场机器负载很高（load average 15–21），ViT-B 视频 FPS 比上一次（baseline 226 / MXFP4 281）低。
- FP16 一行在本场的整帧耗时偏高（ViT-L 7.76 ms，别的场 7.45–7.50 ms），所以 GPU 倍率 1.28× 比模型段倍率 1.38× 低；别的场的 GPU 倍率约 1.32×。
- 各方案做了什么：
  - FP16 累加（round12）：fp16 输入 + `mma.sync …f16.f16.f16.f16`，每个 k-tile 提升到 fp32；attention 也是 fp16 累加；LN、RoPE、残差流仍是 fp32。
  - MXFP8（round13）：自写 `kind::mxf8f6f4.block_scale` GEMM，LN + 量化融合，fc1 的 epilogue 直接量化成 fc2 的输入；qkv 用 cuBLASLt `_scaled_mm`。
  - NVFP4 / MXFP4（round14 step 1）：QuACK 的 `kind::mxf4nvf4.block_scale` GEMM + 自定义 epilogue（bias / GELU / γ + fp32 残差），激活用 fp4q2 量化器（NVFP4 的 per-tensor scale 是第一次调用时静态标定的）；骨干 4 个 GEMM 和 head 两个 ConvT 都是 FP4。
- 精度没有判定（用户决定：模型是 bf16 训练的，不指望低精度过门）。mask-logit 对 baseline 的余弦只作记录：ViT-L FP16 0.999996 / MXFP8 0.99989 / NVFP4 0.9990 / MXFP4 0.9960；ViT-B 0.999992 / 0.9995 / 0.9972 / 0.9883。
- R14 的自写 FP4 GEMM（step 2）没做完，目前比 QuACK 慢（见 `round14_fp4_full/NOTES.md` §7.3）。
