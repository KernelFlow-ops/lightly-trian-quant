# DINOv3 + EoMT 实例分割推理优化方案（RTX 5090 / Blackwell sm_120）

> 位置：`/root/lightly-train/infer-opt/`　　基线：PyTorch eager（与 `scripts/infer_*_video.py` 相同的调用路径）
> 主目标：**DINOv3 ViT-L/16 + EoMT**（辅：ViT-B/16），512×512 输入，batch=1 视频逐帧推理
> 方案日期：2026-09-30

---

## ⚠️ 修订 v2（拿到真实权重后，2026-09-30）

`/root/autodl-tmp/vitl-best.pth`、`vitb-best.pth` 是 **EoMT 语义分割**模型（`DINOv3EoMTSemanticSegmentation`），不是实例分割。与 v1 假设的差异如下，**以本节为准**：

| 项 | v1 假设（实例分割） | 实际（语义分割 checkpoint） |
|---|---|---|
| 类别 | — | `{0: background, 1: head}`，class_head 输出 K+1=3 |
| num_queries / joint blocks | 200 / L=4 | **100 / 3**（L 与 B 相同） |
| 输入切块 | resize + pad 到 512×512，batch=1 | 短边 resize 到 512，按长边切成若干个 512×512 重叠块：**1920×1080 → 910×512 → 2 块，batch=2** |
| 后处理 | 每个 query 插值到原图 + 打分 | `sigmoid(mask)` 与 `softmax(class)` 做 einsum → 逐像素 3 类 logits → untile 取平均 → 去掉 void 类 → 插值到原图 → argmax + head 置信度 |
| 视频脚本 | `infer_dinov3_eomt_head_instance_video.py` | `infer_dinov3_eomt_head_video.py::predict_head`：把 **(2,1080,1920) fp32 logits 整个拷到 CPU**，再用 numpy 做 argmax |

更新后的计算量（ViT-L/16，N=1029，joint 层 N=1129）：
- 每块约 **785 GFLOP**：21 个普通层 635 + 3 个 joint 层 101 + head 47
- 每帧 2 块，约 **1.57 TFLOP/帧**，按 bf16 实测峰值算下界约 9 ms
- ViT-B/16 约 247 GFLOP/块，约 0.5 TFLOP/帧

wave 利用率：batch=2 时 M=2058，qkv/fc1 约 80%，proj/fc2 是 136 个 tile，**仍不到 1 个 wave（80%）**。

对各轮的影响：
- v1 的 §1.2 第 6 条（按类别分数精确剪枝）**不再适用**。
- **后处理新方案**：一个融合 kernel 从 128×128 的 query logits 现算双线性插值、sigmoid，与 softmax(class) 做点积，并把两块重叠区的 untile 平均也并进去，直接得到 (2,512,910) 的逐像素 logits；再用一个 kernel 完成 resize 到 1080×1920、argmax 和 head 置信度规约。最终**只回传 uint8 mask 和一个标量**，不再拷 16.6 MB 的 fp32 数据、也不在 CPU 上做 argmax。
- 精度校验用 docs 里的真实照片（street/train/cats），resize 到 1920×1080 模拟视频帧，与 fp32 eager 参考比较 **head IoU、像素一致率和置信度差**。

---

## 0. 现状与约束（已在服务器上核实）

| 项 | 结果 |
|---|---|
| GPU | RTX 5090，**sm_120**，170 SM，32 GB，L2 96 MB，每 block 可用 smem 99 KB（opt-in 101376 B） |
| 驱动 / CUDA | 580.105.08 / CUDA 13.0（nvcc 13.0.88，`/usr/local/cuda/bin`，不在 PATH 中） |
| Python | `/root/miniconda3/bin/python` 3.12.3：torch **2.12.1+cu130**，torchvision 0.27.1，triton **3.7.1**，nvidia-cutlass 4.2.0，cuda-python 13.3 |
| 缺失 | lightly-train 依赖（cv2、lightning 等）未装；**tilelang 未装**（pip 上有 0.1.14）；无 TensorRT |
| 权重 | 服务器上**没有** DINOv3/EoMT 权重，先用随机权重（性能不受影响），真实权重后续补 |
| nsys | 2025.3.1 可用（CUDA/NVTX trace 正常） |
| **ncu** | **不可用**：`ERR_NVGPUCTRPERM`，宿主机没开 GPU 性能计数器；nsys `--gpu-metrics` 同样受限，容器内无法修复 |
| 参考 | torch 的 bf16 matmul 在 sm_120 上走 `cutlass_80_tensorop_bf16_s16816gemm`（Ampere 的 mma.sync 路径），2048³ 实测约 **174 TFLOPS** |

已确认的决策：ViT-L/16 为主、B/16 为辅；依赖直接装在 miniconda base（`pip install -e . --no-deps` 加缺的包，**不替换 torch**）；ncu 用替代方案，ncu 脚本预先写好。

---

## 1. 推理路径拆解（热点在哪里）

调用链：`model.predict(PIL)` → `preprocess_image`（to_dtype → normalize → resize → pad）→ `forward_backend` → `forward_train` → `_predict` → `F.interpolate(→512×512)` → `postprocess`（插值到原图分辨率 + `get_labels_masks_scores`）

关键代码：
- `src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/task_model.py`（`forward_train`、`_attn`、`_predict`、`get_labels_masks_scores`）
- `src/lightly_train/_models/dinov3/dinov3_src/layers/`（`attention.py`：RoPE / `LinearKMaskedBias`，`block.py`，`layer_scale.py`，`ffn_layers.py`，`rope_position_encoding.py`）
- `.../dinov3_eomt_instance_segmentation/scale_block.py`（ConvTranspose2d k2s2 → GELU → DWConv3×3 → LayerNorm2D，共 2 级：32→64→128）

### 1.1 解析计算量（ViT-L/16，d=1024，16 头，head_dim=64，24 层，joint=4，Q=200）

- token 数：32×32=1024 个 patch + 1 个 cls + 4 个 register = **1029**；最后 4 个 joint 层再拼 200 个 query，共 **1229**
- 每层线性层 FLOPs = 24·N·d²，attention = 4·N²·d

| 部分 | GFLOP/帧 | 占比 |
|---|---|---|
| 20 个普通层（N=1029）：GEMM 25.9 + attn 4.3 每层 | 605 | 75% |
| 4 个 joint 层（N=1229）：GEMM 30.9 + attn 6.2 每层 | 148 | 18% |
| Head：ConvT1 8.6 + **ConvT2 34.4** + einsum 6.7 + mask MLP 1.3 | 51 | 6% |
| patch embed | 1.6 | <1% |
| **合计** | **≈805** | GEMM ≈80%，attention ≈14% |

- 下界：805 GFLOP ÷ 实测 174 TFLOPS ≈ **4.6 ms**（bf16）；如果 FP8 真能到 2× 吞吐，约 2.5 ms（峰值在 round0 实测后再定）
- ViT-B/16：约 **256 GFLOP/帧**，下界约 1.5 ms。eager 模式下多半是**启动开销主导**

### 1.2 规模特征（决定优化方向）

1. **激活很小**：N×d bf16 = 1029×1024×2 B ≈ **2.1 MB**，单层权重约 25 MB，都能放进 96 MB 的 L2。LN、add、GELU、RoPE 这类逐元素 kernel 受延迟和启动开销限制，不受 DRAM 带宽限制。所以融合的收益主要来自**减少 kernel 数**，其次才是省带宽。
2. **eager 下 kernel 数很多（假设，round0 验证）**：每层 RoPE 的 cat/neg/mul/add/cast 约 16 个 kernel，**`rope_embed(H,W)` 每层都重新算一遍**（约 10 个 kernel），`LinearKMaskedBias` 每次调用都算一遍 bias×mask，LayerScale 是单独的 mul，autocast 会做 cast。估计每层 45~50 个 kernel，**整个 forward 超过 1100 个 kernel**，ViT-L 也可能是 CPU-bound。
3. **autocast 每帧重新 cast 权重**：`predict()` 每帧都进一次 autocast，fp32 权重（约 1.2 GB）每帧都要读一遍再写成 bf16（0.6 GB），大约是 **1 ms 以上的纯 DRAM 流量**。
4. **GEMM 的 wave quantization**（M=1029，128×128 tile，170 SM）：

   | GEMM | 形状 (M×N×K) | tile 数 | wave | SM 利用上限 |
   |---|---|---|---|---|
   | qkv | 1029×3072×1024 | 216 | 1.27 | 64% |
   | proj | 1029×1024×1024 | 72 | 0.42 | **42%** |
   | fc1 | 1029×4096×1024 | 288 | 1.69 | 85% |
   | fc2 | 1029×1024×4096 | 72 | 0.42 | **42%** → 需要 split-K / stream-K |

   另外 1029 = 8×128 + 5，最后一行 M tile 只有 5 行有效，浪费约 10.7% 的 MMA。
5. **attention 不到一个 wave**：16 头 × ⌈1029/128⌉ = 144 个 CTA，少于 170 个 SM，需要改 BLOCK_M 或做 split-KV。
6. **后处理可能是端到端最大头（假设）**：以 1080p 原图为例，200×1920×1080 ≈ 4.15 亿个元素。`interpolate` → `.float()` → `sigmoid` → `>0` → 乘 → sum 这一串会产生 **8 GB 以上的显存流量，约 5 ms 以上**。
   - **精确剪枝**：`score = cls_score × mask_score`，而 `mask_score` 是 logit>0 区域内 sigmoid 的均值，取值在 (0.5, 1]，所以 `score ≤ cls_score`。`cls_score < threshold` 的 query **一定会被丢掉**，可以在插值之前就剔除。阈值 0.9 时通常只剩几个 query，结果与原实现等价。
7. **每帧还有 telemetry**：`predict()` 会调 `_track_inference()`。基准测试统一设 `LIGHTLY_TRAIN_EVENTS_DISABLED=1`，并在 round0 单独测一下它的开销。

---

## 2. 目录结构

```
/root/lightly-train/infer-opt/
├── PLAN.md                  # 本方案
├── README.md                # 结果总表（每轮更新）+ 怎么复现
├── env/
│   ├── setup_env.sh         # 装依赖（base 环境，--no-deps，不动 torch）
│   └── env_report.txt       # 版本、GPU 时钟、驱动等快照
├── common/                  # 所有轮次共用
│   ├── model_factory.py     # 构建 ViT-L/B EoMT（随机/真实权重，固定 seed）
│   ├── reference.py         # 生成并缓存 fp32 eager 参考输出
│   ├── bench.py             # CUDA event 计时：warmup、迭代、p50/p90/p99，分阶段 NVTX
│   ├── check.py             # 正确性：cos-sim / max err / mask IoU / score 差 / top-1 一致性
│   ├── roofline.py          # 每个算子的解析 FLOPs/字节 ÷ 实测时间 → 占峰值的百分比
│   ├── peak_bench.py        # 实测峰值：bf16/fp8/(nvfp4) GEMM、DRAM/L2 带宽、空 kernel/graph 启动延迟
│   ├── profile_nsys.sh      # 标准 nsys 采集 + stats 导出 CSV
│   ├── profile_ncu.sh       # ncu 命令（有计数器权限的机器上一键运行）
│   ├── kernel_static.sh     # ptxas -v / cuobjdump -sass / occupancy 静态分析
│   └── summarize.py         # 汇总各轮 results.json → results/summary.md + 图
├── round0_baseline/
├── round1_torch_cleanup/
├── round2_cuda_graph/
├── round3_fusion/
├── round4_attention/
├── round5_gemm_sm120/
├── round6_low_precision/
├── round7_head_postprocess/
├── round8_pipeline/
└── results/                 # summary.md / summary.csv / latency.png
```

每个 `roundK_*/` 的结构相同：
```
run.py          # 入口：--model vitl16|vitb16 --check --bench --profile
impl/           # 本轮新增的 kernel（.cu / TileLang .py / Triton .py）+ 单测
results.json    # 固定 schema（见 §5）
profile/        # *.nsys-rep、nsys stats CSV、静态分析输出
NOTES.md        # 做了什么 → 假设 → nsys 发现 → 结果 → 正确性 → 下一个瓶颈
```

原则：
- **累积式**：第 K 轮在第 K-1 轮之上叠加，从前面轮次 import 组件。每轮在 NOTES 里写清楚本轮单独贡献多少。
- **不改 `src/lightly_train`**：优化后的模型是 `infer-opt` 里的一个包装模块。它接收训练好的 `DINOv3EoMTInstanceSegmentation`，重排或折叠权重后自己实现 forward，基线始终可以原样跑。
- 临时产物放 `/root/autodl-tmp/infer-opt-tmp/`，nsys 报告比较大，只保留关键的几份。

---

## 3. 各轮计划

> 顺序按"收益大、成本低"优先；每轮开始前根据上一轮的 nsys 结果**确认瓶颈再动手**，必要时调整顺序。

### Round 0：基线与测量体系
- 装环境：`pip install -e . --no-deps`，加上 `opencv-python-headless`、lightning、pydantic 等缺的包，以及 `tilelang==0.1.14`。先确认 **TileLang 在 CUDA 13 + sm_120 上能编译运行**（这是风险点，退路见 §7）。
- 随机权重模型：显式调 `init_weights`（`bias_mask` 按 q/k/v 设为 1/0/1，LayerScale 的 γ 初始化），γ 设成接近真实训练后的量级（0.1~1），**断言输出没有 NaN**。
- 三种基线：`B0a` fp32 eager、`B0b` bf16 autocast eager（**和现有脚本一致，作为主基线**）、`B0c` 纯 bf16 权重 eager。
- 三个测量口径：
  1. **model-only**：GPU 上 1×3×512×512 → `forward_backend` 输出
  2. **postprocess**：logits → 原图分辨率（1920×1080）的 labels/masks/scores
  3. **end-to-end**：uint8 HWC 帧 → 结果回到 host
- 跑 `peak_bench.py` 实测：bf16 mm、FP8 `torch._scaled_mm`、DRAM/L2 带宽、空 kernel 启动延迟、graph launch 延迟。
- 跑 nsys 统计：每次 forward 的 kernel 数、GPU busy / wall time、CPU launch API 总耗时、Top-15 kernel，再结合解析 roofline 得到各 kernel 的效率。
- 产出：一张瓶颈排序表（验证 §1.2 的 1~7 条假设）和所有 fp32 参考输出。

### Round 1：PyTorch 层面清理（不写自定义算子）
- **纯 bf16 权重常驻**，去掉 autocast 每帧 cast 权重。
- **预计算常量**：RoPE sin/cos（输入固定 512×512，只算一次）、`LinearKMaskedBias` 的 masked bias、query 的 expand。
- **等价折叠权重**：
  - LayerScale 并进 proj/fc2（W′=diag(γ)W，b′=γ⊙b）
  - normalize 并进 patch_embed 卷积（resize 是线性的且权重和为 1，与逐通道仿射可交换；padding 改成用均值填充，就能与原来"先 normalize 再 pad 0"等价）
- SDPA 后端对比（flash / cuDNN / efficient），选最快的。
- 预处理放到 GPU：pinned 内存 → H2D → GPU resize/pad。
- 后处理 v1：**按类别分数精确剪枝**（§1.2 第 6 条），只对剩下的 query 做插值和打分。
- 预期：model-only 小到中等提升，**end-to-end 大幅提升**。

### Round 2：CUDA Graph（减少启动开销）+ torch.compile 对照
- 用静态输入输出 buffer，把 backbone+head 整段 `torch.cuda.graph` capture；后处理用固定 top-K（K 为剪枝后的上限）保证形状静态，也能进 graph。
- 对照组：eager / CUDA Graph / `torch.compile(mode="max-autotune-no-cudagraphs")` / `torch.compile(mode="max-autotune")`。
- nsys 对比：CPU launch 的空隙消失、GPU 空闲气泡、单次 graph launch 耗时。
- 预期：**ViT-B 提升最明显**；ViT-L 看 round0 是否 CPU-bound。

### Round 3：访存型算子融合（CUDA C++ / TileLang）
把每层 eager 的约 45 个 kernel 压到约 6 个：

| Kernel | 融合内容 | 实现 |
|---|---|---|
| K1 | 残差 add + LayerNorm（输出 bf16，warp-per-row，128-bit 向量化读写） | CUDA C++ |
| K2 | QKV GEMM + bias + **RoPE**（只作用于 patch token 的 q/k）+ 直接写成 head-major `[3,B,H,N,64]` | TileLang GEMM epilogue |
| K3 | attention | 先用 SDPA，round4 替换 |
| K4 | proj GEMM + bias(γ 已折叠) + 残差 add | TileLang |
| K5 | fc1 GEMM + bias + GELU(erf) | TileLang（与 cuBLASLt `GELU_BIAS` epilogue 对比） |
| K6 | fc2 GEMM + bias + 残差 add | TileLang |

- 每个 kernel 都有单测（与 torch 同 dtype 对比误差）和 microbench（与 torch 算子对比时间）。

### Round 4：attention（TileLang FlashAttention，针对 sm_120）
- head_dim=64，N=1029/1229，不需要 mask。
- 调 BLOCK_M/BLOCK_N/num_stages，按 wave 选配置：例如 BLOCK_M=64 → 16×17=272 个 CTA。尾部 5 行 / 77 行的 padding 单独处理。
- **把 RoPE 融到 Q/K 的加载 prologue 里**，K2 就不用再写回 rope 后的 q/k。softmax 用 exp2，把 scale·log2e 预先折进去。
- 对比 SDPA flash / cuDNN / TileLang，另外看 split-KV 能否把 SM 填满。

### Round 5：GEMM 针对 5090 调优
- 4 类 GEMM 形状各自选 tile：proj/fc2 用 **split-K / stream-K** 解决 0.42 wave 的问题；尝试 BLOCK_M=64 或 192 等配置，减少 M=1029 的尾部浪费。
- sm_120 能用的硬件特性：**TMA**（`cp.async.bulk.tensor`）+ mbarrier 多级流水、swizzle smem、`ldmatrix` + `mma.sync m16n8k16`、persistent kernel 加 tile scheduler。注意：sm_120 **没有** wgmma（Hopper 专有），也**没有** tcgen05/TMEM（sm_100 专有），tensor core 只能通过 warp 级 `mma.sync` 使用。
- 对照：cuBLAS（torch）、CUTLASS 4.2 自带的 sm120 kernel（examples/79_blackwell_geforce_gemm）、TileLang、手写 CUDA。
- 用 `ptxas -v` 查寄存器/spill/smem，用 SASS 确认用到了 HMMA / UTMALDG 等指令。

### Round 6：低精度（Blackwell 新特性：FP8 / MXFP8 / NVFP4）
- **FP8 E4M3 GEMM**：激活用 per-token 动态 scale，**在前一个 LN 或 GELU kernel 的 epilogue 里直接量化后写出**；权重用 per-channel 静态 scale；用 `mma.sync ... kind::f8f6f4`，FP32 累加。
- **MXFP8 块缩放**：每 1×32 个元素一个 UE8M0 scale，用 `kind::mxf8f6f4`，由硬件在 MMA 内完成缩放。
- **NVFP4**：每 16 个元素一个 E4M3 scale，再加 per-tensor FP32 scale，用 `kind::mxf4nvf4`。先用在 MLP 上（fc1/fc2 约占 2/3 的 FLOPs），做 W4A8 / W4A4。
- 敏感层（首尾若干层、head、attention 的 softmax）保留 bf16；FP8 attention（SageAttention 式的 K smoothing）作为可选项。
- **精度门槛**：与 fp32 参考比，mask IoU ≥ 0.98，score 差 ≤ 0.01，类别一致。随机权重下只作初步结论，**最终要等真实权重和真实帧复核**。

### Round 7：Head 与后处理融合
- **ConvTranspose2d k2s2 本质上是一个 GEMM**：[HW,C]×[C,4C] 再做 pixel-shuffle。bias + GELU + pixel-shuffle 的散写都放进 GEMM epilogue。
- 整个 head 改用 channels-last，去掉 permute；DWConv3×3 + LayerNorm2D 融成一个 CUDA kernel（NHWC，一个 CTA 处理若干像素，一整行 C 放在寄存器或 smem 里）。
- einsum 改成 GEMM，而且**只算剪枝后剩下的 query**。
- **融合后处理 kernel**（CUDA C++）：输入低分辨率 logits（Q_kept×128×128），在 kernel 里现算两次双线性插值（128→512，裁剪，再到原图尺寸）、sigmoid 和 >0，每个 query 做规约得到 mask_score。然后只为最终保留的 query 写出 uint8 mask，整个过程不在显存里物化 200×H×W 的 fp32 张量。
- 只把最终 mask 拷回 host（可选 bit-pack）。

### Round 8：系统级流水线
- 解码和推理重叠：解码线程（cv2，或 NVDEC 可选）→ pinned 环形缓冲 → copy stream 和 compute stream 通过 event 串联；每帧的整段 GPU 工作（预处理 → 模型 → 后处理）放进一个 graph。
- **PDL（Programmatic Dependent Launch，`griddepcontrol`）**：让下一个 kernel 的 prologue 和上一个 kernel 的尾部重叠，batch=1 时 kernel 尾部占比高，收益明显。
- 探索项：用 `cp.async.bulk.prefetch.L2` 把下一层权重（约 25 MB）预取进 L2；另加吞吐模式 B=2/4 对比。

---

## 4. 性能分析方法（ncu 不可用时的替代方案）

### 4.1 nsys（主力）
```bash
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o roundK/profile/<model>_<tag> python roundK/run.py --model vitl16 --profile
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum,cuda_gpu_trace \
  --format csv -o roundK/profile/<model>_<tag> roundK/profile/<model>_<tag>.nsys-rep
```
- NVTX 分段：`preprocess / patch_embed / block_00..23 / head / postprocess`
- 每轮都记录：kernel 数/forward、GPU busy vs wall（空闲占比）、`cudaLaunchKernel` 总时间、各阶段 GPU 时间、Top-15 kernel

### 4.2 代替 ncu 的手段
- **解析 roofline**：`roofline.py` 按算子形状算 FLOPs 和最少访存字节，用 nsys 得到的 kernel 时间换算出实际 TFLOPS 和 GB/s，与 `peak_bench.py` 的**实测峰值**比较，得到效率百分比。
- **静态分析**：`nvcc -Xptxas -v`（寄存器 / spill / smem）；`cuobjdump -sass`（确认 HMMA/QMMA、LDGSTS/UTMALDG 等指令）；`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算理论 occupancy 和 wave 数；TileLang 导出生成的 CUDA 源码检查流水和 layout。
- **受控实验**：扫 tile 参数，把"改动 X 前后"的时间差作为因果证据。
- **ncu 脚本预先写好**（`common/profile_ncu.sh`）：`--set full --import-source yes --nvtx --nvtx-include "block_05/" -k regex:<kernel> --launch-skip 50 --launch-count 1`，覆盖 SOL / Roofline / MemoryWorkload / WarpStateStats / SourceCounters。拿到有权限的机器后一键补跑。

---

## 5. 结果记录与对比

- 计时规范：CUDA event；warmup 50 次、测 200 次、重复 3 轮，报 **p50 / p90 / 均值±std**。每次跑之前检查 GPU 没有被其他进程占用（**注意：当时 nvidia-smi 显示 Util 100%、显存 0 MiB，需要排查**），并记录时钟和温度。
- 每轮 `results.json` 的 schema：
```json
{"round": "round2_cuda_graph", "model": "vitl16", "dtype": "bf16", "batch": 1,
 "latency_ms": {"model_only": {"p50":0,"p90":0,"mean":0,"std":0},
                "postprocess": {...}, "e2e": {...}},
 "fps_e2e": 0, "kernels_per_forward": 0, "gpu_busy_ratio": 0, "peak_mem_mb": 0,
 "accuracy": {"class_logits_cos":0, "mask_logits_cos":0, "mask_iou_min":0, "score_max_abs_diff":0},
 "env": {"torch":"2.12.1+cu130","cuda":"13.0","gpu_clock_mhz":0}}
```
- `summarize.py` 生成 `results/summary.md`：

| Round | 变更 | ViT-L model-only p50 | 较上轮 | 较 R0 | ViT-L e2e FPS | kernels/fwd | mask IoU | ViT-B model-only p50 |
|---|---|---|---|---|---|---|---|---|

  同时生成 `latency.png`（各轮 × 各阶段的堆叠柱状图）。

## 6. 正确性标准
- **A 级（单个 kernel）**：与 torch 同 dtype 同输入比，bf16 要求 atol/rtol ≈ 1e-2，并报告 fp32 累加的最大误差。
- **B 级（整模型）**：与 fp32 eager 参考比（固定随机输入，有真实帧后加真实帧）。bf16 轮次要求 logits cos-sim ≥ 0.999、mask IoU ≥ 0.99、score 差 ≤ 5e-3；FP8/FP4 轮次见 Round 6 的门槛。
- 只有通过 B 级的轮次才进入结果总表；精度下降的优化单独标注。

## 7. 风险与退路
| 风险 | 退路 |
|---|---|
| TileLang 0.1.14 对 CUDA 13 / sm_120 支持不完整 | 同类 kernel 改用 Triton 3.7.1（支持 sm_120，`tl.dot_scaled` 可做 MX 格式）或 CUTLASS 4.2 / 手写 CUDA；在 NOTES 里说明 |
| ncu 始终没权限 | 按 §4.2 执行；ncu 脚本保留，换机器后补跑 |
| FP8/FP4 精度不达标 | 只把量化用在 MLP 或对精度不敏感的层；或者只保留 FP8 |
| 随机权重下的数值不代表真实分布 | 性能结论不受影响；精度结论标为"待真实权重复核" |
| 容器里不能锁 GPU 时钟 | 多次重复取中位数，并记录时钟 |

## 8. 初始实施顺序（历史）
方案确认后从 **Round 0** 开始：装环境（base，`--no-deps`）→ 验证 TileLang 能在 sm_120 上跑 → 搭 `common/` 测量体系 → 跑出 ViT-L/B 基线和 nsys 报告 → 用实测数据确认 §1.2 的假设，再决定 Round 1 以后的先后顺序。

## 9. 执行状态（2026-10-02）

§0-8 是初版研究计划，顶部 v2 修订和本节记录实际执行结果。真实 checkpoint 是 **语义分割**，评测视频每帧切为两个 crop；下表的精度门槛以真实权重上的 pooled head IoU ≥ 0.95、逐帧 confidence 相对误差 ≤ 5% 为准。原 §1.2/§3 针对实例分割的 query 剪枝和 200 个 query 后处理不再适用。

| 轮次 | 状态 | 实测结论 / 采用方式 |
|---|---|---|
| R0-R2 | 完成 | 真实权重/视频基线、峰值与 nsys、PyTorch 清理、CUDA Graph 和 torch.compile 对照均已记录。整帧 graph 最终在 R9 集成。 |
| R3-R4 | 完成 | 融合 block 保留 fp32 残差；TileLang attention 已实现。R9 配对中 `opt_tl_rope` 比 `opt` 快约 9%，但 `fused_tl` 未稳定快于 `fused`。 |
| R5 | 完成并采用 v1 | 16 个真实形状的调参表、split-K 工作区生命周期修复和静态检查已完成；交错模型 graph 快 4.3%（L）/6.7%（B）。v2/PDL 与 L2 预取未获稳定收益，保留实验实现。 |
| R6 | 完成筛选 | 逐行 FP8、真实 MXFP8 与 NVFP4 假量化模型级精度检查已完成。MXFP8 在 ViT-B 14 units 上比同配置逐行 FP8 快约 3%，但精度余量缩小；NVFP4 可覆盖的层太少，故默认不量化。 |
| R7 | 完成目标形状 | GPU 前后处理与融合 head 已上线；宽/竖图切块和合并与基线逐项对齐。内核仍固定 2 个已知类 + void。 |
| R8-R9 | 完成主流水线 | pinned ring、3 stream、整帧单图已集成。双 crop 宽图默认 R5 v1；同轮整视频为 94.7 FPS（L）/214.1 FPS（B），比原脚本约 6.3×/13.0×。竖图或其他 crop 数自动回退 `lightly`；融合 backbone 在旋转真帧上未达 head IoU 0.95。 |

仍未覆盖的扩展项：ncu 计数器权限仍由宿主机限制；CUTLASS example 79 的同 epilogue 公平对照、多路视频、batch≠2 的调参、K≠2 的融合后处理，以及完整竖图视频的 fp32 精度门槛。NVFP4 真实接入和 fused+FP8 在当前精度/收益筛选下未采用。具体复现命令、原始结果及限制见各轮 `NOTES.md` 和 `results/summary.md`。
