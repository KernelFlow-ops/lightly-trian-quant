# Round 7：前处理 / 后处理 / head 融合（postfuse）

测试数据：真实视频 1900×1440（8 帧），短边缩到 512 → 675×512 → 2 个 crop（起点 0 / 163）。所有 kernel 按 meta 参数化（不写死尺寸），
2 个已知类别（+void）、宽图（tall 图未实现，会 assert）。

## 1. 目标与结论
基线的 prep（5.8 ms）和 post（35 ms）几乎全是 CPU 开销（cvtColor→PIL→拷贝；numpy argmax/映射/mean；16 MB 的 pageable D2H），GPU 忙碌率 7%。
本轮用 3 个 kernel 替换：P1 prep 1 个 kernel；P2 post 2 个 kernel，回传只有 uint8 mask（2.7 MB）和 4 个 int64 标量。
P3 把 head 的 GELU / 像素重排 / DWConv / LN2D 合成 1 个 kernel。

## 2. 设计
### P1 `prep_kernel`（impl/postfuse_ext.cu）
- host：frame → 持久 pinned buffer（memcpy）→ 异步 H2D（`copy_(non_blocking)`）→ 1 个 kernel。
- 每个输出像素（crop c, y, x）：`X = x + start_c`，对 5×5（实际 K=6 padding）个源像素做加权，权重是**预计算表**
  （`postfuse.aa_table`，用 float32 逐步复刻 ATen `_compute_indices_min_size_weights_aa`：triangle filter、support=scale、按和归一化），
  同时 BGR→RGB、`/255`、`-mean`、`/std`（与基线相同的运算，权重和为 1 所以与先 normalize 后 resize 等价）。
- 输出直接是 (2,3,512,512)，dtype 为 fp32 / bf16 / fp16。表和缓冲区按帧尺寸缓存。
- 表 vs `F.interpolate(antialias=True)`：max_abs 1e-5（随机输入，cos 1-2e-12）。

### P2 `post_a_kernel` + `post_b_kernel`
- A：线程 = 一个 untiled 像素 (Y∈512, X∈675)。对覆盖它的每个 crop，从 (100,128,128) 的低分辨率 logits 现算 128→512 双线性
  （align_corners=False，`src=max((dst+.5)*.25-.5,0)`，与 ATen 同公式），sigmoid，与 smem 里的 softmax(class) 点积（只算 2 个非 void 类；
  softmax 的分母包含 void），重叠区 `sum/count`。输出 (2,512,675) fp32（2.8 MB，L2 内）。
- B：第二次双线性 512×675→1440×1900（独立的第二步，不与第一步合并），argmax（平局取 0，同 numpy），
  class id 映射，uint8 mask，block 规约 head 像素数 / head 分数和（double 原子加）/ 全局 max（有序 uint 键 atomicMax）。
  `return_logits=True` 调试开关时额外写 (2,H,W) fp32 logits。
- host：mask→pinned、stats→pinned，一次 stream sync；confidence = sum/count，count=0 时取 max。返回的 mask 是 pinned buffer 的 numpy view，**下次调用前有效**
  （视频脚本里 `mask.copy()` 只在 best 帧发生，不影响）。

### P3 `dw_ln_kernel`（impl/head_ext.cu，impl/head.py）
- ConvT(k2,s2) = GEMM `[B·Hs·Ws, C] @ [C, 4C]`（列 = (ki,kj,co)），用 torch.mm（cuBLAS/cutlass）。
- 1 个融合 kernel：读 pre-shuffle 的 GEMM 输出，+bias、erf-GELU、像素重排（读地址直接算）、DWConv3×3（zero pad）、LN over C（fp32 统计，affine）→ NHWC bf16。
  block = 2×8 输出像素 × 全部 C 通道，4×10 halo 以 bf16 放 smem（ViT-L 80 KB），512 线程，一个 warp 一个像素，lane 拥有 C/32 个通道。
- 最后的 einsum 变成 `bmm(mask_head(q), U^T)`（NHWC，无布局转换）。
- 通过 `model._predict = fast_head.predict` 挂接（不改 src）；仅适用于 bf16（autocast 或 pure）推理。

## 3. 正确性
### P1（`test_prepost.py`，8 帧，vs `mf.base_prep`）
| dtype | max_abs | mean_abs | cos |
|---|---|---|---|
| fp32 | 3.6e-5 ~ 4.9e-5 | 1e-6 | 1.000000 |
| bf16 | 0.03125（1 个 bf16 ulp@值域 4） | 0.005 | 0.99998 |
- crop 起点、resized 尺寸与 meta 完全一致（assert）。
- bf16 的差异来自**基线**：基线 bf16 模型先在原分辨率做 bf16 的 `/255`、`-mean`、`/std`（每步舍入）再 resize，我们是 fp32 算完再舍入一次，更准。
### P2（同一个低分辨率输入，vs 基线 `F.interpolate→512→to_per_pixel_logits→untile→interpolate→numpy`）
| 输入精度 | logits max_abs | mask 一致率 | head IoU | confidence 差 |
|---|---|---|---|---|
| fp32 输入（隔离 kernel 误差） | ≤3.6e-7，cos 1.0 | 1.000000（8 帧全部） | 1.000000 | ≤9e-8 |
| bf16-autocast 输入（vitl16, 8 帧） | 1.9e-3 ~ 3.1e-3（基线 bf16 舍入噪声） | ≥0.999999 | 0.9995 ~ 1.0 | 0.9e-4 ~ 8.2e-4 |
| bf16-autocast 输入（vitb16, 8 帧） | 2.2e-3 ~ 5.6e-3 | ≥0.999999 | 0.9983 ~ 1.0 | 0.3e-3 ~ 3.1e-3 |
（完整日志：`test_prepost_vitl16.log` / `test_prepost_vitb16.log`。）
### P3（`test_head.py`，真实 frame0 的 backbone.norm 输出，vs `model._predict`，autocast）
| | mask_logits max_abs | mean_abs | cos | logit>0 一致率 | class_logits |
|---|---|---|---|---|---|
| vitl16 | 1.0（|logit| 均值 75，bf16 1~2 ulp） | 0.076 | 0.999997 | 0.9999993 | 完全一致 |
| vitb16 | 0.5（|logit| 均值 32） | 0.029 | 0.999997 | 0.9999963 | 完全一致 |

### 与 fp32 参考的整体精度门槛（`accuracy_vs_fp32`，8 帧）
| 模型 / 变体 | pooled head IoU | 最差单帧 IoU | 最大 conf 相对误差 | gate |
|---|---|---|---|---|
| vitl16 autocast 基线（R0） | 0.9931 | 0.9898 | 1.11% | pass |
| vitl16 autocast + fused-prepost | 0.9933 | 0.9905 | 0.45% | pass |
| vitl16 autocast + fused-prepost + fused-head | 0.9930 | 0.9909 | 0.44% | pass |
| vitb16 autocast 基线（R0） | 0.9926 | 0.9870 | 0.38% | pass |
| vitb16 autocast + fused-prepost | 0.9923 | 0.9871 | 0.09% | pass |
| vitb16 autocast + fused-prepost + fused-head | 0.9921 | 0.9859 | 0.11% | pass |
| vitl16 bf16-pure（R0） | 0.9897 | 0.9508 | 4.92% | pass（贴线） |
| vitl16 bf16-pure + fused-prepost | 0.9837 | 0.9149 | 12.4% | **fail** |
| vitb16 bf16-pure + fused-prepost | 0.9904 | 0.9837 | 0.27% | pass |

**vitl16 bf16-pure 的失败不是 kernel 的问题**：`diag_pure.py` 做 prep/post 2×2 交换，base-prep+fast-post = 0.9896/0.9500/5.2%（与基线 post 一致），
fast-prep+base-post = 0.9837/0.9141/12.8%（与 fast-post 一致）。即 post 等价，差异来自输入：我们的 bf16 输入更"精确"，但 ViT-L 纯 bf16 权重对输入舍入模式很敏感，
基线的"带噪输入"恰好落在门槛边上。结论：ViT-L 不能用 pure-bf16（基线本身已贴线），autocast 才是安全配置。

## 4. 性能（wall p50 ms，`bench.time_fn`，GPU 锁内，frame0，100 iters×3 repeats）
| 模型 / 变体 | prep | model | post | e2e | FPS | kernel 数(e2e) |
|---|---|---|---|---|---|---|
| vitl16 autocast 基线 R0 | 5.77 | 17.05 | 35.36 | 60.65 | 16.5 | 1324 |
| vitl16 autocast + fused-prepost | **0.70** | 16.93 | **0.22** | **17.81** | **56.1** | 1303 |
| vitl16 autocast + fused-prepost + fused-head | 0.71 | 16.46 | 0.22 | **17.38** | **57.5** | 1285 |
| vitb16 autocast 基线 R0 | 5.76 | 9.62 | 34.84 | 51.71 | 19.3 | 704 |
| vitb16 autocast + fused-prepost | 0.73 | 9.49 | 0.20 | 10.45 | 95.7 | 683 |
| vitb16 autocast + fused-prepost + fused-head | 0.71 | 9.16 | 0.20 | **10.07** | **99.3** | 661 |
| vitl16 bf16-pure + fused-prepost（gate 不过） | 0.71 | 15.05 | 0.23 | 15.90 | 62.9 | |
| vitb16 bf16-pure + fused-prepost | 0.71 | 7.95 | 0.20 | 8.86 | 112.8 | |
（"model" 一栏：基线含 mask_logits 128→512 的插值（(2,100,512,512) bf16，100 MB），我们不含；因此 model 略降。）

nsys（vitl16 autocast + fused-prepost，`profile/vitl16_bf16autocast_fused_e2e_summary.md`，每帧）：
- prep：1 个 kernel 17.5 µs + H2D 0.148 ms；wall 0.7 ms 主要是 host 的 6 MB pinned memcpy + 调用开销（GPU 忙碌 97%）。
- post：`post_a` 116 µs + `post_b` 15.6 µs（+ D2H 0.049 ms）；wall 0.22 ms。
- 基线 post 是 35 ms 的 CPU 瓶颈，现在整帧的后处理只占 GPU 0.19 ms。
- model 段仍是 1300 个 kernel / 15.1 ms kernel 时间、GPU 忙碌 74%：**下一个瓶颈就是 backbone 本身（Round 1~6 的内容）**。

P3 microbench（`_predict` 全部，含 class_head/mask_head，autocast，GPU p50）：vitl16 1.457 → 0.958 ms（-0.50 ms）；vitb16 1.061 → 0.643 ms（-0.42 ms）。
dw_ln kernel 从 256 线程改成 512 线程后 1.083→0.958 ms。剩余时间主要在 2 个 cuBLAS GEMM 和 2 次 dw_ln（尚未逐 kernel 精确拆分，torch profiler 上 GEMM 约 0.55 ms、dw_ln 约 0.39 ms，均为 256 线程版本的数字）。

## 5. 遗留 / 下一步
- prep 的 0.7 ms 主要是 host 端 memcpy 到 pinned；若解码端能直接把帧写入 pinned buffer（或 cudaHostRegister 解码缓冲），可再省 ~0.5 ms。
- P3：dw_ln 仍是 2.5× halo 冗余 + 1 block/SM；GEMM 与 GELU 之前的 bias 没并进 epilogue（用 cuBLAS）；可用 TileLang/CUTLASS epilogue 进一步融合。收益上限有限（整个 head 现在约 0.96 ms，占 e2e 5%）。
- 后处理精度：kernel 内全 fp32（比基线 bf16 更准），因此与基线逐像素不是位一致（mask 一致率 ≥0.999999）。
- 只支持 K=2 类 + void；竖图（h>w）现已支持按纵向切块和合并，K≠2 仍需扩展。
- 与 B 的融合 backbone 配合：model_lowres 只取 `forward_train`，B 的 kernel 接入后 e2e 会再降；P3 通过 `model._predict` 挂接，与替换 block 不冲突。

### 竖图补充验证
- `python test_prepost.py --model vitb16 --frames 1 --tall` 使用原视频帧旋转后的 1440×1900 图，和模型自带的纵向切块路径逐项对照。crop 起点及缩放尺寸一致；fp32 prep max_abs 7.2e-5，fp32 post logits max_abs 5.96e-8，mask 完全一致，confidence 差 2.34e-9。bf16-mixed 的 mask 完全一致，confidence 差 0.00175。
- 宽图 `--frames 1` 回归：fp32 post logits max_abs 1.19e-7，mask 完全一致；bf16-mixed head IoU 0.999568，confidence 差 2.91e-4。正确性脚本现对核心误差界和 mask 一致率执行断言。

## 6. 文件
- `impl/postfuse_ext.cu`, `impl/postfuse.py`（P1/P2）；`impl/head_ext.cu`, `impl/head.py`（P3）
- `pipeline.py`（`init / fast_prep / model_lowres / fast_post / fast_e2e`）；`run_r7.py`（bench/acc/profile，`--head` 开 P3）
- `test_prepost.py`（P1/P2 正确性）、`test_head.py`（P3 正确性 + microbench）、`diag_pure.py`（bf16-pure 2×2 诊断）
- `results.json`、`results_head.json`、`test_prepost_*.log`、`profile/`
