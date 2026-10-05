# Round 9：集成骨架（整帧单图 + 可插拔 model_fn）

## 1. 设计
- **整帧单图**（`video_infer.py::VideoPipeline`）：每个 ring 槽位一张 CUDA Graph，内容 = `prep kernel → model_fn(static bf16 [2,3,512,512]) → post_a → memset → post_b`。各槽位的 graph 共享同一个 mempool（串行 replay，安全），只有输入（该槽位的 GPU 帧缓冲）和输出（该槽位的 mask/stats）按槽位区分。
- H2D / D2H 仍在独立 stream（h2d / d2h）上，用 event 与 compute stream 串联；解码线程直接写 pinned ring；result 线程 `event.synchronize()` 后取 mask 与 confidence（同 Round 8）。
- **model_fn 接口**：`model_fn(static_input_bf16) -> (mask_logits_lr, class_logits)`，可 capture，两个输出 dtype 相同，post kernel 直接读它们（无拷贝）。实现在 `impl/model_impls.py::make_model_fn(name, model, impl, fp8)`：
  - `lightly`：基线 lightly 模型 + bf16 autocast（`cache_enabled=False`）+ Round 7 融合 head（P3）；`--fp8` 叠加 Round 6 的真实 FP8（ViT-L 6 unit、ViT-B 14 unit）。
  - `opt` / `opt_gemmhead`：graphopt 的 `round1_torch_cleanup/opt_model.py::OptEoMT.forward_lowres`（只读 import，未改动）。
  - `fused`：blockfuse 的 `round3_fusion/impl/ops.py::fused_block`（fp32 残差流，`prepare_block_weights`，按 `test_model_fused.py` 的方式组装）+ 融合 head。
  - `opt_tl` / `opt_tl_rope`：Round 1 OptEoMT 接 Round 4 TileLang attention（独立 RoPE / 融合 RoPE）。
  - `fused_tl`：Round 3 融合 GEMM 的 `[3,B,N,H,64]` QKV 物理布局直接接 Round 4 TileLang attention。
  - `r5_v1`：Round 5 tuned GEMM + LayerNorm block（7 kernel/层）+ 融合 head。
- CLI：`python video_infer.py --model X.pth --video-dir D --out-dir O [--model-impl NAME] [--fp8] [--slots N]`。`--video-dir` 可指定单个视频或递归扫描目录；扫描时排除 `--out-dir` 下的旧输出。默认双 crop 宽图使用 `r5_v1`；竖图或其他 crop 数统一使用 `lightly`（即使指定其他实现也回退，并打印原因和实际实现；非双 crop 宽图的 `--fp8` 报错）。最多支持 8 个 crop。所有需要的 model_fn 在捕获 graph 前构建，之后按尺寸缓存 pipeline。输出 overlay 视频、CSV、best frame/mask、summary.json（含每视频的 `model_impl`）。
- FP8 只接在 `lightly` 上：`fused_block` 的 GEMM 是 TileLang 自带 epilogue（bias/GELU/残差），没有 FP8 版本，`fused+fp8` 未接（接口 `make_model_fn` 会直接报错）。

## 2. 原始 Round 9 测量（历史对照：整段 1.mp4，126 帧，循环 5 遍稳态；GPU 锁内；`bench9.py`）
基线 = 原脚本 decode + predict_head：ViT-L 15.9 FPS，ViT-B 16.4 FPS。服务延迟 = 提交→host 结果（inflight=1）。
本表是接入 Round 4/5 前的历史结果；当前推荐依据见下方同轮配对表。

| 模型 | model_fn 实现 | 吞吐 FPS（inflight=2） | 加速 | 服务延迟 p50 / p90 (ms) | 126 帧 vs 原脚本：mask 一致率 min / head IoU min / pooled / conf 差 max | 8 帧 gate（IoU / conf max） |
|---|---|---|---|---|---|---|
| ViT-L | lightly + head | 64.3 | 4.0× | 15.0 / 15.0 | 0.99999 / 0.984 / 0.9942 / 0.0125 | 0.9930 / 0.44% |
| ViT-L | lightly + head + FP8(6) | 65.8 | 4.1× | 14.7 / 14.7 | 0.99995 / 0.941 / 0.9895 / 0.0192 | 0.9888 / 1.60% |
| ViT-L | opt（graphopt） | 77.6 | 4.9× | 12.6 / 12.6 | 0.99999 / 0.983 / 0.9931 / 0.0225 | 0.9921 / 2.71% |
| ViT-L | opt + gemm head | 75.7 | 4.8× | 12.65 / 12.7 | 同上 | 0.9920 / 2.71% |
| ViT-L | fused（blockfuse）+ head | 91.1 | 5.7× | 10.5 / 10.5 | 0.99998 / 0.969 / 0.9923 / 0.0156 | 0.9937 / 2.10% |
| ViT-B | lightly + head | 153.6 | 9.4× | 5.84 / 5.85 | 0.99998 / 0.978 / 0.9927 / 0.0033 | 0.9921 / 0.11% |
| ViT-B | lightly + head + FP8(14) | 162.4 | 9.9× | 5.47 / 5.48 | 0.99995 / 0.922 / 0.9799 / 0.0128 | 0.9825 / 0.53% |
| ViT-B | opt（graphopt） | 176.2 | 10.8× | 4.94 / 4.96 | 0.99998 / 0.976 / 0.9910 / 0.0032 | 0.9930 / 0.15% |
| ViT-B | opt + gemm head | 178.4 | 10.9× | 4.94 / 4.95 | 同上 | 0.9932 / 0.12% |
| ViT-B | fused + head | 188.4 | 11.5× | 4.24 / 4.25 | 0.99998 / 0.977 / 0.9923 / 0.0036 | 0.9943 / 0.21% |

- **所有 10 个变体的 8 帧 fp32 gate 全部通过**；graph 版与非 graph 版的整条 pipeline 输出在 126 帧上 mask 逐像素完全一致（confidence 差 0.0）；model_fn 单独在 graph replay vs eager 下 logits 逐元素完全一致（max_abs 0.0，含 TileLang 的 fused、OptEoMT、FP8）——TileLang kernel 的输出在 graph 私有 mempool 上工作正常。
- inflight=1/2/3 的 FPS 差别很小（fused ViT-L 87.9 / 91.1 / 89.5；ViT-B 187.9 / 188.4 / 196.7），流水线深度 2 已足够。
- 与 Round 8（prep/post 分别 graph，model eager）对比：ViT-L 57.3 → 91.1 FPS（fused），ViT-B 118 → 188 FPS。
- FP8 的收益与 Round 6 一致但很小（ViT-L +2%，ViT-B +6%）；126 帧 head IoU 最低帧降到 0.94/0.92，需要接受。ViT-L 的 fused 与 FP8 不能叠加（见上）。
- ViT-L 的 opt/fused 的 conf 误差在 gate 内（2.7%/2.1%）但比 lightly（0.44%）大，这是它们各自 Round 的已知代价；126 帧 conf 差最大 0.0225/0.0156（对原脚本）。
- overlay + 写视频时瓶颈仍是 CPU 端 overlay/编码（约 34 FPS，见 Round 8），与 model_fn 无关。

### 后续组合验证（同一命令内配对，126 帧循环 3 遍；独占 GPU 锁）
| 模型 | 对照 → 新实现 | 对照 FPS → 新实现 FPS | 变化 | 新实现 gate：pooled IoU / conf 相对误差 max |
|---|---|---:|---:|---:|
| ViT-L | `opt` → `opt_tl_rope` | 76.90 → 83.98 | +9.2% | 0.99373 / 2.13% |
| ViT-L | `fused` → `fused_tl` | 91.96 → 86.25 | -6.2% | 0.99572 / 1.08% |
| ViT-L | `fused` → `r5_v1` | 92.45 → 94.65 | +2.4% | 0.99459 / 0.13% |
| ViT-B | `opt` → `opt_tl_rope` | 181.98 → 197.60 | +8.6% | 0.99205 / 0.14% |
| ViT-B | `fused` → `fused_tl` | 208.30 → 209.03 | +0.4% | 0.99341 / 0.12% |
| ViT-B | `fused` → `r5_v1` | 208.30 → 214.10 | +2.8% | 0.99426 / 0.21% |

- 以上所有新实现：model_fn graph replay 与 eager logits 逐元素相同（max_abs 0）；整段 126 帧 graph 与非 graph mask 完全相同、confidence 差 0；8 帧 fp32 gate 通过。`opt_tl` plain 另在 ViT-L 完整跑过 126 帧（2 遍，81.31 FPS，gate 通过），但没有相邻的同轮 `opt` 性能对照。
- ViT-L 的 `fused_tl` 首次单独跑到 97.25 FPS，同轮配对为 86.25 FPS，存在明显运行间漂移；因此不以单次最高值判断它快于 `fused`。`fused_tl` 保留为实验接口；推荐 `r5_v1` 只用于已验证的宽图 2 crop/batch=2 路径。
- 复现：`python bench9.py --model vitl16 --variants opt,opt_tl_rope,fused,fused_tl --loops 3`；`python bench9.py --model vitb16 --variants opt,opt_tl_rope,fused,fused_tl,r5_v1 --loops 3`；`python bench9.py --model vitl16 --variants fused,r5_v1 --loops 3`。各模型/实现的完整数字保存在 `results.json`。

### 竖图入口验证
- 对原视频首帧旋转、编码成 1440×1900 的单帧 mp4；`python check_tall_graph.py --model vitb16 --video /root/autodl-tmp/infer-opt-tmp/round9_tall_one_frame.mp4`：`lightly` 的 graph/non-graph mask 和 confidence 完全相同；相对原脚本 bf16 输出，head IoU **0.98212**（参考头部 1270 px，graph 1279 px）、置信度相对误差 **0.44%**。同帧 `fused` / `r5_v1` 均为 IoU **0.87520**（1119 px），`opt` 为 **0.85714**，故竖图默认回退 `lightly`，不能只按全图像素一致率判断（`fused` 仍有 0.99994 像素一致率）。这是对 bf16 原脚本的单帧验证，不是完整竖图视频的 fp32 gate。
- 直接 CLI 验证 `a_wide.mp4 → b_tall.mp4 → c_wide.mp4`：`summary.json` 实际实现为 `r5_v1, lightly, r5_v1`；两段相同宽图的 best mask PNG 逐字节一致，置信度完全一致；竖图输出 overlay/CSV/best mask 均成功。图捕获后不会因不同方向模型函数的建立而改变已缓存宽图结果。
- 解码打不开或零帧、写出/消费线程异常会向调用方报告，线程能够退出；`python -m unittest -v test_video_entry` 的 4 个 CPU-only 回归测试通过（含非双 crop 形状回退）。`--precision` 仅接受 `bf16-mixed`，无效输入和 slots 会在启动前报错。

## 3. nsys（`profile/*_analysis.txt`，捕获 1 遍视频 126 帧；`analyze9.py` 读 sqlite）
| | cudaGraphLaunch / 帧 | 其他 launch API / 帧 | GPU kernel / 帧 | kernel 忙 / 空闲（ms/帧） | GPU 忙碌率 |
|---|---|---|---|---|---|
| ViT-L fused | **1.00** | 0（只剩 3 次 memcpyAsync + event 调用） | 204 | 10.03 / **0.10（1.0%）** | 99.0% |
| ViT-L lightly | 1.00 | 0 | 1284 | 14.53 / 0.77（5.0%） | 95.0% |
| ViT-B fused | 1.00 | 0 | 120 | 3.91 / 0.23（5.6%） | 94.4% |
- 每帧 GPU 上确实只剩 **1 次 graph launch**（+ H2D 1 次 0.148 ms、D2H 2 次 0.049 ms，在别的 stream，与 compute 重叠）；graph 内 kernel 间隙中位数 0.45 µs，最大 5 µs。
- ViT-L fused：GPU 每帧 10.13 ms（≈98.7 FPS 的 GPU 上限），实测 91 FPS，差距来自 host 线程（解码/result）与 GIL，nsys 窗口内无 GPU 空洞。ViT-B fused 空闲 5.6% 主要是启动/结束的边界。
- ViT-B fused 的 “groups separated by >20 µs = 2” 是窗口起始的 22 ms 的空隙（cudaProfilerStart 前后），不是帧间空洞。

## 4. 接入指南 / 遗留
- 换实现：`make_model_fn` 里加一个分支，返回 `model_fn(x_bf16)->(ml, cl)` 即可（要求 capture-safe，输出同 dtype，Q≤128，K+1=3）；`bench9.py --variants` 会自动测吞吐/延迟/精度并做 graph-vs-eager 逐元素检查。
- graphopt / blockfuse 在本轮读取时的版本（opt_model.py 15:30 修改；round3 fused_block）通过了 gate；它们之后若改动，重跑 `python bench9.py --model vitl16 --variants opt,fused` 即可。
- 未做：fused + FP8（需要 FP8 版 TileLang GEMM epilogue）；多路并发视频；竖图完整视频 fp32 gate；`fused` / `r5_v1` 在 batch≠2 时要重新调 tile。`opt_tl` plain 只验证 ViT-L，`fused_tl` 没有稳定性能收益。
- host 侧 ~10% 的 GPU 上限差距（ViT-L fused：91 vs 98.7 FPS）可以通过把 result 线程改成 C++ / 减少 Python 每帧调用来压缩，未做。

## 5. 文件
`video_infer.py`（集成版流水线 + CLI）、`impl/model_impls.py`、`bench9.py`、`check_tall_graph.py`、`test_video_entry.py`、`analyze9.py`、`results.json`（每个 variant 含 latency_ms / fps_e2e / accuracy_vs_fp32 / video）、`profile/`。依赖 round7 / round6 / round5 / round4 / round3 / round1 的目录。
