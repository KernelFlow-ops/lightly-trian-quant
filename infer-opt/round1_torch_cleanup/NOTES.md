# Round 1：PyTorch 层面清理（OptEoMT）

文件：`opt_model.py`（`OptEoMT`，接口 `forward_lowres` / `forward_backend`）、`harness.py`、`run.py`、`test_opt.py`、`results.json`、`profile/`。
计时口径同 round0（wall p50，2 crop batch=2；e2e = 基线 prep + 我们的 model + 基线 post；prep/post 仍在 bf16 autocast 下，**我们的 model 在 autocast 之外**）。表中为"最佳 repeat 的 wall p50"，因为其他 worker 会在锁外跑正确性任务，单次 p50 偶尔被污染（第一版数据里 model 出现过 24 ms，已作废，见 round2 的 old_contaminated/）。

## 做了什么
1. 权重 bf16 常驻，无 autocast；2. RoPE sin/cos 预计算（fp32）；3. k-masked bias 预计算；4. LayerScale 折进 proj/fc2（fp32 折叠后再转 bf16）；
5. RoPE 只用 4 个 kernel：`roll` + `mul` + `addcmul`（fp32，与原实现同精度）+ `copy_` 就地写回 qkv，q/k/v 是 qkv 的视图（无 cat/neg/前缀拷贝）；
6. SDPA 后端可选；7. head 可选 channels_last / ConvTranspose 改 GEMM。
8. **残差流用 fp32**（LN 在 fp32 里算，输出 cast 到 bf16 喂 GEMM），与基线 autocast 的数值行为一致。bf16 残差版本（`res_bf16`）少 51 个 kernel、快约 0.5 ms，但在 ViT-L 上 conf 误差 4.9-5.7%，贴近/超过门槛，因此不用。
数学核对：把 OptEoMT 全 fp32 运行，与原 `forward_backend` 的 token 相对误差 3e-7（数学等价）。

## 结果（基线 autocast：ViT-L model 17.05 / e2e 60.65，ViT-B model 9.62 / e2e 51.71 ms）
| 变体 | ViT-L model | ViT-L e2e | kernels | conf 相对误差 | gate | ViT-B model | ViT-B e2e | kernels | gate |
|---|---|---|---|---|---|---|---|---|---|
| **opt-eager-bf16（cudnn SDPA，推荐）** | 12.93 | 54.9 | 440 | 2.11% | 通过 | 5.13 | 47.4 | 240 | 通过 |
| sdpa-flash（torch auto 选它） | 12.91 | 57.3 | 440 | **5.65%** | **不通过** | 5.06 | 46.0 | 240 | 通过 |
| sdpa-efficient | 14.35 | 55.0 | 440 | 1.11% | 通过 | 5.49 | 47.4 | 240 | 通过 |
| sdpa-math | 36.82 | 77.7 | 848 | 5.66% | 不通过 | 13.89 | 57.2 | 444 | 通过 |
| head channels_last（+cudnn.benchmark） | 12.93 / 12.95 | 53.9 / 53.5 | 440 | 同 flash | 不通过 | 5.06 / 5.07 | 47.6 / 45.7 | 240 / 236 | 通过 |
| head ConvT 改 GEMM | 12.98 | 53.6 | 438 | 同 flash | 不通过 | 5.04 | 47.5 | 234 | 通过 |
| res_bf16（flash） | 12.36 | 54.2 | 389 | 4.93% | 勉强过 | 4.83 | 45.4 | 213 | 通过 |

e2e 主要由基线 post 的 CPU numpy 决定，噪声约 ±2 ms；model 数字稳定。head 两项对 model 无收益，cudnn.benchmark 也无收益。
ViT-L model：17.05 -> 12.93 ms（-24%），kernel 1302 -> 440，nsys 的 kernel 总时间 15.35 -> 12.12 ms，model 段 GPU 忙碌率 71% -> 93%（剩余空闲是 launch 间隙，R2 处理）。ViT-B：9.62 -> 5.13 ms，682 -> 240 kernels。
autocast 的 309 次 bfloat16_copy（1.61 ms）和 96 次 RoPE cat（0.91 ms）已消除（`profile/vitl16_r1_eager_e2e_summary.md`）。
剩余大头：cutlass sm80 GEMM 约 7.2 ms（约 126-168 TFLOPS，属 worker B 的融合/GEMM 范围）。

## 为什么 flash 在 ViT-L 上不过 gate、推荐哪个后端
不过的只有 frame 7（head 仅 1139 像素，confidence 参考 0.4786，flash 版给出 0.505，相对误差 5.65%）。排查：
- 数学没错：fp32 版误差 3e-7；bf16 版 token 相对误差 0.20%，比基线 autocast（0.38%）还小。
- 后端本身没有更差：在 frame 7 的真实 q/k/v 上，24 层每层对 fp64 SDPA 的相对 L2 误差 flash 1.68e-3、efficient 1.68e-3、cudnn 1.68e-3、math 1.63e-3，无差异。
- frame 7 的 confidence 对任何 bf16 扰动都极敏感：把 blocks 的不同子集换成 fp32，conf 误差在 0.5%-5.7% 间无规律跳动（全 fp32 blocks 仍有 1.9%）；基线 bf16-pure 也是 4.92%。head（bf16）本身只贡献 0.1%。
结论：这是 frame 7 小目标边界像素的运气，不是 flash 的缺陷；同样的 bf16 数值下 cudnn/efficient 恰好落在 gate 内。推荐 **cudnn**（速度与 flash 相同，12.96 vs 12.91 ms，且通过 gate）。但 ViT-L 上这个 gate 在 frame 7 上是脆弱的，后续每个新变体都要重测。TileLang attention（R4）在 ViT-L 上 conf 误差 0.49%。

## 遗留
- 建议调度者知悉该 gate 的脆弱性（可考虑用更多帧，或对小目标更稳的指标）。
