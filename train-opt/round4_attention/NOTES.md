# Round4：Attention（负责：Agent B，2026-10-02）

## 1. 目标与假设
- 对应 H2/H4：融合 packed-QKV 的 RoPE，降低逐元素 kernel 启动；同步实现可验证的 Triton FA 与 compact mask。

## 2. 改动
- `impl/rope.py`：packed QKV 一次 fwd RoPE、一次 bwd 逆 RoPE；`impl/mask.py`：精确 bilinear 128→32 + disable；`impl/attention.py`：无原子 Triton FA 前后向。
- patch variants：`sdpa_rope`（默认）、`triton`、`triton_rope`、`triton_rope_mask`、`flex`、`sdpa_compact`。infer-opt 复制文件仅作参考，未用于训练 patch。
- 新增真实 CUDA C++ 扩展 `impl/cuda_attention.cu` + `impl/cuda_attention.py`，由 `cuda_ref` variant 调度；forward、dQ、dK/dV 均在 CUDA kernel 中执行，编译缓存位于 `/root/autodl-tmp/train-opt-tmp/B/cuda_attention_build_v2`。

CUDA C++ A 级命令：`/root/miniconda3/bin/python train-opt/round4_attention/tests/test_cuda_attention.py --output train-opt/round4_attention/profile/cuda_ref_a.json`。逐线程参考实现用于验证真实 kernel 的输出和 qkv 全部梯度，正式 B_AUTO 性能字段暂留 null。

## 3. nsys 分析过程
- M0/harness/B_AUTO 尚未完成，未运行 nsys；整步字段记 null。独立 microbench 使用双 GPU 锁、warmup 20、50 iter×3 median。
- 命令：`flock /root/autodl-tmp/infer-opt-tmp/gpu.lock flock /root/autodl-tmp/train-opt-tmp/gpu.lock env TRAIN_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python train-opt/round4_attention/tests/bench_rope.py --batch 16 --output train-opt/round4_attention/profile/rope_microbench_B16.json`。

## 4. 结果（开发 B=16，非正式 B_AUTO）
| variant | ViT-L N=1029/1129 fwd+bwd us | ViT-B N=1029/1129 fwd+bwd us |
|---|---:|---:|
| torch RoPE + SDPA | 4300.2 / 4506.9 | 3046.0 / 3182.9 |
| fused RoPE + SDPA | 1853.6 / 1979.8 | 1416.8 / 1502.0 |
| Triton FA joint B=16 (N=1129) | 6069.5 | 4645.6 |

详见 `profile/rope_microbench_B16.json`、`profile/microbench_B16.json`。Triton FA 反向资源和时延没有优势，不推荐主线。

## 5. 正确性
- `rope_B16.json`：H=12/16、N=1029/1129、scale=1/8，RoPE fwd 和所有输入 grad 与 torch 旋转逐位相等。
- `correctness_B16.json`：普通/compact mask 场景通过；完整 FA 的 RoPE scale8 相对 fp32 attention 参考 dq/dk cosine 约 0.995（torch bf16 参考同样受舍入影响，custom 对 torch bf16 cosine >=0.999993），该严格案例保留 pass=false。
- Full harness C/D：null，等待 M0 和 A 调度。官方 B_AUTO 单步门已覆盖两模型三阶段：ViT-B14 `loss_rel=0`、`grad_cos_min=0.999999404`、`grad_norm_rel=5.48e-7`、update cosine 1；ViT-L7 `loss_rel=6.45e-8`、`grad_cos_min=0.999996483`、`grad_norm_rel=2.38e-7`、update cosine 1。每阶段证据在 `profile/check_vit*_B*_sdpa_rope_{early,mid,late}.json`，耐久日志在 `/root/autodl-tmp/train-opt-tmp/B/`。
- 已在正式 ViT-B14、实际原脚本 FP32、early阶段完成full-step B：loss_rel=0，grad_cos_min=0.999999642，grad_norm_rel=5.33e-7，step update cos=1，pass=true。mid/late和ViT-L尚未测，不能据early单项认定本轮全部门槛通过。源Fabic默认不会开启autocast，bf16 microbench只作独立开发对照。

## 6. 静态分析
`profile/static/summary.json` 真实 Triton cache：forward 86 regs/24 KiB shared，backward_q 96 regs/28 KiB，backward_kv 最高 255 regs/78 KiB；HMMA 可见，FA 未产生 UTMALDG；occupancy null。

## 7. 结论
推荐接入 `sdpa_rope`，保留原随机 RoPE 调用顺序和 joint mask 语义；compact/Triton/Flex 只作实验。
### cuda_fast tiled experiment (2026-10-02)

Implemented a Blackwell sm_120 tiled online-softmax prototype in
`impl/cuda_attention.cu` (`forward_fast`/`backward_fast`). It stages 64-token
K/V tiles in shared memory, uses warp shuffles for fp32 dot reductions and
fuses RoPE. The prototype was compiled with nvcc `compute_120/sm_120`.

The prototype is intentionally kept out of the training patch until A-level
checks pass. Early smoke testing exposed numerical mismatches in the shared
warp reduction/backward path (FP32 N=33 output cosine below 0.999), so the
supported variant remains `cuda_ref` and no speedup is reported. The fast
entrypoints remain available for follow-up debugging; the reference CUDA path
continues to provide the verified correctness baseline.
