# 推理优化结果汇总

基线 = `round0_baseline` / `bf16-autocast`（与 scripts/infer_dinov3_eomt_head_video.py 一致）。延迟为 wall-clock p50（ms，含 CPU 下发开销），输入是真实视频 1.mp4 的 1900×1440 帧，切成 2 个 512×512 crop。精度门槛（±5%）：与 fp32 eager 相比，8 帧合并的 head IoU ≥ 0.95，且每帧置信度相对误差 ≤ 5%。

## vitl16

| round | variant | prep | model | post | e2e | e2e FPS | kernels/model fwd | vs R0 e2e | head IoU (pooled) | conf 相对误差 max | 精度门槛 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| round0_baseline | bf16-autocast | 5.77 | 17.05 | 35.36 | 60.65 | 16.5 | 1302 | 1.00x | 0.9931 | 0.0111 | ✅ |
| round0_baseline | bf16-pure | 5.56 | 15.10 | 35.09 | 57.83 | 17.3 | 1040 | 1.05x | 0.9897 | 0.0492 | ✅ |
| round0_baseline | fp32 | 5.35 | 40.84 | 34.77 | 81.09 | 12.3 | 933 | 0.75x | 1.0000 | 0.0000 | ✅ |
| round1_torch_cleanup | opt-eager-bf16 | - | 12.93 | - | 57.72 | - | 440 | 1.05x | 0.9947 | 0.0211 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-res_bf16 | - | 12.36 | - | 54.39 | - | 389 | 1.12x | 0.9854 | 0.0493 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-flash | - | 12.92 | - | 57.30 | - | 440 | 1.06x | 0.9918 | 0.0565 | ❌ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-efficient | - | 14.36 | - | 55.31 | - | 440 | 1.10x | 0.9945 | 0.0111 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-cudnn | - | 12.98 | - | 53.69 | - | 440 | 1.13x | 0.9947 | 0.0211 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-math | - | 36.86 | - | 77.90 | - | 848 | 0.78x | 0.9912 | 0.0566 | ❌ |
| round1_torch_cleanup | opt-eager-bf16-head-channels_last | - | 12.93 | - | 54.02 | - | 440 | 1.12x | 0.9918 | 0.0565 | ❌ |
| round1_torch_cleanup | opt-eager-bf16-head-convt_gemm | - | 12.98 | - | 53.76 | - | 438 | 1.13x | 0.9918 | 0.0566 | ❌ |
| round1_torch_cleanup | opt-eager-bf16-head-channels_last-cudnnbench | - | 12.95 | - | 53.58 | - | 440 | 1.13x | 0.9918 | 0.0565 | ❌ |
| round2_cuda_graph | paired-eager-vs-graph | - | - | - | - | - | - | - | - | - | - |
| round2_cuda_graph | opt-eager | - | 13.02 | - | 55.45 | - | 440 | 1.09x | 0.9947 | 0.0211 | ✅ |
| round2_cuda_graph | opt-cudagraph | - | 12.59 | - | 54.76 | - | 440 | 1.11x | 0.9947 | 0.0211 | ✅ |
| round2_cuda_graph | compile-max-autotune-no-cudagraphs | - | 10.41 | - | 51.56 | - | 241 | 1.18x | 0.9933 | 0.0486 | ✅ |
| round2_cuda_graph | compile-max-autotune | - | 10.26 | - | 51.54 | - | 242 | 1.18x | 0.9943 | 0.0169 | ✅ |
| round2_cuda_graph | compile-reduce-overhead | - | 10.71 | - | 53.46 | - | 246 | 1.13x | 0.9930 | 0.0272 | ✅ |
| round3_fusion | fused_block_fp32res | - | 11.27 | - | - | - | 233 | - | 0.9938 | 0.0251 | ✅ |
| round4_attention | opt-eager-attn-sdpa-cudnn | - | 12.95 | - | 69.45 | - | 440 | 0.87x | 0.9947 | 0.0211 | ✅ |
| round4_attention | opt-graph-attn-sdpa-cudnn | - | 12.46 | - | 54.04 | - | 440 | 1.12x | 0.9947 | 0.0211 | ✅ |
| round4_attention | opt-eager-attn-sdpa-flash | - | 13.06 | - | 56.28 | - | 440 | 1.08x | 0.9918 | 0.0565 | ❌ |
| round4_attention | opt-graph-attn-sdpa-flash | - | 12.39 | - | 53.33 | - | 440 | 1.14x | 0.9918 | 0.0565 | ❌ |
| round4_attention | opt-eager-attn-tl | - | 12.98 | - | 54.51 | - | 440 | 1.11x | 0.9949 | 0.0049 | ✅ |
| round4_attention | opt-graph-attn-tl | - | 12.33 | - | 53.53 | - | 440 | 1.13x | 0.9949 | 0.0049 | ✅ |
| round4_attention | opt-eager-attn-tl_rope | - | 11.76 | - | 53.50 | - | 320 | 1.13x | 0.9937 | 0.0314 | ✅ |
| round4_attention | opt-graph-attn-tl_rope | - | 11.29 | - | 52.67 | - | 320 | 1.15x | 0.9937 | 0.0314 | ✅ |
| round5_gemm_sm120 | r5_v1_tuned | - | 10.80 | - | - | - | 233 | - | 0.9952 | 0.0049 | ✅ |
| round6_low_precision | fp8-row-noproj-greedy | - | 13.75 | - | - | - | - | - | 0.9796 | 0.1013 | ❌ |
| round6_low_precision | fp8-row-6units | - | 14.80 | - | - | - | - | - | 0.9888 | 0.0160 | ✅ |
| round6_low_precision | fp8-row-18units | - | 13.71 | - | - | - | - | - | 0.9841 | 0.0218 | ✅ |
| round7_head_postprocess | bf16-autocast+fused-prepost | 0.70 | 16.93 | 0.22 | 17.81 | 56.1 | 1300 | 3.41x | 0.9933 | 0.0045 | ✅ |
| round7_head_postprocess | bf16-pure+fused-prepost | 0.71 | 15.05 | 0.23 | 15.90 | 62.9 | 1039 | 3.81x | 0.9837 | 0.1243 | ❌ |
| round7_head_postprocess | bf16-autocast+fused-prepost+fused-head | 0.70 | 16.46 | 0.22 | 17.38 | 57.5 | 1282 | 3.49x | 0.9930 | 0.0044 | ✅ |
| round8_pipeline | bf16-autocast+pipeline | - | - | - | 16.99 | 57.3 | - | 3.57x | 0.9930 | 0.0044 | ✅ |
| round9_integration | lightly+graph+fused-prepost+fused-head | - | - | - | 14.99 | 64.3 | - | 4.04x | 0.9930 | 0.0044 | ✅ |
| round9_integration | lightly+graph+fused-prepost+fused-head+fp8 | - | - | - | 14.66 | 65.8 | - | 4.14x | 0.9888 | 0.0160 | ✅ |
| round9_integration | opt(graphopt)+graph+fused-prepost | - | - | - | 12.69 | 76.9 | - | 4.78x | 0.9921 | 0.0271 | ✅ |
| round9_integration | opt(graphopt,gemm-head)+graph+fused-prepost | - | - | - | 12.65 | 75.7 | - | 4.79x | 0.9920 | 0.0271 | ✅ |
| round9_integration | fused-block+graph+fused-prepost+fused-head | - | - | - | 10.55 | 92.5 | - | 5.75x | 0.9937 | 0.0210 | ✅ |
| round9_integration | opt(graphopt,tl-attention)+graph+fused-prepost | - | - | - | 12.65 | 81.3 | - | 4.80x | 0.9937 | 0.0132 | ✅ |
| round9_integration | opt(graphopt,tl-rope-attention)+graph+fused-prepost | - | - | - | 11.60 | 84.0 | - | 5.23x | 0.9937 | 0.0213 | ✅ |
| round9_integration | fused-block+tl-attention+graph+fused-prepost+fused-head | - | - | - | 10.57 | 86.3 | - | 5.74x | 0.9957 | 0.0108 | ✅ |
| round9_integration | round5-v1-tuned+graph+fused-prepost+fused-head | - | - | - | 10.24 | 94.7 | - | 5.92x | 0.9946 | 0.0013 | ✅ |

## vitb16

| round | variant | prep | model | post | e2e | e2e FPS | kernels/model fwd | vs R0 e2e | head IoU (pooled) | conf 相对误差 max | 精度门槛 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| round0_baseline | bf16-autocast | 5.76 | 9.62 | 34.84 | 51.71 | 19.3 | 682 | 1.00x | 0.9926 | 0.0038 | ✅ |
| round0_baseline | bf16-pure | 5.86 | 8.09 | 34.10 | 51.61 | 19.4 | 540 | 1.00x | 0.9933 | 0.0028 | ✅ |
| round0_baseline | fp32 | 6.06 | 15.26 | 35.25 | 57.91 | 17.3 | 489 | 0.89x | 1.0000 | 0.0000 | ✅ |
| round1_torch_cleanup | opt-eager-bf16 | - | 5.13 | - | 47.93 | - | 240 | 1.08x | 0.9923 | 0.0024 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-res_bf16 | - | 4.83 | - | 45.51 | - | 213 | 1.14x | 0.9921 | 0.0024 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-flash | - | 5.06 | - | 46.26 | - | 240 | 1.12x | 0.9924 | 0.0023 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-efficient | - | 5.49 | - | 48.06 | - | 240 | 1.08x | 0.9924 | 0.0020 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-cudnn | - | 5.10 | - | 48.72 | - | 240 | 1.06x | 0.9923 | 0.0024 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-sdpa-math | - | 13.89 | - | 57.54 | - | 444 | 0.90x | 0.9924 | 0.0018 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-head-channels_last | - | 5.06 | - | 47.66 | - | 240 | 1.08x | 0.9924 | 0.0023 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-head-convt_gemm | - | 5.04 | - | 49.86 | - | 234 | 1.04x | 0.9924 | 0.0025 | ✅ |
| round1_torch_cleanup | opt-eager-bf16-head-channels_last-cudnnbench | - | 5.07 | - | 45.83 | - | 236 | 1.13x | 0.9924 | 0.0023 | ✅ |
| round2_cuda_graph | paired-eager-vs-graph | - | - | - | - | - | - | - | - | - | - |
| round2_cuda_graph | opt-eager | - | 5.17 | - | 48.40 | - | 240 | 1.07x | 0.9923 | 0.0024 | ✅ |
| round2_cuda_graph | opt-cudagraph | - | 4.82 | - | 47.39 | - | 240 | 1.09x | 0.9923 | 0.0024 | ✅ |
| round2_cuda_graph | compile-max-autotune-no-cudagraphs | - | 4.05 | - | 63.46 | - | 134 | 0.81x | 0.9938 | 0.0025 | ✅ |
| round2_cuda_graph | compile-max-autotune | - | 3.92 | - | 60.15 | - | 135 | 0.86x | 0.9934 | 0.0019 | ✅ |
| round2_cuda_graph | compile-reduce-overhead | - | 4.10 | - | 62.36 | - | 141 | 0.83x | 0.9913 | 0.0041 | ✅ |
| round3_fusion | fused_block_fp32res | - | 4.75 | - | - | - | 153 | - | 0.9932 | 0.0031 | ✅ |
| round4_attention | opt-eager-attn-sdpa-cudnn | - | 5.18 | - | 46.83 | - | 240 | 1.10x | 0.9923 | 0.0024 | ✅ |
| round4_attention | opt-graph-attn-sdpa-cudnn | - | 4.80 | - | 46.50 | - | 240 | 1.11x | 0.9923 | 0.0024 | ✅ |
| round4_attention | opt-eager-attn-sdpa-flash | - | 5.13 | - | 47.17 | - | 240 | 1.10x | 0.9924 | 0.0023 | ✅ |
| round4_attention | opt-graph-attn-sdpa-flash | - | 4.77 | - | 46.92 | - | 240 | 1.10x | 0.9924 | 0.0023 | ✅ |
| round4_attention | opt-eager-attn-tl | - | 5.16 | - | 47.10 | - | 240 | 1.10x | 0.9935 | 0.0018 | ✅ |
| round4_attention | opt-graph-attn-tl | - | 4.74 | - | 46.95 | - | 240 | 1.10x | 0.9935 | 0.0018 | ✅ |
| round4_attention | opt-eager-attn-tl_rope | - | 4.64 | - | 47.03 | - | 180 | 1.10x | 0.9918 | 0.0021 | ✅ |
| round4_attention | opt-graph-attn-tl_rope | - | 4.35 | - | 46.80 | - | 180 | 1.10x | 0.9918 | 0.0021 | ✅ |
| round5_gemm_sm120 | r5_v1_tuned | - | 4.67 | - | - | - | 153 | - | 0.9932 | 0.0031 | ✅ |
| round6_low_precision | fp8-row-noproj-greedy | - | 5.30 | - | - | - | - | - | 0.9825 | 0.0053 | ✅ |
| round6_low_precision | fp8-row-14units | - | 5.12 | - | - | - | - | - | 0.9825 | 0.0053 | ✅ |
| round6_low_precision | fp8-row-20units | - | 4.93 | - | - | - | - | - | 0.9643 | 0.0074 | ✅ |
| round7_head_postprocess | bf16-autocast+fused-prepost | 0.73 | 9.49 | 0.20 | 10.45 | 95.7 | 680 | 4.95x | 0.9923 | 0.0009 | ✅ |
| round7_head_postprocess | bf16-pure+fused-prepost | 0.71 | 7.95 | 0.20 | 8.86 | 112.8 | 539 | 5.83x | 0.9904 | 0.0027 | ✅ |
| round7_head_postprocess | bf16-autocast+fused-prepost+fused-head | 0.71 | 9.16 | 0.20 | 10.07 | 99.3 | 658 | 5.14x | 0.9921 | 0.0011 | ✅ |
| round8_pipeline | bf16-autocast+pipeline | - | - | - | 7.47 | 118.3 | - | 6.92x | 0.9921 | 0.0011 | ✅ |
| round9_integration | lightly+graph+fused-prepost+fused-head | - | - | - | 5.84 | 153.6 | - | 8.86x | 0.9921 | 0.0011 | ✅ |
| round9_integration | fused-block+graph+fused-prepost+fused-head | - | - | - | 4.28 | 208.3 | - | 12.08x | 0.9943 | 0.0021 | ✅ |
| round9_integration | lightly+graph+fused-prepost+fused-head+fp8 | - | - | - | 5.47 | 162.4 | - | 9.46x | 0.9825 | 0.0053 | ✅ |
| round9_integration | opt(graphopt)+graph+fused-prepost | - | - | - | 5.06 | 182.0 | - | 10.23x | 0.9930 | 0.0015 | ✅ |
| round9_integration | opt(graphopt,gemm-head)+graph+fused-prepost | - | - | - | 4.94 | 178.4 | - | 10.47x | 0.9932 | 0.0012 | ✅ |
| round9_integration | opt(graphopt,tl-rope-attention)+graph+fused-prepost | - | - | - | 4.63 | 197.6 | - | 11.17x | 0.9920 | 0.0014 | ✅ |
| round9_integration | fused-block+tl-attention+graph+fused-prepost+fused-head | - | - | - | 4.24 | 209.0 | - | 12.21x | 0.9934 | 0.0012 | ✅ |
| round9_integration | round5-v1-tuned+graph+fused-prepost+fused-head | - | - | - | 4.00 | 214.1 | - | 12.92x | 0.9943 | 0.0021 | ✅ |

## 整段视频吞吐（1.mp4，126 帧循环，稳态）

原脚本 = decode + predict_head 串行循环。服务延迟 = inflight=1 时从提交到 host 拿到结果。

| 模型 | round | variant | 原脚本 FPS | 优化 FPS（inflight=2） | 加速 | 服务延迟 p50 (ms) | 126 帧 vs 原脚本 head IoU（pooled / min） | fp32 gate |
|---|---|---|---|---|---|---|---|---|
| vitl16 | round8_pipeline | bf16-autocast+pipeline | 15.5 | 57.3 | 3.71x | 16.99 | 0.994 / 0.984 | ✅ |
| vitb16 | round8_pipeline | bf16-autocast+pipeline | 12.7 | 118.3 | 9.33x | 7.47 | 0.993 / 0.977 | ✅ |
| vitl16 | round9_integration | lightly+graph+fused-prepost+fused-head | 15.9 | 64.3 | 4.04x | 14.99 | 0.994 / 0.984 | ✅ |
| vitl16 | round9_integration | lightly+graph+fused-prepost+fused-head+fp8 | 15.9 | 65.8 | 4.13x | 14.66 | 0.990 / 0.941 | ✅ |
| vitl16 | round9_integration | opt(graphopt)+graph+fused-prepost | 15.7 | 76.9 | 4.90x | 12.69 | 0.993 / 0.983 | ✅ |
| vitl16 | round9_integration | opt(graphopt,gemm-head)+graph+fused-prepost | 15.9 | 75.7 | 4.76x | 12.65 | 0.993 / 0.983 | ✅ |
| vitl16 | round9_integration | fused-block+graph+fused-prepost+fused-head | 15.1 | 92.5 | 6.13x | 10.55 | 0.992 / 0.969 | ✅ |
| vitl16 | round9_integration | opt(graphopt,tl-attention)+graph+fused-prepost | 15.8 | 81.3 | 5.13x | 12.65 | 0.993 / 0.980 | ✅ |
| vitl16 | round9_integration | opt(graphopt,tl-rope-attention)+graph+fused-prepost | 15.7 | 84.0 | 5.35x | 11.60 | 0.993 / 0.973 | ✅ |
| vitl16 | round9_integration | fused-block+tl-attention+graph+fused-prepost+fused-head | 15.7 | 86.3 | 5.50x | 10.57 | 0.992 / 0.981 | ✅ |
| vitl16 | round9_integration | round5-v1-tuned+graph+fused-prepost+fused-head | 15.1 | 94.7 | 6.28x | 10.24 | 0.992 / 0.980 | ✅ |
| vitb16 | round9_integration | lightly+graph+fused-prepost+fused-head | 16.4 | 153.6 | 9.37x | 5.84 | 0.993 / 0.978 | ✅ |
| vitb16 | round9_integration | fused-block+graph+fused-prepost+fused-head | 16.5 | 208.3 | 12.61x | 4.28 | 0.992 / 0.977 | ✅ |
| vitb16 | round9_integration | lightly+graph+fused-prepost+fused-head+fp8 | 16.4 | 162.4 | 9.91x | 5.47 | 0.980 / 0.922 | ✅ |
| vitb16 | round9_integration | opt(graphopt)+graph+fused-prepost | 16.5 | 182.0 | 11.01x | 5.06 | 0.991 / 0.976 | ✅ |
| vitb16 | round9_integration | opt(graphopt,gemm-head)+graph+fused-prepost | 16.4 | 178.4 | 10.89x | 4.94 | 0.991 / 0.974 | ✅ |
| vitb16 | round9_integration | opt(graphopt,tl-rope-attention)+graph+fused-prepost | 16.5 | 197.6 | 11.96x | 4.63 | 0.991 / 0.977 | ✅ |
| vitb16 | round9_integration | fused-block+tl-attention+graph+fused-prepost+fused-head | 16.5 | 209.0 | 12.65x | 4.24 | 0.992 / 0.977 | ✅ |
| vitb16 | round9_integration | round5-v1-tuned+graph+fused-prepost+fused-head | 16.5 | 214.1 | 12.96x | 4.00 | 0.992 / 0.977 | ✅ |
