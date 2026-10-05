# Round11: torch BF16 mainline "T-BF16" (Agent A, 2026-10-02/03)

STATUS: FINAL. Sections 3, 4.3-4.7 and 7 were completed by the coordinator on 2026-10-03 from the logged measurements (`logs/campaign_final.log`, `logs/campaign_rest.log`) after the agent was cut off by the session limit.

Machine note: the first half of the work (microbenchmarks, first screening) ran on the OLD container
(`profile/old_machine/`, ViT-L mid R0 424.7 / P0 381.7 / TB1 168.3 ms). The server then moved to a new container
with a different physical RTX 5090. Every number below that is compared with another number was re-measured on
the NEW machine in this round (R0 / P0 / T-BF16 all re-run). Old-machine numbers are only quoted when marked "old".
Differences between the machines were <= 2% on the microbenchmarks that were repeated.

## 1. Goal and hypothesis

The original script says `bf16-mixed`, but Lightly's `training_step` never runs under autocast, so training is
FP32/TF32. R0 (ViT-L B=7 425 ms, ViT-B B=14 370 ms) and P0 (382 / 310 ms) are FP32. Hypotheses:
- H-A: Fabric bf16 autocast roughly halves GEMM time (BF16 HMMA with FP32 accumulate ~230 TFLOPS on the 5090 vs ~118 TF32).
- H-B: attention gets much cheaper (FP32 mem-efficient bwd was 86 ms/step on ViT-L).
- H-C: with GEMM and attention cheap, the host-side/elementwise work removed by r1/r3/r5 matters more than before (the step is ~170 ms, so every
  CPU sync and fp32 round trip is a larger fraction).
- H-D: backend choice (flash / cuDNN / efficient, bool vs bias mask) and the RoPE implementation are worth measuring at real shapes.
Deliverable: a same-precision baseline "T-BF16" for rounds 12-17, with BF16-vs-FP32 gates and a measured BF16 self-noise.

## 2. Changes (all in `train-opt/round11_bf16_mixed/`)

- `impl/patch.py`: `+`-composable variants (module docstring has the requires/conflicts table). `autocast` is unchanged
  (sets `meta["enable_autocast"]`, wraps `ops.loss_all_layers` -> fp32 logits, autocast disabled inside the loss).
  New variants: `rope_triton`, `plain_flash|plain_cudnn|plain_eff`, `joint_cudnn|joint_eff`, `joint_bias`, `joint_qsplit`, `attn_core`,
  `head_cl`, `head_ln_bf16`, `bf16_shadow`, `cudnn_bench`, `compile_block`, `compile_block_max`.
  Variant validation: `VARIANT_REQUIRES` / `VARIANT_CONFLICTS`; patch-level checks: r11 must come after `round1_torch_cleanup` and
  `round3_loss_matcher` (otherwise they overwrite `ops.attn` / `ops.loss_all_layers`), and the attention variants conflict with `round4_attention`
  (which replaces `ops.attn` and bypasses `attn_core`). `apply` raises on violations.
- `impl/attn.py`: `ops.attn_core` implementation. Accepts the source bool `[B,N,N]` mask and r1's `[B,1,N,N]` broadcast mask; any non-tensor mask object
  (e.g. round12's `QueryBitmask`) is delegated to the previously installed `attn_core`.
  `joint_qsplit`: only the 100 query rows of a joint block are masked (rows >= 100 see every key, checked in the unit test), and rows are independent in attention, so
  the 1029 unmasked rows go to flash and only the 100 masked rows go to the masked backend; outputs are concatenated in `[B,N,H,D]` order.
  `rope_triton` imports round4's fused packed-QKV RoPE (1 launch fwd, 1 bwd; bit-identical forward to the torch rotation).
- `impl/head.py`: `head_cl` (round1 helper) and `head_ln_bf16` (LayerNorm2D in bf16 in/out, fp32 statistics inside the kernel, bf16-rounded affine params).
- `impl/shadow.py`: `bf16_shadow` (round5's `_ShadowLinear` + a parameter discovery that works under Fabric; round5's `setup` matched nothing in this harness because
  `ctx.train_model` names carry the Fabric wrapper prefix; round7's `capturable_shadow` uses the same pattern and would hit the same problem).
- `impl/compile_ref.py`: regional `torch.compile` reference (one compiled region per block, default and max-autotune-no-cudagraphs).
- `run.py`: benchmark / gates / max-batch / nsys entry (`--stack "r1,r3,r5,r7,r11:autocast+..."`). `nsys_report.py`: per-category GPU time, CPU-vs-GPU per NVTX range,
  top kernels. `summarize_r11.py`: markdown tables.
- `tests/`: `test_round11_cpu.py` (20 CPU tests: variant rules, mask layouts, bias/qsplit equality with the reference, RoPE path, fallback for unknown masks, patch order / conflicts,
  fp32 loss wrapper, shadow protocol, compile-reference block equality), `test_round11_gpu.py` (56 GPU tests vs fp32 math reference incl. spiky inputs and masks),
  microbenchmarks `bench_attn.py`, `bench_rope.py`, `bench_head.py`.

## 3. nsys analysis process

Command (per model, phase mid, warmup 20, 5 captured steps, `step_XX`/`fwd`/`blk_XX`/`predict_*`/`mask_*`/`loss`/`bwd`/`clip_opt_sched` NVTX ranges):
```
bash train-opt/common/profile_nsys.sh /root/autodl-tmp/train-opt-tmp/round11_bf16_mixed/nsys/tb_<model>_mid \
     round11_bf16_mixed/run.py --model <model> --stack "<T-BF16 spec>" --action profile --phases mid
python train-opt/common/nsys_analyze.py <prefix>
python train-opt/round11_bf16_mixed/nsys_report.py <prefix>
```
`nsys_report.py` attributes every GPU op to the NVTX phase active on the host at launch time and classifies kernels by name.

## 4. Results

### 4.1 Operator level (new machine, bf16, real shapes, median of 3 x 50)

Attention fwd+bwd per block, ms (B=7,H=16 for ViT-L; B=14,H=12 for ViT-B; D=64). `qsplit(a,b)`: unmasked rows on a, masked rows on b.

| case | ViT-L | ViT-B |
|---|---:|---:|
| plain N=1029 flash / auto / cudnn / efficient | 0.659 / 0.656 / 0.779 / 1.478 | 1.021 / 1.020 / 1.219 / 2.119 |
| joint N=1129 no mask (prob 0) flash / auto / cudnn / efficient | 0.699 / 0.697 / 0.847 / 1.613 | 1.088 / 1.085 / 1.322 / 2.402 |
| joint masked: coordinator default (bool [B,N,N] expanded, auto) | 2.527 | 3.754 |
| joint masked: bool [B,1,N,N] auto / efficient / cudnn | 1.758 / 1.759 / 1.312 | 2.617 / 2.616 / 2.074 |
| joint masked: additive bias auto / efficient / cudnn | 1.751 / 1.748 / 1.279 | 2.608 / 2.608 / 2.030 |
| joint masked: qsplit(flash, efficient) bool / bias | 1.236 / 1.234 | 1.804 / 1.801 |
| joint masked: qsplit(flash, cudnn) bool / bias | 1.260 / 1.425 | 1.601 / 1.595 |
| joint masked: qsplit(cudnn, cudnn) bool | 1.351 | 1.819 |

No backend/shape combination failed (all 17 masked candidates and 4 mask-free candidates ran for all 6 cases). All candidates vs an fp32 math-backend reference on spiky inputs
(large q/k, huge-norm keys): output cosine >= 0.9999996, dqkv cosine >= 0.99997, dqkv rel-L2 <= 6.5e-3. `auto` selects flash without a mask and mem-efficient with a mask.
Choice: flash for mask-free, `joint_qsplit` + cuDNN for the 100 masked rows (best on ViT-B, within noise of the best on ViT-L). Saving vs the coordinator default: ~1.3 ms per masked block on ViT-L
(3 blocks at most) - attention is no longer a bottleneck either way (24 blocks x ~0.7 ms = 17 ms/step on ViT-L).

RoPE (fused, with flash): fwd+bwd per block 0.979 ms vs 2.092 ms torch (ViT-L N=1029), 1.194 vs 2.577 (ViT-B); forward output bit-identical, dqkv rel-L2 1e-5.
That is ~27 ms/step on ViT-L and ~17 ms/step on ViT-B.

Head (`_predict`, fwd+bwd of one call, the step has 4): default torch 14.68 ms (ViT-L) / 20.62 (ViT-B); `head_cl` 14.64 / 20.60 (no effect: the token transpose is already channels-last-strided);
`head_ln_bf16` 11.17 / 15.44; round6 `channels_last_bmm` 14.69 / 20.59, `convt_gemm|dwln|all` 18.25 / 25.5, `fused_dwln` 20.4 / 29.1 (all slower than torch under bf16).
Accuracy vs fp32 head: mask-logit cos 0.99996, parameter-grad cos min 0.99996 for default and `head_ln_bf16` alike.
`head_ln_bf16` saves ~3.5 ms (ViT-L) and ~5.2 ms (ViT-B) per call = 14 / 21 ms per step.

### 4.2 Step level, mid phase screening on the new machine (50 x 3, warmup 20; ms per step; fwd/loss/bwd/opt in ms)

Stack names: `TB1` = `r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16` (r1=nosync+mask_skip+adamw_fused, r3=batched_torch, r5=ln_fuse, r7=capturable_torch).

| composition | ViT-L ms | ViT-L img/s | fwd/loss/bwd/opt | ViT-B ms | ViT-B img/s |
|---|---:|---:|---|---:|---:|
| autocast + SDPA backends only (`AC_SDPA`) | 234.3 | 29.9 | 64.1/21.3/125.5/22.3 | 228.9 | 61.2 |
| + rope_triton (`AC_SDPA_ROPE`) | 216.4 | 32.3 | 55.9/21.9/116.9/20.8 | 209.3 | 66.9 |
| P0 patches under autocast (`P0_AC`, r4 sdpa_rope) | 185.0 | 37.8 | 54.2/3.8/114.5/12.1 | 168.2 | 83.2 |
| **TB1** | **169.4** | **41.3** | 47.7/3.6/105.6/12.1 | **145.9** | **95.9** |
| TB1 without r5 (`ln_fuse`) | 175.8 (+6.4) | 39.8 | 49.3/3.9/109.9/12.3 | 151.5 (+5.6) | 92.4 |
| TB1 without r3 (batched loss) | 190.8 (+21.4) | 36.7 | 48.0/22.6/106.9/12.8 | 181.0 (+35.1) | 77.4 |
| TB1 without r1 | FAILED (other process on GPU; old machine: +1.1) | | | 148.2 (+2.3) | 94.5 |
| TB1 without r7 | 166.5 (-2.9) | 42.1 | 47.6/3.6/104.6/10.3 | not run | |
| TB1 with r7 `cuda_multi_tensor` | 165.8 (-3.5) | 42.2 | 47.7/3.5/105.8/8.5 | not run | |
| TB1 + bf16_shadow | 175.2 (+5.8) | 40.0 | 46.4/3.6/111.6/12.1 | 151.5 (+5.6) | 92.4 |
| TB1 + head_cl | 170.9 (+1.5) | 41.0 | | 147.2 (+1.3) | 95.1 |
| TB1 + cudnn_bench | FAILED (other process); old machine -2.1 | | | 145.5 (-0.5) | 96.2 |
| TB1 with joint_eff instead of joint_cudnn | 169.6 (+0.3) | 41.3 | | not run | |
| TB1 with joint_cudnn + joint_bias, no qsplit | 170.1 (+0.8) | 41.1 | | not run | |
| TB1 with plain_cudnn | 172.9 (+3.6) | 40.5 | | not run | |
| TB1 with r3 `cuda` variant | 170.1 (+0.8) | 41.2 | | not run | |

Reading (all from the table; differences below ~2 ms are noise level):
- r3 batched loss is the largest single item (21 / 35 ms): the unbatched loss/matcher does ~100 syncs per step regardless of precision.
- r5 `ln_fuse` still pays (+6 ms both models) and already emits bf16 under autocast; r4 `sdpa_rope` is superseded by `attn_core` + `rope_triton` + backend choice (P0_AC -> TB1: -15.6 / -22.3 ms).
- r1 (nosync + mask_skip + adamw_fused) is worth ~1-2 ms in the mid phase now that GEMM/attention are cheap (ViT-B +2.3 ms without it); kept because it also removes the host syncs that
  block graph capture and its mask_skip is what makes the late phase cheaper.
- r7 `capturable_torch` costs 3 ms in eager (its `zero_grad(set_to_none=False)` makes backward accumulate into pre-zeroed fp32 gradients, which is also why the opt column differs); it is kept in the published
  stack only because round16 (full graph) needs a capturable optimizer. `cuda_multi_tensor` is -3.5 ms; `bf16_shadow` is +6 ms slower (the post-optimizer refresh costs as much as autocast's cached cast saves).
- `head_cl`, `cudnn_bench`, joint backend variants: no measurable effect (<= 1-2 ms).

### 4.3 Final matrix, three phases weighted 0.2/0.6/0.2, same machine

Same machine (new container), 50 x 3, warmup 20, phases weighted 0.2/0.6/0.2, fixed B. T-BF16 = TB1 = `r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16`. `compile_default` = `r1,r3,r7,r11:autocast+compile_block` (per-block `torch.compile`; it lacks r5, rope_triton, the attention backend choice and head_ln_bf16, so it is a reference, not the same stack). `compile_max_autotune` was not run (long and flaky; trimmed by the coordinator).

| model | composition | early ms | mid ms | late ms | weighted img/s | vs R0 | vs P0 | peak GB | fwd/loss/bwd/opt ms (mid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| vitl16 B=7 | R0 | 432.2 | 432.8 | 432.5 | 16.18 | 1.00x | 0.90x | 28.8 | 118.1/21.6/271.2/20.9 |
| vitl16 B=7 | P0 | 389.3 | 390.3 | 389.2 | 17.95 | 1.11x | 1.00x | 27.1 | 107.8/3.7/265.1/12.6 |
| vitl16 B=7 | T-BF16 | 169.8 | 169.3 | 167.7 | 41.40 | 2.56x | 2.31x | 17.5 | 47.7/3.6/105.4/12.2 |
| vitl16 B=7 | compile_default | 189.5 | 188.5 | 186.1 | 37.19 | 2.30x | 2.07x | 19.3 | 54.7/4.0/116.0/13.0 |
| vitb16 B=14 | R0 | 376.4 | 376.9 | 376.4 | 37.16 | 1.00x | 0.84x | 29.1 | 99.7/40.1/225.3/10.8 |
| vitb16 B=14 | P0 | 316.1 | 315.5 | 313.2 | 44.42 | 1.20x | 1.00x | 26.3 | 87.6/5.6/214.6/6.8 |
| vitb16 B=14 | T-BF16 | 148.7 | 145.8 | 143.3 | 95.97 | 2.58x | 2.16x | 15.3 | 39.9/5.9/93.0/6.2 |

### 4.4 nsys category table

**vitl16, phase mid, T-BF16** (nsys, warmup 20, 5 captured steps; host times include nsys overhead)

Captured steps: 5; step wall (NVTX, host) 187.1 ms; summed GPU op time 160.6 ms/step
GPU busy ratio 0.8585; kernels/step 3607; syncs/step 8.0; gaps>20us: 216.0/step = 14.39 ms/step; launch API time / GPU kernel time = 0.14415796072853485

| category | ms/step | share | kernels/step | fwd | loss | bwd | opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEMM | 71.24 | 44.4% | 352 | 23.76 | 0.00 | 47.48 | 0.00 |
| elementwise/reduce | 25.30 | 15.8% | 1959 | 6.72 | 0.00 | 18.57 | 0.00 |
| head conv/dwconv | 23.32 | 14.5% | 68 | 7.31 | 0.00 | 16.00 | 0.00 |
| attention bwd | 11.04 | 6.9% | 76 | 0.00 | 0.00 | 11.04 | 0.00 |
| optimizer | 10.62 | 6.6% | 723 | 0.00 | 0.00 | 0.00 | 10.62 |
| LN | 9.85 | 6.1% | 132 | 2.52 | 0.00 | 7.33 | 0.00 |
| attention fwd | 4.59 | 2.9% | 26 | 4.59 | 0.00 | 0.00 | 0.00 |
| loss/matcher | 2.35 | 1.5% | 220 | 0.00 | 2.35 | 0.00 | 0.00 |
| RoPE | 1.19 | 0.7% | 48 | 0.59 | 0.00 | 0.60 | 0.00 |
| other | 0.61 | 0.4% | 3 | 0.52 | 0.00 | 0.09 | 0.00 |
| memcpy/memset | 0.51 | 0.3% | 538 | 0.00 | 0.03 | 0.24 | 0.24 |

Phase GPU time (ms/step, summed op duration by launch phase): fwd=46.0, loss=2.4, outside_phase=0.0, bwd=101.3, clip_opt_sched=10.9

Largest 'other' kernels (ms/step): void at::native::<unnamed>::upsample_bilinear2d_out_frame<float, float> 0.52; void at::native::<unnamed>::grid_sampler_2d_backward_kernel<float, int> 0.09

CPU vs GPU per NVTX range:

| range | CPU ms | GPU busy ms | GPU ops | CPU/GPU |
|---|---:|---:|---:|---:|
| fwd | 43.60 | 46.02 | 998 | 0.95 |
| loss | 15.67 | 2.38 | 255 | 6.57 |
| bwd | 66.29 | 101.35 | 1919 | 0.65 |
| clip_opt_sched | 59.14 | 10.85 | 959 | 5.45 |
| predict_final | 1.25 | 3.09 | 27 | 0.41 |
| blk_* (sum of 24) | 26.17 | 32.15 | 490 | 0.81 |

**vitb16, phase mid, T-BF16** (nsys, warmup 20, 5 captured steps; host times include nsys overhead)

Captured steps: 5; step wall (NVTX, host) 152.3 ms; summed GPU op time 136.4 ms/step
GPU busy ratio 0.8961; kernels/step 2317; syncs/step 8.0; gaps>20us: 99.4/step = 6.39 ms/step; launch API time / GPU kernel time = 0.11296833520812527

| category | ms/step | share | kernels/step | fwd | loss | bwd | opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEMM | 41.64 | 30.5% | 220 | 13.23 | 0.00 | 28.41 | 0.00 |
| head conv/dwconv | 30.39 | 22.3% | 96 | 9.36 | 0.00 | 21.03 | 0.00 |
| elementwise/reduce | 28.38 | 20.8% | 1191 | 7.75 | 0.00 | 20.63 | 0.00 |
| LN | 12.60 | 9.2% | 84 | 3.37 | 0.00 | 9.23 | 0.00 |
| attention bwd | 8.70 | 6.4% | 40 | 0.00 | 0.00 | 8.70 | 0.00 |
| loss/matcher | 4.56 | 3.3% | 253 | 0.00 | 4.56 | 0.00 | 0.00 |
| optimizer | 3.82 | 2.8% | 392 | 0.00 | 0.00 | 0.00 | 3.82 |
| attention fwd | 3.30 | 2.4% | 14 | 3.30 | 0.00 | 0.00 | 0.00 |
| RoPE | 1.52 | 1.1% | 24 | 0.73 | 0.00 | 0.79 | 0.00 |
| other | 1.22 | 0.9% | 3 | 1.10 | 0.00 | 0.12 | 0.00 |
| memcpy/memset | 0.31 | 0.2% | 345 | 0.00 | 0.05 | 0.13 | 0.13 |

Phase GPU time (ms/step, summed op duration by launch phase): fwd=38.8, loss=4.6, outside_phase=0.0, bwd=89.0, clip_opt_sched=3.9

Largest 'other' kernels (ms/step): void at::native::<unnamed>::upsample_bilinear2d_out_frame<float, float> 1.10; void at::native::<unnamed>::grid_sampler_2d_backward_kernel<float, int> 0.12

CPU vs GPU per NVTX range:

| range | CPU ms | GPU busy ms | GPU ops | CPU/GPU |
|---|---:|---:|---:|---:|
| fwd | 20.85 | 38.83 | 610 | 0.54 |
| loss | 27.07 | 4.61 | 302 | 5.87 |
| bwd | 54.16 | 89.04 | 1216 | 0.61 |
| clip_opt_sched | 48.63 | 3.94 | 520 | 12.33 |
| predict_final | 0.90 | 4.28 | 35 | 0.21 |
| blk_* (sum of 12) | 10.87 | 19.43 | 250 | 0.56 |

Reading:
- ViT-L: GEMM is 44% of GPU time; head convolutions plus the elementwise work around them are the next largest block; attention (fwd 4.6 + bwd 11.0 ms) and RoPE (1.2 ms) are small after the backend choices.
- ViT-B: the head is the biggest block after GEMM: `head conv/dwconv` alone is 30.4 ms (22% of GPU time), plus a large part of the 28.4 ms elementwise/reduce bucket.
- The step is partly host-bound outside the backward pass: for ViT-L `loss` CPU 15.7 ms vs GPU 2.4 ms and `clip_opt_sched` CPU 59 ms vs GPU 11 ms. The backward pass is GPU-bound (the CPU runs ahead), so the CPU can hide only part of the optimizer launch time; the rest shows up as GPU idle gaps (gaps over 20 us: 14.4 ms/step ViT-L, 6.4 ms/step ViT-B in the profile). About 3600 / 2300 kernels per step.
- Full per-NVTX tables and the top-20 kernels: `/root/autodl-tmp/train-opt-tmp/round11_bf16_mixed/nsys/tb_<model>_mid_r11_report.md`.

### 4.5 Gates

**B-lite, precision change (BF16 TB1 vs FP32 R0).** Criterion: `loss_rel <= 2e-2 and global grad cos >= 0.99` (CONTRACTS section 7). Every run passes. The per-parameter minimum cosine (`grad_cos_min`) is reported only: a few parameters (layer-0 LayerNorm weight/bias, qkv weight, storage tokens) are extremely sensitive to rounding noise because their gradients are amplified through 24 blocks.

| comparison | model | seed | phase | loss_rel | global grad cos | per-param min cos | grad_norm_rel | update cos min |
|---|---|---:|---|---:|---:|---:|---:|---:|
| TB1 vs R0 (fp32) | vitb16 | 0 | early | 3.8e-05 | 0.99993 | 0.958 | 1.0e-03 | 1.0000 |
| TB1 vs R0 (fp32) | vitb16 | 0 | late | 6.4e-05 | 0.99984 | 0.939 | 1.0e-03 | 1.0000 |
| TB1 vs R0 (fp32) | vitb16 | 0 | mid | 3.2e-04 | 0.99915 | 0.804 | 3.1e-03 | 1.0000 |
| TB1 vs R0 (fp32) | vitb16 | 0 | mid | 6.9e-05 | 0.99994 | 0.943 | 7.1e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitb16 | 1 | mid | 1.2e-04 | 0.99791 | 0.660 | 1.4e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitb16 | 2 | mid | 6.6e-05 | 0.99561 | 0.618 | 2.1e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitl16 | 0 | early | 3.1e-04 | 0.99996 | 0.968 | 4.9e-05 | 1.0000 |
| TB1 vs R0 (fp32) | vitl16 | 0 | late | 7.3e-05 | 0.99996 | 0.957 | 1.0e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitl16 | 0 | mid | 1.9e-04 | 0.99996 | 0.855 | 6.5e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitl16 | 1 | mid | 2.8e-05 | 0.99996 | 0.985 | 2.0e-04 | 1.0000 |
| TB1 vs R0 (fp32) | vitl16 | 2 | mid | 2.4e-04 | 0.99970 | 0.950 | 3.6e-05 | 1.0000 |

**BF16 self-noise** (same weights, batch and RNG tape; a different SDPA backend or mask form versus TB1). Global metrics stay at grad cos >= 0.999 and loss_rel <= 2e-4, but the per-parameter minimum cosine drops as low as 0.77, so the per-parameter threshold `>= 0.99` of `common/check.py` cannot be applied to two BF16 stacks (the rows that show pass=False in the raw JSON fail only on that threshold). This is the basis for the calibrated criterion of Amendment 1, section 9.4 in CONTRACTS.md.

| comparison | model | seed | phase | loss_rel | global grad cos | per-param min cos | grad_norm_rel | update cos min |
|---|---|---:|---|---:|---:|---:|---:|---:|
| noise: eff/eff vs TB1 | vitb16 | 0 | mid | 2.0e-04 | 0.99986 | 0.998 | 1.7e-04 | 1.0000 |
| noise: eff/eff vs TB1 | vitb16 | 1 | mid | 1.9e-05 | 0.99968 | 0.996 | 4.6e-04 | 1.0000 |
| noise: eff/eff vs TB1 | vitb16 | 2 | mid | 4.0e-05 | 0.99992 | 0.968 | 2.4e-04 | 1.0000 |
| noise: eff/eff vs TB1 | vitl16 | 0 | mid | 9.5e-05 | 0.99999 | 0.967 | 4.3e-04 | 1.0000 |
| noise: eff/eff vs TB1 | vitl16 | 1 | mid | 1.2e-04 | 0.99997 | 0.979 | 2.6e-04 | 1.0000 |
| noise: eff/eff vs TB1 | vitl16 | 2 | mid | 3.9e-05 | 0.99993 | 0.946 | 6.9e-04 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitb16 | 0 | mid | 4.6e-05 | 0.99983 | 0.996 | 6.4e-05 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitb16 | 1 | mid | 8.2e-05 | 0.99935 | 0.965 | 1.0e-03 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitb16 | 2 | mid | 1.1e-04 | 0.99968 | 0.989 | 2.0e-03 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitl16 | 0 | mid | 8.9e-05 | 0.99999 | 0.988 | 7.9e-05 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitl16 | 1 | mid | 5.0e-05 | 0.99947 | 0.985 | 1.6e-03 | 1.0000 |
| noise: cudnn+bias vs TB1 | vitl16 | 2 | mid | 1.2e-04 | 0.99998 | 0.769 | 1.1e-04 | 1.0000 |

**S, 50-step trajectories on 2 real batches** (`common/check.curve`, `gate_c`: EMA(0.9) at checkpoints every 10 steps, relative deviation <= 5%, no NaN/Inf), TB1 vs R0. `schedule` follows the mask schedule; `fixed_mid` / `fixed_late` hold the phase.

| model | run | steps | max EMA rel. dev. | grad-norm same order | finite | pass |
|---|---|---:|---:|---|---|---|
| vitb16 | schedule | 50 | 6.6e-04 | True | True | True |
| vitb16 | fixed_mid | 50 | 1.8e-04 | True | True | True |
| vitb16 | fixed_late | 50 | 1.5e-04 | True | True | True |
| vitl16 | schedule | 50 | 3.7e-04 | True | True | True |
| vitl16 | fixed_mid | 50 | 3.5e-04 | True | True | True |
| vitl16 | fixed_late | 50 | 4.2e-04 | True | True | True |

### 4.6 torch.compile regional reference

Regional `torch.compile` (one compiled region per block, default mode; stack `r1,r3,r7,r11:autocast+compile_block`) on ViT-L: weighted 37.19 img/s, mid step 188.5 ms, peak 19.3 GB. That is faster than plain autocast + SDPA backends (`AC_SDPA`, mid 234 ms in 4.2) but 10% slower than T-BF16 (41.40 img/s). It does not include r5, `rope_triton`, the attention backend choice or `head_ln_bf16`, so it is a reference point only; the hand-composed torch stack wins here. The `max-autotune-no-cudagraphs` variant was not run (trimmed: long compile times and flaky under shared GPU use). ViT-B compile was not run.

### 4.7 Memory and max batch

ViT-L, T-BF16, largest batch that fits on the 32 GB card (secondary metric; the headline numbers use the fixed B=7). FP32 P0 at B=7 uses 27.1 GB; T-BF16 uses 17.5 GB. The FP32 maximum batch was not measured (trimmed).

| batch | status | img/s (mid) | peak allocated GB |
|---:|---|---:|---:|
| 7 | ok | 40.6 | 17.5 |
| 8 | ok | 40.9 | 19.2 |
| 10 | ok | 42.4 | 22.7 |
| 12 | ok | 42.6 | 26.1 |
| 14 | ok | 43.8 | 29.6 |
| 16 | OOM | - | - |

Batch 14 fits and runs 8% faster per image than B=7 with this stack; changing the batch also changes the training recipe (learning-rate scaling, steps per epoch), so it is not used for the headline comparison.

## 5. Correctness

- CPU tests: 20 passed (`PYTHONPATH=train-opt:src python -m pytest -q train-opt/round11_bf16_mixed/tests/test_round11_cpu.py`).
- GPU tests: 56 passed on the old machine (operator level vs fp32 reference; they hold the dual lock). Not re-run on the new machine.
- Step-level gates: see 4.5 (B-lite and S pass for both models).

## 6. Static analysis

No new CUDA/Triton kernel in this round (reuses round4's RoPE Triton kernel and torch/cuDNN SDPA); no ncu.

## 7. Conclusion, failures, next bottleneck

- **Result.** The torch BF16 mainline (T-BF16, TB1) reaches 41.4 img/s on ViT-L (2.56x R0, 2.31x P0) and 96.0 img/s on ViT-B (2.58x R0, 2.16x P0) on the same machine, with peak memory 17.5 / 15.3 GB (R0: 28.8 / 29.1 GB). Gates: B-lite and S pass for both models (4.5).
- **What paid off** (ViT-L, mid, ms): autocast plus the attention path (flash for mask-free, `joint_qsplit` with cuDNN for the 100 masked rows) and fused RoPE took the FP32 P0 stack from 382 to 185 ms (P0 under autocast) and then to 169 ms (TB1). r3 batched loss is still the largest single item (+21 ms ViT-L / +35 ms ViT-B without it); r5 `ln_fuse` gives about 6 ms; r1 gives 1-2 ms.
- **What did not**: `bf16_shadow` (+6 ms, the post-optimizer refresh costs as much as autocast's cached cast saves), `head_cl`, `cudnn_bench`, joint backend variants (all within 1-2 ms), per-block `torch.compile` (10% slower than TB1). r7 `capturable_torch` costs about 3 ms in eager mode (its `cuda_multi_tensor` variant is 3.5 ms faster); it stays in the published stack because round16 (full graph) needs a capturable optimizer.
- **Next bottlenecks** (nsys, 4.4): GEMM (44% / 30% of GPU time) -> round13 fused block and round17 MXFP8; the head (convolutions plus elementwise, 14-22% of GPU time plus part of the elementwise bucket) -> round14; optimizer and loss launch overhead leave the GPU idle (CPU 59 ms vs GPU 11 ms in `clip_opt_sched`; about 3600 kernels per step) -> round15 flat optimizer and round16 whole-step graph; attention is already near the BF16 roofline (the round12 fused kernel is the remaining ~5% option).
- **Failures and caveats.** round5 `shadow.setup` matched no parameter under the Fabric wrapper (fixed in `impl/shadow.py`); `head_ln_bf16` v1 with fp32 LayerNorm parameters failed in torch (no mixed bf16-input/fp32-weight path), fixed by rounding the affine parameters to bf16; two screening runs were aborted when another agent's process shared the GPU (retried). The 56 GPU operator tests were run on the old machine only and were not re-run on the new one. The final matrix, nsys tables, gates, compile reference and max-batch sections were completed by the coordinator from the logged measurements after the agent was cut off by the session limit.
- **Published for the next rounds:** `common/config.py` `P1_STACK` / `P1_PATCH_KWARGS` (TB1).
