# Round1: PyTorch cleanup (Agent A measurement collaborator, 2026-10-02)

## 1. Goal and hypothesis

Targets PLAN H2/H3/H4/H6/H7/H8/H9. The source training step currently computes its model and loss in fp32 even when the Fabric configuration string is `bf16-mixed`; `default` keeps this behavior. `autocast_bf16` is an explicit experimental numerical change and is excluded from the mainline.

## 2. Changes

- `impl/patch.py`: independently selectable variants, composed with `+`; `mask_skip` automatically includes `nosync`. The patch does not replace Parameters or change state_dict keys/shapes.
- `nosync`: host float annealing and a host branch for mask disable. Unlike an unconditional random draw, it preserves the source's absence of a draw at prob=1.
- `mask_skip`: at prob=0, keeps the `(B,Q)` random draw but skips the dense mask. At prob>0, passes `(B,1,N,N)` directly to SDPA without expanding over heads.
- The source's `rand > 0` is false if a generated float is exactly zero (about 2^-24 probability per element); `mask_skip` treats prob=0 as fully unmasked. This rare numerical behavior requires C/D checks before inclusion.
- `rope_batch`: batches trigonometry for plain blocks only. It still calls `uniform_` once per block, in order; joint blocks use the source path so mask random draws remain interleaved.
- `sdpa_flash`, `sdpa_cudnn`, `sdpa_efficient`, `sdpa_math`: force one PyTorch backend for controlled fwd+bwd comparisons; unsupported shapes/backends must be recorded as failures.
- `adamw_fused`, `cudnn_bench`, `head_cl`: optimizer and cuDNN toggles. `kmask_bias` remains experimental; it specializes only modules whose whole bias mask is zero and must pass full B before inclusion.
- `loader`, `val_fast`, `ckpt_async`: adjust real loader workers or expose optional fast-loop hooks. `impl/io.py::AsyncCheckpointWriter` snapshots an exact checkpoint payload to CPU then saves on a worker thread.

## 3. nsys analysis process

Not yet collected. After M0, capture both models and early/mid/late through `common/profile_nsys.sh`, then run `common/nsys_analyze.py`. Record the exact command, CSV paths, sync call sites, GPU gaps, stage times, Top-20 kernels, controlled before/after differences, and the next bottleneck here.

## 4. Results

`results.json` contains `null` for every unmeasured performance and accuracy field. The source-fp32 `default` and every candidate need fixed-batch B-level checks before any candidate enters `OPT_EAGER_PATCHES`.

| variant | ViT-L weighted img/s | ViT-B weighted img/s | B gate | status |
|---|---:|---:|---|---|
| default | null | null | null | baseline harness semantics under M0 review |
| nosync, mask_skip, rope_batch, adamw_fused, SDPA, head_cl | null | null | null | implementation and CPU checks only |
| autocast_bf16, kmask_bias | null | null | null | experimental; outside mainline |

## 5. Correctness

- CPU unit suite: nine tests passed for source mask and RNG consumption, broadcast SDPA fwd/all grads, batched RoPE output/RNG, host annealing, zero masked bias effective update, default precision/idempotence, async CPU checkpoint snapshot, channels-last Parameter identity, and fused AdamW group/update semantics.
- No B/C/D-level GPU result is claimed. `autocast_bf16` had a preliminary B=1 gradient cosine failure reported by A (minimum 0.388), so it must not enter `opt-eager` without a separate numerical study.

## 6. Static analysis

No custom kernel was introduced. Static CUDA inspection is not applicable to this round.

## 7. Conclusion and next bottleneck

Await M0 source-equivalent harness and locked early/mid/late profiles. Select `opt-eager` only from variants with B pass and real throughput gain. Validation and asynchronous checkpoint hooks require integration-loop wiring and epoch-level timing.
