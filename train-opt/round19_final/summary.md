# round19_final summary (waterfall and final-stack tables)

Weighted = 0.2 early + 0.6 mid + 0.2 late (step-time weighted), B fixed, warmup 20, 50 x 3, GPU exclusive, `common.bench.time_steps`. Rows marked *(this round)* were measured in round19 on this container with one harness; the earlier-stage rows are the numbers published by each round (same hardware class, measured at other times: +-2-3% drift) and are listed for the waterfall only. All ratios use the round19 R0 re-measure as denominator.

## ViT-L/16 B=7

| stage | weighted img/s | mid step ms | x vs R0 (this round) | x vs previous row | source |
|---|---:|---:|---:|---:|---|
| R0 (original Lightly step, fp32/TF32) *(this round, re-measured)* | 16.31 | 429.1 | 1.00x | - | round19 bench |
| P0 (r1+r3+r4+r5+r7, fp32) | 18.00 | 382.0 | 1.10x | 1.104x | CONTRACTS 9.3 / PLAN |
| TB1 (+ BF16 autocast, flash/cuDNN attn, triton RoPE, head LN bf16) | 41.40 | 169.0 | 2.54x | 2.300x | CONTRACTS 9.3 |
| + r13 fused block (P1R13) | 43.70 | 158.0 | 2.68x | 1.056x | CONTRACTS 10.1 / 9.5 |
| + r17 MXFP8 fc+proj+qkv_dw (P2, eager) | 51.70 | 135.6 | 3.17x | 1.183x | round16 NOTES 4.1 |
| + r15 flat optimizer (eager) | 56.00 | 125.1 | 3.43x | 1.083x | round16 NOTES 4.1 |
| + r16 whole-step CUDA graph (P2+r15 graph) | 59.50 | 117.8 | 3.65x | 1.062x | round16 NOTES 4.1 |
|   P2+r15 graph re-measured *(this round, drift check)* | 59.37 | 118.1 | 3.64x | 0.998x | round19 bench |
| + r14 fused head, eager (P2+r14+r15, no graph) *(this round)* | 65.46 | 107.1 | 4.01x | 1.100x | round19 bench |
| **+ r14 fused head under the graph = FINAL** *(this round)* | 68.87 | 101.8 | 4.22x | 1.052x | round19 bench |

Final stack, this round (all numbers measured here):

| runner | early ms (img/s) | mid ms (img/s) | late ms (img/s) | weighted img/s | peak allocated GB | peak reserved GB |
|---|---|---|---|---:|---:|---:|
| R0 eager | 429.4 (16.3) | 429.1 (16.3) | 429.5 (16.3) | 16.31 | 28.81 | 30.02 |
| FINAL eager | 107.3 (65.2) | 107.1 (65.4) | 106.2 (65.9) | 65.46 | 15.56 | 15.81 |
| FINAL graph | 101.8 (68.8) | 101.8 (68.8) | 101.2 (69.2) | 68.87 | 6.09 | 16.50 |
| FINAL_fc graph (r17 policy fc) | 108.4 (64.6) | 108.4 (64.6) | 107.7 (65.0) | 64.65 | 5.90 | 16.12 |
| FINAL_noMX graph (bf16 only) | 128.9 (54.3) | 128.9 (54.3) | 128.0 (54.7) | 54.37 | 5.51 | 16.03 |
| P2+r15 graph (drift check) | 117.7 (59.5) | 118.1 (59.3) | 117.5 (59.6) | 59.37 | 6.09 | 18.12 |
| TB1 eager (drift check) | 168.1 (41.6) | 167.0 (41.9) | 165.7 (42.3) | 41.94 | 17.52 | 17.55 |

FINAL graph vs R0: 4.22x (16.31 -> 68.87 img/s); vs FINAL eager 1.052x; memory: R0 28.8 GB allocated -> FINAL 15.6 GB (eager peak allocated), 16.5 GB reserved under the graph.

## ViT-B/16 B=14

| stage | weighted img/s | mid step ms | x vs R0 (this round) | x vs previous row | source |
|---|---:|---:|---:|---:|---|
| R0 (original Lightly step, fp32/TF32) *(this round, re-measured)* | 37.77 | 370.7 | 1.00x | - | round19 bench |
| P0 (r1+r3+r4+r5+r7, fp32) | 44.40 | - | 1.18x | 1.176x | CONTRACTS 9.3 |
| TB1 (+ BF16 autocast, flash/cuDNN attn, triton RoPE, head LN bf16) | 96.00 | 146.0 | 2.54x | 2.162x | CONTRACTS 9.3 |
| + r13 fused block (P1R13) | 100.70 | 138.3 | 2.67x | 1.049x | CONTRACTS 10.1 / r14 NOTES |
| + r17 MXFP8 fc+proj+qkv_dw (P2, eager) | 109.90 | 127.6 | 2.91x | 1.091x | round16 NOTES 4.1 |
| + r15 flat optimizer (eager) | 116.60 | 120.3 | 3.09x | 1.061x | round16 NOTES 4.1 |
| + r16 whole-step CUDA graph (P2+r15 graph) | 123.10 | 113.9 | 3.26x | 1.056x | round16 NOTES 4.1 |
|   P2+r15 graph re-measured *(this round, drift check)* | 123.75 | 113.3 | 3.28x | 1.005x | round19 bench |
| + r14 fused head, eager (P2+r14+r15, no graph) *(this round)* | 152.08 | 92.3 | 4.03x | 1.235x | round19 bench |
| **+ r14 fused head under the graph = FINAL** *(this round)* | 161.94 | 86.6 | 4.29x | 1.065x | round19 bench |

Final stack, this round (all numbers measured here):

| runner | early ms (img/s) | mid ms (img/s) | late ms (img/s) | weighted img/s | peak allocated GB | peak reserved GB |
|---|---|---|---|---:|---:|---:|
| R0 eager | 369.8 (37.9) | 370.7 (37.8) | 371.6 (37.7) | 37.77 | 29.20 | 29.70 |
| FINAL eager | 93.0 (150.5) | 92.3 (151.7) | 90.4 (154.8) | 152.08 | 11.95 | 12.15 |
| FINAL graph | 87.2 (160.5) | 86.6 (161.7) | 85.4 (164.0) | 161.94 | 2.07 | 12.52 |
| FINAL_fc graph (r17 policy fc) | 90.0 (155.6) | 89.6 (156.2) | 88.0 (159.2) | 156.67 | 2.02 | 12.43 |
| FINAL_noMX graph (bf16 only) | 100.8 (138.9) | 100.5 (139.3) | 99.1 (141.3) | 139.63 | 1.91 | 12.76 |
| P2+r15 graph (drift check) | 113.9 (122.9) | 113.3 (123.5) | 111.8 (125.3) | 123.75 | 2.07 | 15.35 |
| TB1 eager (drift check) | 144.6 (96.8) | 143.1 (97.8) | 140.7 (99.5) | 97.93 | 15.38 | 15.44 |

FINAL graph vs R0: 4.29x (37.77 -> 161.94 img/s); vs FINAL eager 1.065x; memory: R0 29.2 GB allocated -> FINAL 12.0 GB (eager peak allocated), 12.5 GB reserved under the graph.

Note: `peak allocated` of a graph run does not contain the graph's private pool (round16 NOTES 5.5), so the graph row's memory figure that counts is `peak reserved`.

## End to end with the real DataLoader (200 steps, FINAL under the graph runner)

`img/s steady` = steps 21-200 (sync at both ends); `incl. startup` = all 200 steps from before the first `next()` (the `time_e2e` pattern, includes DataLoader worker spawn + first batch). GPU-only = preloaded batches, early phase, same process.

### ViT-L/16 B=7: GPU-only early 100.9 ms = 69.4 img/s

| workers | prefetch | pin | persistent | img/s steady | steady / GPU-only | img/s incl. startup | first batch s | host wait mean / p95 / max ms |
|---:|---:|---|---|---:|---:|---:|---:|---|
| 16 | 2 | on | no | 68.26 | 0.984 | 64.62 | 1.15 | 3.5 / 15.0 / 15.9 |
| 32 | 2 | on | no | 67.90 | 0.978 | 61.81 | 2.04 | 4.0 / 15.0 / 31.4 |
| 64 | 2 | on | no | 67.92 | 0.979 | 54.13 | 4.85 | 2.6 / 14.8 / 17.8 |
| 32 | 4 | on | no | 67.83 | 0.977 | 60.99 | 2.32 | 4.3 / 15.5 / 18.5 |
| 32 | 8 | on | no | 67.61 | 0.974 | 61.75 | 1.94 | 2.1 / 14.2 / 16.8 |
| 32 | 2 | off | no | 64.77 | 0.933 | 60.76 | 1.37 | 106.8 / 109.4 / 112.9 |
| 32 | 2 | on | yes | 67.64 | 0.975 | 61.31 | 2.14 | 1.1 / 1.9 / 14.9 |

### ViT-B/16 B=14: GPU-only early 88.3 ms = 158.5 img/s

| workers | prefetch | pin | persistent | img/s steady | steady / GPU-only | img/s incl. startup | first batch s | host wait mean / p95 / max ms |
|---:|---:|---|---|---:|---:|---:|---:|---|
| 16 | 2 | on | no | 155.68 | 0.982 | 139.11 | 2.15 | 11.2 / 26.3 / 31.5 |
| 32 | 2 | on | no | 155.51 | 0.981 | 133.74 | 2.73 | 11.1 / 23.6 / 36.2 |
| 64 | 2 | on | no | 155.78 | 0.983 | 105.11 | 8.32 | 9.9 / 20.8 / 27.4 |
| 32 | 4 | on | no | 155.61 | 0.982 | 128.78 | 3.75 | 11.2 / 22.4 / 152.6 |
| 32 | 8 | on | no | 155.42 | 0.981 | 122.29 | 4.88 | 14.5 / 27.4 / 776.5 |
| 32 | 2 | off | no | 116.91 | 0.738 | 104.52 | 3.10 | 117.6 / 123.5 / 589.3 |
| 32 | 2 | on | yes | 155.64 | 0.982 | 122.48 | 4.87 | 9.0 / 22.9 / 33.8 |

`run_final.py --model vitl16 --steps 200 --workers 16 --persistent`: 67.87 img/s steady (103.1 ms/step), 67.93 img/s over the 199 steps after the first, peak reserved 16.57 GB, graph captures 1, eager fallbacks 0.
`run_final.py --model vitb16 --steps 200 --workers 16 --persistent`: 155.78 img/s steady (89.9 ms/step), 155.88 img/s over the 199 steps after the first, peak reserved 12.82 GB, graph captures 1, eager fallbacks 0.

## Validation pass and checkpoint (one each, after 20 real steps; Lightly helpers, unpatched fp32 validation path)

| model | val pass s (477 images) | checkpoint save s | export save s | checkpoint GB | export GB |
|---|---:|---:|---:|---:|---:|
| ViT-L/16 B=7 | 12.2 | 7.94 | 1.50 | 3.52 | 1.18 |
| ViT-B/16 B=14 | 8.7 | 2.36 | 0.34 | 1.03 | 0.35 |

## Max batch, ViT-L, FINAL under the graph (mid phase, warmup 20, 30 steps; B=7 row from the 3-phase bench)

| B | OOM | mid step ms | img/s | reserved GB | device used GB (mem_get_info) |
|---:|---|---:|---:|---:|---:|
| 7 | no | 101.8 | 68.79 | 16.50 | - |
| 9 | no | 130.0 | 69.24 | 19.30 | 19.99 |
| 10 | no | 144.2 | 69.36 | 20.69 | 21.38 |
| 11 | no | 159.3 | 69.06 | 22.02 | 22.71 |
| 12 | no | 172.5 | 69.58 | 23.43 | 24.12 |
| 13 | no | 185.4 | 70.11 | 24.88 | 25.57 |
| 14 | no | 199.7 | 70.10 | 26.25 | 26.94 |
| 15 | no | 213.5 | 70.26 | 27.58 | 28.27 |
| 16 | no | 228.0 | 70.17 | 29.08 | 29.77 |
| 17 | no | 241.9 | 70.27 | 30.43 | 31.12 |
| 18 | yes | - | - | 30.59 | 31.26 |

## Gates on the FINAL stack

### Gate B-lite, ViT-L/16 B=7, 8 batches (1 process, seeds 0-7; each seed is a different real batch)

| reference | grad cos median | grad cos min | loss_rel max | grad_norm_rel max | rule (median>=0.995, min>=0.98, loss_rel<=2e-2) |
|---|---:|---:|---:|---:|---|
| R0 (fp32) | 0.99649 | 0.98865 | 1.94e-03 | 0.0106 | PASS |
| P1/TB1 (bf16) | 0.99694 | 0.98805 | 1.97e-03 | 0.0095 | PASS |

per batch global grad cos vs R0: 0.9992, 0.9982, 0.9902, 0.9988, 0.9976, 0.9954, 0.9943, 0.9887; vs P1: 0.9992, 0.9981, 0.9912, 0.9991, 0.9970, 0.9939, 0.9968, 0.9880
P1 itself vs R0 on the same batches: 1.0000, 0.9997, 0.9993, 0.9997, 0.9993, 0.9991, 0.9977, 0.9990

### Gate B-lite, ViT-B/16 B=14, 8 batches (1 process, seeds 0-7; each seed is a different real batch)

| reference | grad cos median | grad cos min | loss_rel max | grad_norm_rel max | rule (median>=0.995, min>=0.98, loss_rel<=2e-2) |
|---|---:|---:|---:|---:|---|
| R0 (fp32) | 0.99451 | 0.98240 | 2.60e-03 | 0.0169 | FAIL |
| P1/TB1 (bf16) | 0.99419 | 0.98230 | 2.56e-03 | 0.0178 | FAIL |

per batch global grad cos vs R0: 0.9986, 0.9950, 0.9938, 0.9933, 0.9942, 0.9824, 0.9966, 0.9948; vs P1: 0.9975, 0.9942, 0.9896, 0.9932, 0.9942, 0.9823, 0.9961, 0.9946
P1 itself vs R0 on the same batches: 0.9993, 0.9987, 0.9969, 0.9995, 0.9997, 0.9987, 0.9993, 0.9994

### Gate B-lite bisect, ViT-B/16 B=14 (same batches for all candidates)

| candidate | batches | vs R0: median / min / loss_rel max | vs P1: median / min | PASS (R0 and P1) |
|---|---:|---|---|---|
| FINAL | 12 | 0.99241 / 0.98803 / 3.3e-03 | 0.99409 / 0.98873 | FAIL |
| FINAL_fc | 12 | 0.99467 / 0.99151 / 3.3e-03 | 0.99587 / 0.99204 | FAIL |
| FINAL_noR14 | 12 | 0.99180 / 0.98795 / 3.4e-03 | 0.99347 / 0.98836 | FAIL |
| FINAL_noMX | 12 | 0.99938 / 0.99853 / 3.5e-04 | 0.99946 / 0.99768 | PASS |

### Gate S: FINAL graph vs R0 eager, 50 steps, 2 real batches, EMA(0.9) deviation, worst of steps 10..50 (threshold 5e-2)

| model | schedule | fixed mid | fixed late | compressed (7 captures) |
|---|---:|---:|---:|---:|
| ViT-L/16 B=7 | 1.77e-03 PASS | 1.32e-03 PASS | 2.05e-03 PASS | 1.64e-03 PASS |
| ViT-B/16 B=14 | 2.26e-03 PASS | 9.39e-04 PASS | 1.78e-03 PASS | 7.93e-04 PASS |

### Graph vs eager, same RNG (first-step loss bitwise equal in every scenario)

| model | scenario | steps | graph vs eager: loss rel max / update rel L2 / update cos | eager vs eager: loss rel max / update rel L2 |
|---|---|---:|---|---|
| ViT-L/16 B=7 | fixed_mid | 6 | 3.2e-05 / 3.5e-03 / 0.999994 | 7.4e-07 / 8.8e-04 |
| ViT-L/16 B=7 | schedule | 6 | 8.6e-05 / 2.6e-03 / 0.999997 | 1.0e-04 / 2.8e-03 |
| ViT-L/16 B=7 | fixed_late | 6 | 6.6e-06 / 1.0e-03 / 0.999999 | 7.0e-06 / 9.9e-04 |
| ViT-B/16 B=14 | fixed_mid | 8 | 9.2e-05 / 1.8e-03 / 0.999998 | 3.6e-04 / 2.4e-02 |
| ViT-B/16 B=14 | schedule | 8 | 1.7e-04 / 7.5e-03 / 0.999972 | 2.3e-04 / 5.8e-03 |
| ViT-B/16 B=14 | fixed_late | 8 | 3.0e-05 / 1.6e-03 / 0.999999 | 3.0e-05 / 1.6e-03 |
