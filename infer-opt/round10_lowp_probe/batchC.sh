#!/usr/bin/env bash
# tile sweep stage 2 (re-time best-of-stage-1 + r5_v1 + cublas fp16acc, 3 interleaved rounds), M = 2058, 2258
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python tune16.py --phase stage2 --Ms 2058,2258
