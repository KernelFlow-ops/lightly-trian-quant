#!/usr/bin/env bash
# int8 prototype: tile sweep (compile on CPU first, no GPU), then stage1 + stage2 timing (GPU, dual lock via run_locked)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python tune8.py --phase stage1 --Ms 2058
python tune8.py --phase stage2 --Ms 2058,2258
