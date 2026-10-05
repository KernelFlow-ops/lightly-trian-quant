#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python tune16.py --phase stage2 --Ms 2058,2258
python tune16.py --phase stage2 --Ms 2058,2258 --pkt1
