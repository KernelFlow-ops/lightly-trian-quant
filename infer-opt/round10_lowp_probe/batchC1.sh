#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
rm -f /root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/tmp/tune16_stage1.json
python tune16.py --phase stage1 --Ms 2058
