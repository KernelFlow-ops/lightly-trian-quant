#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python robust_plans.py --model vitl16
python robust_plans.py --model vitb16
