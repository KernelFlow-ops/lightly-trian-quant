#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
N="vitb16_safe_recommended vitb16_fast_recommended vitl16_safe_strict vitl16_fast_strict vitl16_safe_recommended vitl16_fast_recommended"
python verify_plans.py $N
python verify_plans.py --real $N
MG16_TUNED=tuned16_pkt1.json python verify_plans.py vitl16_safe_strict vitl16_safe_recommended vitb16_safe_recommended
