#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python decision.py stage1 > /dev/null
python tests/test_graph_g.py
python verify_plans.py vitb16_safe_margin vitb16_fast_margin vitl16_safe_margin vitl16_fast_margin
python verify_plans.py --real vitb16_safe_margin vitb16_fast_margin vitl16_safe_margin vitl16_fast_margin
python verify_plans.py vitb16_safe_bare vitb16_fast_bare vitl16_safe_bare vitl16_fast_bare
