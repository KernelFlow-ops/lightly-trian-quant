#!/bin/bash
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
C="H1 H2 H3 B1 B2 B3 B4 B5 B3+mx4 B3+rn B3+w1d B3+norht B3+fcF4 B3+qkvF8 B4+mx4 B4+rn B4+w1d B4+norht B4+fcF4 B4+qkvF8 B5+mx4 B5+rn B5+w1d B5+norht B5+fcF4 B5+qkvF8"
bash tests/retry.sh $L/r20b_gateB_vitl16_screen2.log python tests/gate_b_multi.py --model vitl16 --cands $C --seeds 0 1 --out profile/gateB_r20b_vitl16_screen.json
echo done > $L/r20b_l_screen_rest_done.flag
