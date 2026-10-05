#!/bin/bash
# sequential gate-B screening jobs (each python process holds the dual GPU lock for its whole life, <= ~10 min each)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
C="H1 H2 H3 B1 B2 B3 B4 B5 B3+mx4 B3+rn B3+w1d B3+norht B3+fcF4 B3+qkvF8 B4+mx4 B4+rn B4+w1d B4+norht B4+fcF4 B4+qkvF8 B5+mx4 B5+rn B5+w1d B5+norht B5+fcF4 B5+qkvF8"
python tests/gate_b_multi.py --model vitb16 --cands $C --seeds 2 --out profile/gateB_r20b_vitb16_screen.json >> $L/r20b_gateB_vitb16_screen.log 2>&1
python tests/gate_b_multi.py --model vitl16 --cands $C --seeds 0 1 --out profile/gateB_r20b_vitl16_screen.json > $L/r20b_gateB_vitl16_screen.log 2>&1
python tests/gate_b_multi.py --model vitl16 --cands $C --seeds 2 --out profile/gateB_r20b_vitl16_screen.json >> $L/r20b_gateB_vitl16_screen.log 2>&1
echo done > $L/r20b_screen_done.flag
