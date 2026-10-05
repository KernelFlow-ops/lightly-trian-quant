#!/bin/bash
# ViT-B finals: B gate seeds 3-7 for the finalists (seeds 0-2 come from the screen json), then the S gate (2 scenarios per process, <= ~10 min each)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
until [ -f $L/r20b_screen_done.flag ]; do sleep 5; done
[ -f profile/gateB_r20b_vitb16_finals.json ] || cp profile/gateB_r20b_vitb16_screen.json profile/gateB_r20b_vitb16_finals.json
python tests/gate_b_multi.py --model vitb16 --cands B3 B5 H1 H3 --seeds 0 1 2 3 4 5 6 7 --per-param --out profile/gateB_r20b_vitb16_finals.json > $L/r20b_gateB_vitb16_finals.log 2>&1
python tests/gate_s_multi.py --model vitb16 --cands B3 H3 --scenarios schedule compressed --out profile/gateS_r20b_vitb16_a.json > $L/r20b_gateS_vitb16_a.log 2>&1
python tests/gate_s_multi.py --model vitb16 --cands B3 H3 --scenarios fixed_mid fixed_late --out profile/gateS_r20b_vitb16_b.json > $L/r20b_gateS_vitb16_b.log 2>&1
echo done > $L/r20b_b_finals_done.flag
