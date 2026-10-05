#!/bin/bash
# ViT-L completion queue (each python process holds the dual GPU lock for its whole life; retry.sh handles assert_gpu_exclusive aborts)
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
R="bash tests/retry.sh"
C="H1 H2 H3 B1 B2 B3 B4 B5 B3+mx4 B3+rn B3+w1d B3+norht B3+fcF4 B3+qkvF8 B4+mx4 B4+rn B4+w1d B4+norht B4+fcF4 B4+qkvF8 B5+mx4 B5+rn B5+w1d B5+norht B5+fcF4 B5+qkvF8"
$R $L/r20b_gateB_vitl16_screen3.log python tests/gate_b_multi.py --model vitl16 --cands $C --seeds 1 --out profile/gateB_r20b_vitl16_screen.json
$R $L/r20b_gateB_vitb16_finals2.log python tests/gate_b_multi.py --model vitb16 --cands B4 H2 --seeds 3 4 5 6 7 --per-param --out profile/gateB_r20b_vitb16_finals.json
[ -f profile/gateB_r20b_vitl16_finals.json ] || cp profile/gateB_r20b_vitl16_screen.json profile/gateB_r20b_vitl16_finals.json
$R $L/r20b_gateB_vitl16_finals.log python tests/gate_b_multi.py --model vitl16 --cands H1 H2 H3 B3 B4 B5 --seeds 0 1 2 3 4 5 6 7 --per-param --out profile/gateB_r20b_vitl16_finals.json
$R $L/r20b_gateS_vitl16_a.log python tests/gate_s_multi.py --model vitl16 --cands B3 H3 B4 --scenarios schedule compressed --out profile/gateS_r20b_vitl16_a.json
$R $L/r20b_gateS_vitl16_b.log python tests/gate_s_multi.py --model vitl16 --cands B3 H3 B4 --scenarios fixed_mid fixed_late --out profile/gateS_r20b_vitl16_b.json
echo done > $L/r20b_queue_l_done.flag
