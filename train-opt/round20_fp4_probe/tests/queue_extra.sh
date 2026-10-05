#!/bin/bash
# extra measurements after queue_l: single-process ViT-B finals, control arms (B4 base, >= 8 batches, paired), ViT-B S gate for B4 / exact-amax arms
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
R="bash tests/retry.sh"
until [ -f $L/r20b_queue_l_done.flag ]; do sleep 10; done
[ -f profile/gateB_r20b_vitb16_finals_mixedproc.json ] || mv profile/gateB_r20b_vitb16_finals.json profile/gateB_r20b_vitb16_finals_mixedproc.json
$R $L/r20b_gateB_vitb16_finals3.log python tests/gate_b_multi.py --model vitb16 --cands H1 H2 H3 B3 B4 B5 --seeds 0 1 2 3 4 5 6 7 --per-param --out profile/gateB_r20b_vitb16_finals.json
$R $L/r20b_gateB_vitb16_arms.log python tests/gate_b_multi.py --model vitb16 --cands B4 B4+mx4 B4+rn B4+w1d B4+norht --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_r20b_vitb16_arms.json
$R $L/r20b_gateB_vitl16_arms.log python tests/gate_b_multi.py --model vitl16 --cands B4 B4+mx4 B4+rn B4+w1d B4+norht --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_r20b_vitl16_arms.json
$R $L/r20b_gateS_vitb16_c.log python tests/gate_s_multi.py --model vitb16 --cands B4 B3x B4x --scenarios schedule compressed --out profile/gateS_r20b_vitb16_c.json
$R $L/r20b_gateS_vitb16_d.log python tests/gate_s_multi.py --model vitb16 --cands B4 B3x B4x --scenarios fixed_mid fixed_late --out profile/gateS_r20b_vitb16_d.json
echo done > $L/r20b_queue_extra_done.flag
