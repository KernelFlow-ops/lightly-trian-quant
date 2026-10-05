#!/bin/bash
# Part 0: round13 closure gates (TB1+R13 vs TB1 and R0): gate B (3 seeds, calibrated) and gate S (50 steps, 2 batches), both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
R13P=/root/lightly-train/train-opt/round13_gemm_fusion_sm120/profile
echo "gateB vitl $(date +%T)"
python tests/gate_b_multi.py --model vitl16 --cands TB1_R13 --seeds 0 1 2 --out $R13P/gateB_multi_vitl16.json > $LOG/p0_gateB_vitl.log 2>&1
echo "gateB vitb $(date +%T)"
python tests/gate_b_multi.py --model vitb16 --cands TB1_R13 --seeds 0 1 2 --out $R13P/gateB_multi_vitb16.json > $LOG/p0_gateB_vitb.log 2>&1
echo "gateS vitl $(date +%T)"
python tests/gate_s_multi.py --model vitl16 --cands TB1_R13 --out $R13P/gateS_multi_vitl16.json --tag p0 > $LOG/p0_gateS_vitl.log 2>&1
echo "gateS vitb $(date +%T)"
python tests/gate_s_multi.py --model vitb16 --cands TB1_R13 --out $R13P/gateS_multi_vitb16.json --tag p0 > $LOG/p0_gateS_vitb.log 2>&1
echo QUEUE0_DONE $(date +%T)
