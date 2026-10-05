#!/bin/bash
# round17 queue 1: block-level gate A, gate B-lite (MX8_all, MX8_fc) both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
echo "blockA $(date +%T)"
python tests/test_block_mx8.py --variants all fc --json profile/gateA_block_mx8.json > $LOG/q1_blockA.log 2>&1
echo "gateB vitl $(date +%T)"
python tests/gate_b_multi.py --model vitl16 --cands TB1_R13 MX8_all MX8_fc --seeds 0 1 2 --out profile/gateB_vitl16_mx8_v1.json > $LOG/q1_gateB_vitl.log 2>&1
echo "gateB vitb $(date +%T)"
python tests/gate_b_multi.py --model vitb16 --cands TB1_R13 MX8_all MX8_fc --seeds 0 1 2 --out profile/gateB_vitb16_mx8_v1.json > $LOG/q1_gateB_vitb.log 2>&1
echo QUEUE1_DONE $(date +%T)
