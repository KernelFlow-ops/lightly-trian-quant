#!/bin/bash
# round17 queue 5 (waits for queue 4): host profile of the block, then gate S (50 steps x 2 batches) for the MX8 policies, both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
until grep -q QUEUE4_DONE $LOG/queue4.log; do sleep 10; done
python tests/profile_host_block.py --variant all > $LOG/q5_host_all.log 2>&1
python tests/profile_host_block.py --kind r13 --variant none > $LOG/q5_host_r13.log 2>&1
echo "gateS vitl $(date +%T)"
python tests/gate_s_multi.py --model vitl16 --cands TB1_R13 MX8_fc MX8_fc+proj MX8_all --tag v3 --out profile/gateS_vitl16_mx8.json > $LOG/q5_gateS_vitl.log 2>&1
echo "gateS vitb $(date +%T)"
python tests/gate_s_multi.py --model vitb16 --cands TB1_R13 MX8_fc MX8_fc+proj MX8_all --tag v3 --out profile/gateS_vitb16_mx8.json > $LOG/q5_gateS_vitb.log 2>&1
echo QUEUE5_DONE $(date +%T)
