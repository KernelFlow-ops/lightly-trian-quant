#!/bin/bash
# round17 queue 9 (waits for queue 8): gate S with all stacks recomputed on the same two batches, both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
until grep -q QUEUE8_DONE $LOG/queue8.log; do sleep 10; done
echo "gateS vitl $(date +%T)"
python tests/gate_s_multi.py --model vitl16 --cands TB1_R13 MX8_fc MX8_fc+proj MX8_all --tag w1 --out profile/gateS_vitl16_mx8.json > $LOG/q9_gateS_vitl.log 2>&1
echo "gateS vitb $(date +%T)"
python tests/gate_s_multi.py --model vitb16 --cands TB1_R13 MX8_fc MX8_fc+proj MX8_all --tag w1 --out profile/gateS_vitb16_mx8.json > $LOG/q9_gateS_vitb.log 2>&1
echo QUEUE9_DONE $(date +%T)
