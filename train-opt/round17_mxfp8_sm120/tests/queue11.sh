#!/bin/bash
# round17 queue 11: Gate S fixed-late phase for the recommended policy (R0/TB1/candidate on the same two batches), both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
P=MX8_fc+proj+qkv_dw
for m in vitl16 vitb16; do
  python tests/gate_s_multi.py --model $m --cands $P --tag w3 --phases late --out profile/gateS_${m}_mx8_rec_late.json > $LOG/q11_gateS_$m.log 2>&1
done
echo QUEUE11_DONE $(date +%T)
