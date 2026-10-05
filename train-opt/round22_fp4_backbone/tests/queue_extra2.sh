#!/bin/bash
# round22: S gate of the RN arm (ViT-B P3B_rnB, graph, 4 scenarios): is RN (no Philox cost, +1.4% step speed) distinguishable from SR over 50 steps?  Runs after queue_extra.sh.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
until grep -q EXTRA_DONE /queue_extra.log; do sleep 10; done
echo == gs_vitb_rn $(date +%T)
tests/retry.sh $LOG/gs_vitb_rn.log python run.py --model vitb16 --stack P3B_rnB --runner graph --action gate_s --ref R0 --scenarios schedule compressed fixed_mid fixed_late
echo EXTRA2_DONE $(date +%T)
