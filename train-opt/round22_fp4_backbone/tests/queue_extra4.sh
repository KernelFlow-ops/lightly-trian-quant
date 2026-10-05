#!/bin/bash
# round22: after the draw-3 gate B (diagnostic arms showed wgrad-only FP4 ~ FINAL, dgrad FP4 carries the accuracy cost): RN-arm S gate on ViT-B (2 scenarios) and same-session
# benches of the wgrad-only FP4 policy (all.w=nv4+sr, nofirst on ViT-L) against FINAL, both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
until ! pgrep -f 'gb_draw3_vitl.log' > /dev/null; do sleep 5; done
echo == gs_vitb_rn $(date +%T)
tests/retry.sh $LOG/gs_vitb_rn.log python run.py --model vitb16 --stack P3B_rnB --runner graph --action gate_s --ref R0 --scenarios schedule fixed_mid
echo == bench $(date +%T)
tests/queue_bench.sh vitl16:FINAL:graph "vitl16:FINAL,round22_fp4_backbone:all.w=nv4+sr+nofirst:graph" vitb16:FINAL:graph "vitb16:FINAL,round22_fp4_backbone:all.w=nv4+sr:graph"
echo EXTRA4_DONE $(date +%T)
