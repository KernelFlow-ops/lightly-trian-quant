#!/bin/bash
# round22 gates queue 3: extra batches for the gate B margin (seeds 8-15, other loader draws), dY-headroom 16 S gate on ViT-L (the model with residual saturations).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
step gb_more_vitb python tests/gate_b_real.py --model vitb16 --seeds 8 9 10 11 12 13 14 15 --cands SIM_B3 REAL_B3 REAL_B4 --out profile/gateB_r22_vitb16_seeds8_15.json
step gb_more_vitl python tests/gate_b_real.py --model vitl16 --seeds 8 9 10 11 12 13 14 15 --cands SIM_B4 REAL_B4 --out profile/gateB_r22_vitl16_seeds8_15.json
step gs_vitl_hr16 python run.py --model vitl16 --stack "FINAL,round22_fp4_backbone:all.d=nv4+sr+all.w=nv4+sr+nofirst+dyhr=16" --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
echo GATES3_DONE $(date +%T)
