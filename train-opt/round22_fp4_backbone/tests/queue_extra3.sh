#!/bin/bash
# round22 (after the ViT-L seeds 8-15 draw showed one pathological batch, seed 13, where FINAL itself is already at 0.9715):
#  1. a third ViT-L draw (seeds 16-23) with diagnostic arms (dgrad-only / wgrad-only FP4, nojoint, nolast) and per-parameter cosines,
#  2. the RN arm S gate on ViT-B (is RN, +1.4% step speed, distinguishable from SR over 50 steps?),
#  3. dY headroom 16 S gate on ViT-L (the model with residual dY saturations).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
B3="all.d=nv4+sr+all.w=nv4+sr"; B4="$B3+nofirst"
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
step gb_draw3_vitl python tests/gate_b_real.py --model vitl16 --seeds 16 17 18 19 20 21 22 23 --per-param --cands SIM_B4 REAL_B4 "REAL_B4d=real:all.d=nv4+sr+nofirst" "REAL_B4w=real:all.w=nv4+sr+nofirst" "REAL_B4nj=real:$B4+nojoint" "REAL_B4nl=real:$B4+nolast" --out profile/gateB_r22_vitl16_draw3.json
step gs_vitb_rn python run.py --model vitb16 --stack P3B_rnB --runner graph --action gate_s --ref R0 --scenarios schedule compressed fixed_mid fixed_late
step gs_vitl_hr16 python run.py --model vitl16 --stack "FINAL,round22_fp4_backbone:$B4+dyhr=16" --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
echo EXTRA3_DONE $(date +%T)
