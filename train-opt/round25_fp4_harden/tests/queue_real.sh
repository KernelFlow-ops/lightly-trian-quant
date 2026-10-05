#!/bin/bash
# round25 task 1.1 / 1.2 extra: dY saturation with the REAL loader (random batches, real schedule, first 200 steps) at headroom 8 / 32 on ViT-L (run_final.py prints FP4State stats at the end); 8-batch rotation study.
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
for s in P3 P3_hr32; do
  step real_vitl_$s "" python run_final.py --model vitl16 --stack $s --steps 200 --workers 16 --persistent --json profile/realsat_vitl16_${s}_$(date +%H%M%S).json
done
step sat_vitl_n8 "" python run.py --model vitl16 --action sat_study --scenarios schedule --seq P3 P3B --n-batches 8 --tag schedule_n8
echo REAL_DONE $(date +%T)
