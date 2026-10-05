#!/bin/bash
# round25 task 1.1 (more batch pairs): each process draws its own pair of real batches; P3B / P3 on the same pair, schedule scenario, per-step amax trace.
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
for t in c d e; do
  step sat_vitl_schedule_$t "" python run.py --model vitl16 --action sat_study --scenarios schedule --seq P3B P3 --tag schedule_$t
done
step sat_vitb_schedule_b "" python run.py --model vitb16 --action sat_study --scenarios schedule --seq P3B P3 --tag schedule_b
echo SAT2_DONE $(date +%T)
