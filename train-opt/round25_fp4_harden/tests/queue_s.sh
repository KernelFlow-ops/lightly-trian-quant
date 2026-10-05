#!/bin/bash
# round25 task 1.2: S gate (4 scenarios x 50 steps, 2 batches) of the dY headroom settings under StepGraphRunner against R0 (eager).  ONE process holds the same 2 batches for all stacks (R0 once per scenario).
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
S="P3 P3_hr16 P3_hr32 P3_ex"
step gs_vitl_a "" python run.py --model vitl16 --action gate_s_sweep --stacks $S --scenarios schedule compressed --tag a
step gs_vitl_b "" python run.py --model vitl16 --action gate_s_sweep --stacks $S --scenarios fixed_mid fixed_late --tag b
echo S_DONE $(date +%T)
