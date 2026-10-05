#!/bin/bash
# round25 task 1.1: saturation study.  Same process, same 2 batches, same seed: P3B, P3, P3B, P3 (repeats = run-to-run noise floor), with per-step per-site amax traces.
# One process per scenario (each process draws its own pair of batches: the loader is not deterministic across processes).  usage: queue_sat.sh
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
step sat_vitl_schedule_a "" python run.py --model vitl16 --action sat_study --scenarios schedule --seq P3B P3 P3B P3 --tag schedule_a
step sat_vitl_fixedmid_a "" python run.py --model vitl16 --action sat_study --scenarios fixed_mid --seq P3B P3 P3B P3 --tag fixedmid_a
step sat_vitl_schedule_b "" python run.py --model vitl16 --action sat_study --scenarios schedule --seq P3B P3 P3B P3 --tag schedule_b
step sat_vitb_schedule_a "" python run.py --model vitb16 --action sat_study --scenarios schedule --seq P3B P3 P3B P3 --tag schedule_a
echo SAT_DONE $(date +%T)
