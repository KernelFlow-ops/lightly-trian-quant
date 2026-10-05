#!/bin/bash
# round25 task 3: P3_q8 (ViT-L) validation that round23 did not do: graph_equiv, run_final (20 real steps + val + checkpoint + export) + check_export / resume, e2e 200 steps.
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
m=vitl16
step q8_ge "" python run.py --model $m --stack P3_q8 --action graph_equiv --equiv-steps 8
rm -rf $T/ckpt/$m
step q8_exp_run "" python run_final.py --model $m --stack P3_q8 --steps 20 --val --save-ckpt $T/ckpt/$m --workers 16 --persistent --json profile/run_final_${m}_P3_q8_export.json
step q8_exp_check "" python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model $m --ckpt-dir $T/ckpt/$m --out profile/export_${m}_P3_q8.json
step q8_e2e "" python run.py --model $m --stack P3_q8 --action e2e --steps 200 --loader-settings 16:2:1:1 --tag P3_q8
echo T3_DONE $(date +%T)
