#!/bin/bash
# round22 gates queue (each python process takes both GPU locks through tests/retry.sh -> locked_run.sh; every step <= ~10 min)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
step ge_vitl python tests/graph_equiv22.py --model vitl16 --stack P3B --equiv-steps 8 --out profile/graph_equiv_vitl16_P3B.json
step gs_vitb_a python run.py --model vitb16 --stack P3B_B --runner graph --action gate_s --ref R0 --scenarios schedule compressed
step gs_vitb_b python run.py --model vitb16 --stack P3B_B --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
step gs_vitl_a python run.py --model vitl16 --stack P3B --runner graph --action gate_s --ref R0 --scenarios schedule compressed
step gs_vitl_b python run.py --model vitl16 --stack P3B --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
echo GATES_DONE $(date +%T)
