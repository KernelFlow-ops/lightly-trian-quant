#!/bin/bash
# usage: run_mg3.sh <model> <M...>
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/lightly-train/infer-opt/round5_gemm_sm120
model=$1; shift
for M in "$@"; do
  for kind in fc1 fc2 proj qkv; do
    python tune_mg3.py $kind $model $M A > logs/tunemg3_${kind}_${model}_M${M}_A.log 2>&1
  done
  for kind in fc2 proj; do
    python tune_mg3.py $kind $model $M B > logs/tunemg3_${kind}_${model}_M${M}_B.log 2>&1
  done
done
echo done > logs/mg3_${model}_done
