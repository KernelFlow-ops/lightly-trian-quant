#!/bin/bash
# usage: run_mg3b.sh <model> <phaseSpec> <M...>   phaseSpec: "S" (split-K for fc2/proj) or "AS" (A for all kinds + S)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/lightly-train/infer-opt/round5_gemm_sm120
model=$1; spec=$2; shift; shift
for M in "$@"; do
  if [[ $spec == *A* ]]; then
    for kind in fc1 fc2 proj qkv; do
      python tune_mg3.py $kind $model $M A > logs/tunemg3_${kind}_${model}_M${M}_A.log 2>&1
    done
  fi
  for kind in fc2 proj; do
    python tune_mg3.py $kind $model $M S > logs/tunemg3_${kind}_${model}_M${M}_S.log 2>&1
  done
done
echo done > logs/mg3b_${model}_${spec}_done
