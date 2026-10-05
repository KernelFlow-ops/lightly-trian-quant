#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/lightly-train/infer-opt/round5_gemm_sm120
mkdir -p logs
for spec in "vitl16 2058" "vitl16 2258" "vitb16 2058" "vitb16 2258"; do
  set -- $spec
  for kind in fc1 fc2 proj qkv; do
    python tune5.py $kind $1 $2 > logs/tune5_${kind}_$1_M$2.log 2>&1
  done
done
echo ALLDONE > logs/sweeps_done
