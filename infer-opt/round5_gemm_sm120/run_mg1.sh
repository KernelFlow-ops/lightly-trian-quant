#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/lightly-train/infer-opt/round5_gemm_sm120
mkdir -p logs
for kind in fc1 fc2 proj qkv; do
  python tune_mg.py $kind vitl16 2058 A > logs/tunemg_${kind}_vitl16_M2058_A.log 2>&1
done
for kind in fc2 proj; do
  python tune_mg.py $kind vitl16 2058 B > logs/tunemg_${kind}_vitl16_M2058_B.log 2>&1
done
echo done > logs/mg1_done
