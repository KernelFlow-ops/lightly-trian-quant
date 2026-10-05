#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/lightly-train/infer-opt/round5_gemm_sm120
while [ ! -f logs/mg3b_vitb16_AS_done ]; do sleep 20; done
for model in vitl16 vitb16; do
  for M in 2058 2258; do
    for kind in fc1 fc2 proj qkv; do
      python tune_mg3.py $kind $model $M M > logs/tunemg3_${kind}_${model}_M${M}_M.log 2>&1
    done
  done
done
echo done > logs/mg3m_done
