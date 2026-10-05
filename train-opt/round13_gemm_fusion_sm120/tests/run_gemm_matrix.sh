#!/bin/bash
# Final same-machine GEMM candidate matrix (4 Linears x fwd/dgrad/wgrad, plain + joint M, both models).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
for model in vitl16 vitb16; do
  for m in plain joint; do
    python tests/bench_gemm_matrix.py --model $model --m $m --out profile/gemm_matrix_${model}_${m}.json > $LOG/gemm_matrix_${model}_${m}.log 2>&1
    echo "done $model $m $(date +%T)"
  done
done
echo ALL_DONE
