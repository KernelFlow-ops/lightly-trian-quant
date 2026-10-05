#!/bin/bash
# Step-level timing (3 phases, 0.2/0.6/0.2 weighted) for T-BF16 vs T-BF16 + round13, both models.
# Every run.py invocation takes the dual GPU lock itself (common.bench.gpu_lock).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
for model in vitl16 vitb16; do
  for stack in TB1 TB1_R13; do
    echo "=== $model $stack $(date +%T)"
    python run.py --model $model --stack $stack --action bench --phases all --name "${stack}" > $LOG/bench_${model}_${stack}.log 2>&1
    tail -3 $LOG/bench_${model}_${stack}.log
  done
done
for stack in TB1_R13_wgq TB1_R13_torch; do
  echo "=== vitl16 $stack $(date +%T)"
  python run.py --model vitl16 --stack $stack --action bench --phases all --name "${stack}" > $LOG/bench_vitl16_${stack}.log 2>&1
  tail -3 $LOG/bench_vitl16_${stack}.log
done
for model in vitl16 vitb16; do
  for stack in AC "r11:autocast,r13:full"; do
    name=$(echo "$stack" | tr ',:' '__')
    echo "=== $model $stack $(date +%T)"
    python run.py --model $model --stack "$stack" --action bench --phases all --name "$name" > $LOG/bench_${model}_${name}.log 2>&1
    tail -3 $LOG/bench_${model}_${name}.log
  done
done
echo ALL_DONE $(date +%T)
