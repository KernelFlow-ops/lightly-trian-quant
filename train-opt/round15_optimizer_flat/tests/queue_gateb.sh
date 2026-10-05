#!/usr/bin/env bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round15_optimizer_flat
for model in vitb16 vitl16; do
  echo "=== gate_b $model $(date +%T)"
  python run.py --model $model --stack P1,r15 --ref P1 --action gate_b --seeds 0 1 2 2>&1 | grep -v "Seed set\|Tensor Cores\|Automatic Mixed" | tail -12
done
echo ALLDONE
