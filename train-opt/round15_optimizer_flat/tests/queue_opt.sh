#!/usr/bin/env bash
# optimizer-only benchmark: 4 variants x 2 models (one process each, dual lock inside run.py)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round15_optimizer_flat
for model in vitl16 vitb16; do
  for v in torch_fused r7_capturable r7_multi flat; do
    echo "=== $model $v $(date +%T)"
    python run.py --model $model --stack $v --action opt_bench --iters 40 --repeats 3 2>&1 | grep -v 'Seed set\|Tensor Cores\|Automatic Mixed' | tail -40
  done
done
echo ALLDONE
