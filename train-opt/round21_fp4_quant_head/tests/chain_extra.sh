#!/bin/bash
# R21b extra steps (run after chain.sh finished): saturation study with dy_headroom=2 (P3H_HR2), gate S fixed_mid + compressed on both models
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R16=/root/lightly-train/train-opt/round16_full_graph/tests
T=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head
cd /root/lightly-train/train-opt/round21_fp4_quant_head
while ! grep -q CHAIN_DONE $T/logs/chain2.log; do sleep 10; done
for model in vitl16 vitb16; do
  name=gs_hr2_$model
  [ -f $T/marks/$name ] && continue
  $R16/locked_run.sh python run.py --model $model --stack P3H_HR2 --action gate_s --runner graph --steps 50 --scenarios fixed_mid compressed > $T/logs/chain_$name.log 2>&1 && touch $T/marks/$name
done
echo EXTRA_DONE $(date +%T)
