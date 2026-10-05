#!/bin/bash
# R21b timing queue: whole-step bench under the round16 StepGraphRunner (graph replay, 0.2/0.6/0.2 weighted early/mid/late, warmup 20, 50 x 3), one process per stack, all under BOTH GPU locks
# (locked_run.sh: tolerates idle foreign contexts, never a busy one).  The stacks of one model run back to back = one session; FINAL is measured several times to expose the drift.
# usage: queue_bench.sh <tag> <model> <stack> [<stack> ...]      (background: bash tests/bg_raw.sh <log> bash tests/queue_bench.sh ...)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R16=/root/lightly-train/train-opt/round16_full_graph/tests
cd /root/lightly-train/train-opt/round21_fp4_quant_head
LOG=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs
TAG=$1; model=$2; shift 2
i=0
for stack in "$@"; do
  i=$((i+1))
  name=${stack}_${TAG}${i}
  for try in 1 2 3; do
    $R16/locked_run.sh python run.py --model $model --stack $stack --action bench --runner graph --phases all --name "${name}" > $LOG/bench_${model}_${name}.log 2>&1
    if grep -q '^wrote ' $LOG/bench_${model}_${name}.log; then echo "ok ${model} ${name} try $try: $(grep -o '"weighted_img_s": [0-9.]*' $LOG/bench_${model}_${name}.log | head -1)"; break; fi
    echo "retry ${model} ${name} ($try)"; sleep 20
  done
done
echo BENCH_DONE $TAG $model $(date +%T)
