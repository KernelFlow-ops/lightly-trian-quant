#!/bin/bash
# R21b nsys captures (graph mode, mid phase, 5 steps; round16 inner script, raw .nsys-rep stays in tmp), one process per stack under both GPU locks.
# usage: queue_nsys.sh <tag> <model>:<stack> [...]
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R16=/root/lightly-train/train-opt/round16_full_graph/tests
cd /root/lightly-train/train-opt/round21_fp4_quant_head
OUT=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/nsys
LOG=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs
mkdir -p $OUT $LOG
TAG=$1; shift
for spec in "$@"; do
  model=${spec%%:*}; stack=${spec#*:}
  name=step_${model}_${stack//[^A-Za-z0-9_]/_}_graph_$TAG
  for try in 1 2 3; do
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then break; fi
    rm -f $OUT/${name}*
    $R16/locked_run.sh bash $R16/nsys_inner.sh $OUT/$name run.py --model $model --stack "$stack" --runner graph --action profile --phases mid --warmup 20 > $LOG/nsys_${name}.log 2>&1
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then echo "ok $name try $try"; break; fi
    echo "retry $name ($try)"; sleep 20
  done
  python $R16/trace_busy.py $OUT/$name | tee profile/nsys_busy_${name}.txt
done
echo NSYS_DONE $(date +%T)
