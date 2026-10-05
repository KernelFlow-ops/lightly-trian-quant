#!/bin/bash
# round19 nsys captures (graph mode), reusing round16's inner script (NVTX + CUDA graph node trace; raw .nsys-rep stays in tmp)
# usage: nsys_queue.sh <tag> <model>:<stack>:<runner> [...]
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
R16=/root/lightly-train/train-opt/round16_full_graph/tests
cd /root/lightly-train/train-opt/round19_final
OUT=/root/autodl-tmp/train-opt-tmp/round19_final/nsys
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
mkdir -p $OUT $LOG
TAG=$1; shift
for spec in "$@"; do
  model=${spec%%:*}; rest=${spec#*:}; runner=${rest##*:}; stack=${rest%:*}
  name=step_${model}_${stack//[^A-Za-z0-9_]/_}_${runner}_$TAG
  for try in 1 2 3; do
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then break; fi
    rm -f $OUT/${name}*
    $R16/locked_run.sh bash $R16/nsys_inner.sh $OUT/$name run.py --model $model --stack "$stack" --runner $runner --action profile --phases mid --warmup 20 > $LOG/nsys_${name}.log 2>&1
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then echo "ok $name try $try"; break; fi
    echo "retry $name ($try)"; sleep 10
  done
  python $R16/trace_busy.py $OUT/$name | tee profile/nsys_busy_${name}.txt
done
echo NSYS_DONE $(date +%T)
