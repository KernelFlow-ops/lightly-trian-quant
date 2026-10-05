#!/bin/bash
# round16 nsys captures with retry (profile_nsys.sh refuses to start while another compute process is on the GPU).
# usage: nsys_queue.sh <tag> <model>:<stack>:<runner> [...]      e.g. v1 vitl16:P2,r15:graph vitl16:P2:eager
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round16_full_graph
OUT=/root/autodl-tmp/train-opt-tmp/round16_full_graph/nsys
LOG=/root/autodl-tmp/train-opt-tmp/round16_full_graph/logs
mkdir -p $OUT $LOG
TAG=$1; shift
for spec in "$@"; do
  model=${spec%%:*}; rest=${spec#*:}; runner=${rest##*:}; stack=${rest%:*}
  name=step_${model}_${stack//[^A-Za-z0-9_]/_}_${runner}_$TAG
  for try in $(seq 1 6); do
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then break; fi
    rm -f $OUT/${name}*
    tests/locked_run.sh bash tests/nsys_inner.sh $OUT/$name run.py --model $model --stack "$stack" --runner $runner --action profile --phases mid --warmup 20 > $LOG/nsys_${name}.log 2>&1
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then echo "ok $name try $try"; break; fi
    echo "retry $name ($try)"; sleep 15
  done
  python ../common/nsys_analyze.py $OUT/$name --iters 5 --top 25 > profile/nsys_${name}.txt 2>&1
  python ../round17_mxfp8_sm120/tests/nsys_step_categories.py $OUT/$name --json profile/nsys_cat_${name}.json --top 14 > profile/nsys_cat_${name}.txt 2>&1
  head -14 profile/nsys_cat_${name}.txt | cut -c1-200
done
echo NSYS_DONE $(date +%T)
