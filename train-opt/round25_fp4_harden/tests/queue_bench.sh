#!/bin/bash
# round25 whole-step bench (graph, 3 phases 0.2/0.6/0.2, warmup 20, 50 x 3), one process per spec, specs alternate A B C ... A B C ... (same-session ratios).
# usage: queue_bench.sh <model>:<stack> ...    e.g. queue_bench.sh vitl16:FINAL vitl16:P3 vitl16:P3_hr32 vitl16:FINAL vitl16:P3 vitl16:P3_hr32
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
for spec in "$@"; do
  model=${spec%%:*}; stack=${spec#*:}
  name=bench_${model}_${stack}_$(date +%H%M%S)
  echo "== $name $(date +%T)"
  tests/retry.sh $LOG/$name.log python run.py --model $model --stack "$stack" --runner graph --action bench --phases all
  tail -3 $LOG/$name.log | cut -c1-260
done
echo BENCH_DONE $(date +%T)
