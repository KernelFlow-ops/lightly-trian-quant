#!/bin/bash
# round23: 3-phase graph benches (0.2/0.6/0.2, warmup 20, 50x3), same harness as round19/22 (round16 time_steps via run.py bench).  One process per spec, interleaved A B A B for same-session ratios.
# usage: queue_bench.sh <model>:<stack>:<runner> ...      (stack names: R0 (eager), FINAL, P3, P3H, P3B, P3_RN, ...)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
LOG=/root/autodl-tmp/train-opt-tmp/round23_fp4_final/logs
for spec in "$@"; do
  model=${spec%%:*}; rest=${spec#*:}; runner=${rest##*:}; stack=${rest%:*}
  name=bench_${model}_${stack}_${runner}_$(date +%H%M%S)
  echo "== $name $(date +%T)"
  tests/retry.sh $LOG/$name.log python run.py --model $model --stack "$stack" --runner $runner --action bench --phases all
  tail -3 $LOG/$name.log | cut -c1-260
done
echo QUEUE_DONE $(date +%T)
