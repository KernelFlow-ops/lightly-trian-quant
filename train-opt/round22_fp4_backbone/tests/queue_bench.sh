#!/bin/bash
# round22: 3-phase graph benches (0.2/0.6/0.2, warmup 20, 50x3), same harness as round19 (round16 action_bench).  Interleaved A B A B per model for same-session ratios.
# usage: queue_bench.sh <model>:<stack>:<runner> ...      (stack names: FINAL, P3B, P3B_B, P3B_noFirst, ...)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
i=0
for spec in "$@"; do
  i=$((i+1))
  model=${spec%%:*}; rest=${spec#*:}; runner=${rest##*:}; stack=${rest%:*}
  name=bench_${model}_${stack}_${runner}_$(date +%H%M%S)
  echo "== $name $(date +%T)"
  tests/retry.sh $LOG/$name.log python run.py --model $model --stack "$stack" --runner $runner --action bench --phases all
  tail -3 $LOG/$name.log | cut -c1-200
done
echo QUEUE_DONE $(date +%T)
