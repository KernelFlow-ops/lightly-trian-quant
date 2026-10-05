#!/bin/bash
# round19 queue 1: 3-phase benches (0.2/0.6/0.2, warmup 20, 50x3). Same harness for all: round16 action_bench.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
for spec in "$@"; do
  model=${spec%%:*}; rest=${spec#*:}; runner=${rest##*:}; stack=${rest%:*}
  name=bench_${model}_${stack}_${runner}
  echo "== $name $(date +%T)"
  python run.py --model $model --stack $stack --runner $runner --action bench --phases all > $LOG/$name.log 2>&1
  tail -3 $LOG/$name.log | cut -c1-200
done
echo QUEUE_DONE $(date +%T)
