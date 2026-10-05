#!/bin/bash
# weighted step timing with retry (another agent's unlocked process can abort wait_exclusive).
# usage: bench_queue.sh <model>:<stack> [<model>:<stack> ...]   (env R17_WGRAD_SPLITK etc. pass through; NAMESUFFIX appends to --name)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
for spec in "$@"; do
  model=${spec%%:*}; stack=${spec#*:}
  name="${stack}${NAMESUFFIX}"
  safe=${name//[^A-Za-z0-9_]/_}
  for try in 1 2 3 4 5 6; do
    python run.py --model $model --stack "$stack" --action bench --phases all --name "$name" > $LOG/bench_${model}_${safe}.log 2>&1
    if grep -q '"weighted_img_s"' $LOG/bench_${model}_${safe}.log; then
      grep -h '"weighted_img_s"' $LOG/bench_${model}_${safe}.log | head -1 | sed "s/^/$model $name /"
      grep -h '"mid": [0-9.]*,\?' $LOG/bench_${model}_${safe}.log | head -1 | sed "s/^/   mid step_ms_mean /"
      break
    fi
    echo "retry $model $name ($try)"; sleep 20
  done
done
echo BENCH_DONE $(date +%T)
