#!/bin/bash
# usage: run_gateB.sh <model> <tag> <seeds...> -- <cand names...>      (background, logs to tmp/logs; python takes both GPU locks itself)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
model=$1; tag=$2; shift 2
seeds=(); while [ "$1" != "--" ]; do seeds+=("$1"); shift; done; shift
log=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs/r20b_gateB_${model}_${tag}.log
setsid nohup python tests/gate_b_multi.py --model $model --cands "$@" --seeds "${seeds[@]}" --out profile/gateB_r20b_${model}_${tag}.json > $log 2>&1 < /dev/null &
echo started $log
