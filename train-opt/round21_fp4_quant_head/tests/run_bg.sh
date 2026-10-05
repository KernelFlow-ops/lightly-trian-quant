#!/bin/bash
# usage: run_bg.sh <logname> <cmd...>  (background, project env, round21 dir)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1
cd /root/lightly-train/train-opt/round21_fp4_quant_head
log=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs/$1.log; shift
setsid nohup "$@" > $log 2>&1 < /dev/null &
echo started $log
