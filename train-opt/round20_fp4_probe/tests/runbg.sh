#!/bin/bash
# usage: runbg.sh <logname> <cmd...>   (runs under project env in round20 dir, background, logs to tmp/logs)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
log=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs/$1.log; shift
setsid nohup "$@" > $log 2>&1 < /dev/null &
echo started $log
