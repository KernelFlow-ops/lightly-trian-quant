#!/bin/bash
# usage: bg_raw.sh <logname> <cmd...> : run any command in the background (setsid nohup), log to tmp/logs/<logname>.log
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round21_fp4_quant_head
log=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs/$1.log; shift
setsid nohup "$@" > $log 2>&1 < /dev/null &
echo started $log
