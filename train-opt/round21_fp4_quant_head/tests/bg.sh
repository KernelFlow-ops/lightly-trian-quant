#!/bin/bash
# usage: bg.sh <logname> <cmd...> : background retry.sh
R=/root/lightly-train/train-opt/round21_fp4_quant_head/tests
setsid nohup bash $R/retry.sh "$@" > /dev/null 2>&1 < /dev/null &
echo started /root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs/$1.log
