#!/bin/bash
# usage: run_nsys_head.sh <variant> <model>   (retries while another compute process holds the GPU)
cd /root/lightly-train/train-opt
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
O=/root/autodl-tmp/train-opt-tmp/round14_head_sm120
v=$1; m=$2
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  rm -f $O/nsys/head_${v}_${m}*
  bash common/profile_nsys.sh $O/nsys/head_${v}_${m} round14_head_sm120/tests/profile_head.py --model $m --variant $v > $O/logs/nsys_head_${v}_${m}.log 2>&1 && break
  sleep 20
done
