#!/bin/bash
# usage: retry.sh <log> <cmd...> : run a GPU job; if it aborts because another process holds a CUDA context (assert_gpu_exclusive), wait and retry (<= 8 times)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
log=$1; shift
for i in 1 2 3 4 5 6 7 8; do
  "$@" >> $log 2>&1 && exit 0
  if tail -5 $log | grep -q "other compute processes"; then echo "[retry $i] GPU busy, waiting" >> $log; sleep 30; else exit 1; fi
done
