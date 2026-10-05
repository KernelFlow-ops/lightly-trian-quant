#!/bin/bash
# usage: retry.sh <logname> <cmd...> : run a GPU job in the round21 dir (background-safe); if it aborts because another process holds a CUDA context (assert_gpu_exclusive), wait and retry (<= 10 times)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/lightly-train/train-opt/round21_fp4_quant_head
log=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs/$1.log; shift
for i in 1 2 3 4 5 6 7 8 9 10; do
  "$@" >> $log 2>&1 && { echo "[ok]" >> $log; exit 0; }
  if tail -8 $log | grep -q "other compute processes"; then echo "[retry $i] GPU busy, waiting" >> $log; sleep 30; else echo "[failed rc]" >> $log; exit 1; fi
done
