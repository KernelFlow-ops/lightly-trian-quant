#!/bin/bash
# usage: retry.sh <logname> <cmd...> : run a GPU job (round24 dir) under BOTH GPU locks (round16 locked_run.sh); if it aborts because another process holds a CUDA context, wait and retry (<= 10 times)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
log=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs/$1.log; shift
for i in 1 2 3 4 5 6 7 8 9 10; do
  bash /root/lightly-train/train-opt/round16_full_graph/tests/locked_run.sh "$@" >> $log 2>&1 && { echo "[ok]" >> $log; exit 0; }
  if tail -8 $log | grep -q "other compute processes"; then echo "[retry $i] GPU busy, waiting" >> $log; sleep 30; else echo "[failed rc]" >> $log; exit 1; fi
done
