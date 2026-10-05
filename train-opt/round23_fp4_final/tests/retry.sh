#!/bin/bash
# usage: retry.sh <log> <cmd...> : run a GPU job under BOTH locks (round16 locked_run.sh); if it aborts because another process holds a CUDA context
# (assert_gpu_exclusive), wait and retry (<= 10 times).  Appends to <log>.  (adapted from round22_fp4_backbone/tests/retry.sh)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round23_fp4_final/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
log=$1; shift
for i in 1 2 3 4 5 6 7 8 9 10; do
  bash /root/lightly-train/train-opt/round16_full_graph/tests/locked_run.sh "$@" >> $log 2>&1 && exit 0
  rc=$?
  if tail -8 $log | grep -q "other compute processes\|GPU busy"; then echo "[retry $i] GPU busy, waiting" >> $log; sleep 30; else exit $rc; fi
done
