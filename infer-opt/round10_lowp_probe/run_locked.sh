#!/usr/bin/env bash
# Usage: run_locked.sh <logfile> <cmd...>   -- runs <cmd> under both GPU locks (infer-opt then train-opt), as documented in common/lock2.py
LOG=$1; shift
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock flock -x /root/autodl-tmp/train-opt-tmp/gpu.lock \
    env INFER_OPT_GPU_LOCK_HELD=1 TRAIN_OPT_GPU_LOCK_HELD=1 "$@" > "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
