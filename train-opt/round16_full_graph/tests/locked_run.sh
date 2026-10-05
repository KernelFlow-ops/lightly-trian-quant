#!/usr/bin/env bash
# Run a command under BOTH GPU locks (same protocol as common/bench.gpu_lock / profile_nsys.sh) and tolerate other agents' processes that merely
# WAIT for the lock while already holding an idle CUDA context (common.bench.assert_gpu_exclusive would abort on those; it cannot tell idle from busy).
# After taking the locks: if the GPU is not exclusive, sample utilisation for 3 s; if it is idle (<= 3 %), allow the foreign pids through
# TRAIN_OPT_GPU_ALLOWED_PIDS (honoured by assert_gpu_exclusive in run.py / bench.py) and go ahead.  A busy foreign process is never allowed.
# Usage: locked_run.sh <cmd...>
set -uo pipefail
exec 9>>/root/autodl-tmp/infer-opt-tmp/gpu.lock
flock 9
exec 8>>/root/autodl-tmp/train-opt-tmp/gpu.lock
flock 8
export TRAIN_OPT_GPU_LOCK_HELD=1
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ok=0
for i in $(seq 1 60); do
    if /root/miniconda3/bin/python -c 'from common.bench import assert_gpu_exclusive; assert_gpu_exclusive()' 2>/dev/null; then ok=1; break; fi
    u1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    sleep 1
    u2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    sleep 1
    u3=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ "$u1" -le 3 ] && [ "$u2" -le 3 ] && [ "$u3" -le 3 ]; then
        export TRAIN_OPT_GPU_ALLOWED_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
        echo "locked_run: idle foreign contexts allowed: $TRAIN_OPT_GPU_ALLOWED_PIDS" >&2
        ok=1
        break
    fi
    sleep 2
done
if [ "$ok" != 1 ]; then echo "locked_run: GPU busy (foreign process running) for 3+ min, giving up" >&2; exit 75; fi
exec "$@"
