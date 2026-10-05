#!/usr/bin/env bash
# Round 14: copy of round11_attn_tier0/profile11.sh (dual infer-opt + train-opt GPU lock, nsys capture range).
# Usage: prof14.sh <out_prefix> <python args...>
set -euo pipefail
OUT=$1; shift
mkdir -p "$(dirname "$OUT")"
PY=/root/miniconda3/bin/python
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock flock -x /root/autodl-tmp/train-opt-tmp/gpu.lock \
  env INFER_OPT_GPU_LOCK_HELD=1 TRAIN_OPT_GPU_LOCK_HELD=1 \
  /usr/local/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite true -o "$OUT" "$PY" "$@"
/usr/local/bin/nsys stats --force-export=true -r cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" > /dev/null
ls -1 "$OUT"*
