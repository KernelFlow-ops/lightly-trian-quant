#!/usr/bin/env bash
# Usage: common/profile_nsys.sh <out_prefix> <python args...>
# The python entry must call bench.profile_fn (cudaProfilerStart/Stop capture range).
# Writes <out_prefix>.nsys-rep and CSV stats next to it; holds the GPU lock for the
# whole run and tells the python child (via INFER_OPT_GPU_LOCK_HELD) not to re-take it.
set -euo pipefail
OUT=$1; shift
mkdir -p "$(dirname "$OUT")"
PY=/root/miniconda3/bin/python
exec 9>/root/autodl-tmp/infer-opt-tmp/gpu.lock
flock 9
export INFER_OPT_GPU_LOCK_HELD=1
/usr/local/bin/nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite true -o "$OUT" "$PY" "$@"
/usr/local/bin/nsys stats --force-export=true \
    -r cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum,cuda_gpu_mem_time_sum \
    --format csv -o "$OUT" "$OUT.nsys-rep" > /dev/null
ls -1 "$OUT"*
