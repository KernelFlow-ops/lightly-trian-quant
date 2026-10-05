#!/usr/bin/env bash
# adapted from infer-opt/common/profile_nsys.sh
# Usage: profile_nsys.sh <out_prefix> <python entry and args...>
set -euo pipefail
if (( $# < 2 )); then
    echo "usage: $0 <out_prefix> <python entry and args...>" >&2
    exit 2
fi
out=$1
shift
mkdir -p "$(dirname "$out")" /root/autodl-tmp/infer-opt-tmp /root/autodl-tmp/train-opt-tmp
if compgen -G "${out}*" > /dev/null; then
    echo "refusing to overwrite existing profile: $out" >&2
    exit 1
fi
exec 9>>/root/autodl-tmp/infer-opt-tmp/gpu.lock
flock 9
exec 8>>/root/autodl-tmp/train-opt-tmp/gpu.lock
flock 8
export TRAIN_OPT_GPU_LOCK_HELD=1
PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src \
    /root/miniconda3/bin/python -c 'from common.bench import assert_gpu_exclusive; assert_gpu_exclusive()'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "${out}_compute_apps.csv"
/usr/local/bin/nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --python-backtrace=cuda --cudabacktrace=sync \
    -o "$out" /root/miniconda3/bin/python "$@"
/usr/local/bin/nsys export --type sqlite --output "${out}.sqlite" "${out}.nsys-rep"
/usr/local/bin/nsys stats \
    -r cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace \
    --format csv -o "$out" "${out}.nsys-rep"
