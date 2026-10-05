#!/usr/bin/env bash
# adapted from infer-opt/common/profile_ncu.sh; current host denies counters.
# Usage: profile_ncu.sh <out_prefix> <kernel_regex> <nvtx_range> <python args...>
set -euo pipefail
if (( $# < 4 )); then
    echo "usage: $0 <out_prefix> <kernel_regex> <nvtx_range> <python args...>" >&2
    exit 2
fi
out=$1
kernel=$2
range=$3
shift 3
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
/usr/local/cuda/bin/ncu --set full --import-source yes \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis --section WarpStateStats --section SourceCounters \
    --nvtx --nvtx-include "$range" -k "regex:$kernel" \
    --launch-skip 5 --launch-count 3 -o "$out" \
    /root/miniconda3/bin/python "$@"
/usr/local/cuda/bin/ncu --import "${out}.ncu-rep" --page details --csv > "${out}.details.csv"
