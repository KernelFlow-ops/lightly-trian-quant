#!/usr/bin/env bash
# ncu template. NOT runnable on the current AutoDL container (ERR_NVGPUCTRPERM:
# the host driver does not expose GPU performance counters). Run on a machine
# with counter access to back-fill hardware metrics for the kernels found by nsys.
#
# Usage: common/profile_ncu.sh <out_prefix> <kernel_regex> <nvtx_range> <python args...>
#   e.g. common/profile_ncu.sh round3_fusion/profile/ncu_fc1 'gemm|fc1' 'block_05/' \
#        round3_fusion/run.py --profile model --model vitl16
set -euo pipefail
OUT=$1; KERNEL=$2; RANGE=$3; shift 3
exec 9>/root/autodl-tmp/infer-opt-tmp/gpu.lock
flock 9
export INFER_OPT_GPU_LOCK_HELD=1
/usr/local/cuda/bin/ncu --set full --import-source yes \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis --section WarpStateStats --section SourceCounters \
    --nvtx --nvtx-include "$RANGE" -k "regex:$KERNEL" \
    --launch-skip 5 --launch-count 3 \
    -f -o "$OUT" /root/miniconda3/bin/python "$@"
/usr/local/cuda/bin/ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.details.csv"
