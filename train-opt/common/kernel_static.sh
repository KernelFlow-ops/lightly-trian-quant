#!/usr/bin/env bash
# adapted from infer-opt/common/kernel_static.sh
# Usage: kernel_static.sh cu <file.cu> <out_dir> [total_ctas active_ctas_per_sm]
#        kernel_static.sh bin <cubin|so> <out_dir> [total_ctas active_ctas_per_sm]
set -euo pipefail
if (( $# != 3 && $# != 5 )); then
    echo "usage: $0 cu|bin <source_or_binary> <out_dir> [total_ctas active_ctas_per_sm]" >&2
    exit 2
fi
mode=$1
src=$2
out=$3
base=$(basename "$src")
base=${base%.*}
mkdir -p "$out"
if [[ $mode == cu ]]; then
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -cubin -Xptxas -v -lineinfo \
        -o "$out/$base.cubin" "$src" 2> "$out/$base.ptxas.txt"
    bin="$out/$base.cubin"
elif [[ $mode == bin ]]; then
    bin=$src
else
    echo "mode must be cu or bin" >&2
    exit 2
fi
/usr/local/cuda/bin/cuobjdump -sass "$bin" > "$out/$base.sass"
{
    echo '== ptxas registers / spills / shared memory =='
    if [[ -f "$out/$base.ptxas.txt" ]]; then
        rg -i 'registers|spill|smem|shared memory' "$out/$base.ptxas.txt" || true
    else
        echo 'null (prebuilt binary)'
    fi
    echo '== SASS instruction counts =='
    for op in HMMA QMMA OMMA IMMA LDGSTS UTMALDG UTMASTG LDSM STS LDS LDG STG BAR SYNCS; do
        printf '%-8s %s\n' "$op" "$(rg -c "\\b${op}\\b" "$out/$base.sass" || echo 0)"
    done
    if (( $# == 5 )); then
        awk -v ctas="$4" -v active="$5" 'BEGIN {if (active > 0) printf "waves_per_170_SM=%.3f\n", ctas/(170*active); else print "waves_per_170_SM=null"}'
    else
        echo 'active_ctas_per_sm=null (query cudaOccupancyMaxActiveBlocksPerMultiprocessor for the instantiated kernel)'
        echo 'waves_per_170_SM=null (supply total_ctas and active_ctas_per_sm)'
    fi
} | tee "$out/$base.static.txt"
