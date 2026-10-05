#!/usr/bin/env bash
# Static kernel analysis (ncu substitute): registers / spills / smem, SASS instruction mix.
# Usage:
#   common/kernel_static.sh cu  <file.cu> <out_dir>     # compile with ptxas -v, dump SASS
#   common/kernel_static.sh bin <file.cubin|.so> <out_dir>  # dump SASS of an existing binary
set -euo pipefail
MODE=$1; SRC=$2; OUT=$3
CUDA=/usr/local/cuda/bin
mkdir -p "$OUT"
base=$(basename "$SRC"); base=${base%.*}
if [ "$MODE" = cu ]; then
    "$CUDA/nvcc" -O3 -std=c++17 -arch=sm_120a -cubin -Xptxas -v -lineinfo \
        -o "$OUT/$base.cubin" "$SRC" 2> "$OUT/$base.ptxas.txt"
    BIN="$OUT/$base.cubin"
else
    BIN=$SRC
fi
"$CUDA/cuobjdump" -sass "$BIN" > "$OUT/$base.sass" || true
# Instruction-mix summary: tensor-core MMA, TMA/async copies, spills.
{
    echo "== ptxas (regs/smem/spill) =="; [ -f "$OUT/$base.ptxas.txt" ] && grep -E "registers|spill|smem" "$OUT/$base.ptxas.txt" || true
    echo "== SASS instruction counts =="
    for op in HMMA QMMA OMMA IMMA LDGSTS UTMALDG UTMASTG LDSM STS LDS LDG STG BAR SYNCS; do
        printf "%-8s %s\n" "$op" "$(grep -c "\b$op" "$OUT/$base.sass" || true)"
    done
} | tee "$OUT/$base.static.txt"
