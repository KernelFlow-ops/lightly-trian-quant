#!/usr/bin/env bash
# lowp GEMM candidates part 2: QuACK mx8 (with epilogue), FP4 candidates.   usage: batchB2.sh <cands> <out> [time-limit]
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python gemm_matrix.py --cands $1 --Ms 2058,2258 --out $2 --time-limit ${3:-480}
