#!/usr/bin/env bash
# patch-embed / head ConvT / mask-einsum GEMM shapes.  usage: batchB4.sh <cands> <out> [time-limit]
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python gemm_matrix.py --extra --cands $1 --out $2 --time-limit ${3:-420}
