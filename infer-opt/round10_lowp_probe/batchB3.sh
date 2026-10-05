#!/usr/bin/env bash
# epilogue prices (bf16 / int32 plain outputs) -- fixed None handling
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python gemm_matrix.py --cands epi_bf16,epi_i32 --Ms 2058,2258 --out results_gemm_epi.json --redo
