#!/usr/bin/env bash
# re-measure quantiser costs and epilogue prices with 16 back-to-back calls per graph (the single-call version includes the graph launch overhead)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
mv results_quant.json results_quant_singlecall.json 2>/dev/null
mv results_gemm_epi.json results_gemm_epi_singlecall.json 2>/dev/null
python quant_matrix.py
python gemm_matrix.py --cands epi_bf16,epi_i32 --Ms 2058,2258 --out results_gemm_epi.json --redo
