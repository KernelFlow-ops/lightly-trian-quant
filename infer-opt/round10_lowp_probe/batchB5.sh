#!/usr/bin/env bash
# 2-frame batch mode shapes (M = 4116 / 4516): tiles are the nearest tuned ones (not re-tuned)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python gemm_matrix.py --cands r5_v1,cublas_bf16,cublas_fp16acc,mg16_tuned,mg8_tuned,cublaslt_mx8,int8_mm,epi_i32 --Ms 4116,4516 --kinds qkv,proj,fc1,fc2 --out results_gemm_b2.json --time-limit 450
