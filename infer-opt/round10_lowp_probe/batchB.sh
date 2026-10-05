#!/usr/bin/env bash
# lowp GEMM candidates part 1 (backbone shapes): reference + cuBLASLt MXFP8 + Spark fp8 MX + int8 + epilogue prices.   usage: batchB.sh [time-limit]
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python gemm_matrix.py --cands r5_v1,cublas_bf16,cublas_fp16acc,cublaslt_mx8,spark_mx8_v1,spark_mx8_v2,int8_mm,epi_bf16,epi_i32 --Ms 2058,2258 --out results_gemm_lowp1.json --time-limit ${1:-480}
