#!/bin/bash
# queue 2: fused-epilogue candidate matrix, MXFP8 probe (each takes the dual GPU lock inside the python process)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
until grep -q ALL_DONE $LOG/gemm_matrix_driver.log 2>/dev/null; do sleep 10; done
python tests/bench_epilogue.py --out profile/epilogue_matrix.json > $LOG/epilogue_matrix.log 2>&1
echo "epilogue done $(date +%T)"
python tests/mxfp8_smoke.py > $LOG/mxfp8_probe.log 2>&1
echo "mxfp8 done $(date +%T)"
echo QUEUE2_DONE
