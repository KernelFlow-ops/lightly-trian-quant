#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
for job in "vitl16 bb:7203" "vitb16 head" "vitb16 bb:14406" "vitl16 bb:7903" "vitb16 bb:15806"; do
  set -- $job
  python tests/bench_fp4_gemm.py --model $1 --what $2 > $L/matrix_$1_${2/:/}.log 2>&1
done
echo done > $L/matrix_done.flag
