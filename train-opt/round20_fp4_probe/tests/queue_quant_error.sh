#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
python tests/bench_fp4_quant.py > $L/quant.log 2>&1
python tests/fp4_error_table.py vitl16 > $L/err_vitl16.log 2>&1
python tests/fp4_error_table.py vitb16 > $L/err_vitb16.log 2>&1
echo done > $L/queue2_done.flag
