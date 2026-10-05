#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
python tests/fp4_error_table.py vitl16 --save > $L/err_vitl16_save.log 2>&1
python tests/debug_wgrad.py > $L/debug_wgrad.log 2>&1
echo done > $L/q3_done.flag
