#!/bin/bash
# chain C: optional dither variant: statistics test now, step benches after chain A2
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
LOG=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs
bash tests/retry.sh test_q2d python tests/test_q2d.py
until grep -q CHAIN_A2_DONE $LOG/chainA2.log; do sleep 15; done
bash tests/queue_bench.sh vitl16:P3B_q2:graph vitl16:P3B_q2d:graph vitl16:P3B_q2dp7:graph vitl16:P3B_q2d:graph vitl16:P3B_q2:graph
bash tests/queue_bench.sh vitb16:P3B_B_q2:graph vitb16:P3B_B_q2d:graph vitb16:P3B_B_q2dp7:graph vitb16:P3B_B_q2d:graph vitb16:P3B_B_q2:graph
echo CHAIN_C_DONE $(date +%T)
