#!/bin/bash
# chain D: final same-session pairing of P3B (R22 fp4q) and P3B_q2 (fp4q2), P3B-q2-q2-P3B per model, after chain C
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
LOG=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs
until grep -q CHAIN_C_DONE $LOG/chainC.log; do sleep 20; done
bash tests/queue_bench.sh vitl16:P3B:graph vitl16:P3B_q2:graph vitl16:P3B_q2:graph vitl16:P3B:graph
bash tests/queue_bench.sh vitb16:P3B_B:graph vitb16:P3B_B_q2:graph vitb16:P3B_B_q2:graph vitb16:P3B_B:graph
echo CHAIN_D_DONE $(date +%T)
