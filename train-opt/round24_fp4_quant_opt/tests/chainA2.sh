#!/bin/bash
# chain A2: Philox-7 step benches interleaved with the default (10 rounds) fp4q2, after chain A
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
LOG=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs
until grep -q CHAIN_A_DONE $LOG/chainA.log; do sleep 15; done
bash tests/queue_bench.sh vitl16:P3B_q2:graph vitl16:P3B_q2p7:graph vitl16:P3B_q2p7:graph vitl16:P3B_q2:graph
bash tests/queue_bench.sh vitb16:P3B_B_q2:graph vitb16:P3B_B_q2p7:graph vitb16:P3B_B_q2p7:graph vitb16:P3B_B_q2:graph
echo CHAIN_A2_DONE $(date +%T)
