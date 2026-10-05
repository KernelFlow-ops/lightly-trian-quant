#!/bin/bash
# priority chain A: same-session step benches (A-B-C-C-B-A per model) and nsys captures
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
bash tests/queue_bench.sh vitl16:FINAL:graph vitl16:P3B:graph vitl16:P3B_q2:graph vitl16:P3B_q2:graph vitl16:P3B:graph vitl16:FINAL:graph
bash tests/queue_bench.sh vitb16:FINAL:graph vitb16:P3B_B:graph vitb16:P3B_B_q2:graph vitb16:P3B_B_q2:graph vitb16:P3B_B:graph vitb16:FINAL:graph
bash tests/nsys_queue.sh q1 vitl16:P3B:graph vitl16:P3B_q2:graph vitb16:P3B_B:graph vitb16:P3B_B_q2:graph
echo CHAIN_A_DONE $(date +%T)
