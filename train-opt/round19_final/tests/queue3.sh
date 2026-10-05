#!/bin/bash
# round19 queue 3 (runs after queue 2): gate-B bisect (FINAL vs FINAL_fc / FINAL_noR14 / FINAL_noMX on identical batches), cost of the safe r17 policy ("fc") under the graph,
# same-session drift checks of the previous best (P2R15 graph) and the TB1 reference (eager).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
while ! grep -q QUEUE2_DONE $LOG/queue2.log; do sleep 10; done
step() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > $LOG/$name.log 2>&1; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-260 | tr '\n' ' ')"; }
step bisect_vitb16 python tests/gate_b_multi.py --model vitb16 --cands FINAL FINAL_fc FINAL_noR14 FINAL_noMX --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out profile/gateB_vitb16_bisect.json
step bisect_vitl16 python tests/gate_b_multi.py --model vitl16 --cands FINAL FINAL_fc FINAL_noR14 FINAL_noMX --seeds 0 1 2 3 4 5 6 7 8 9 --out profile/gateB_vitl16_bisect.json
tests/queue_bench.sh vitb16:FINAL_fc:graph vitl16:FINAL_fc:graph vitb16:P2R15:graph vitl16:P2R15:graph vitb16:TB1:eager vitl16:TB1:eager > $LOG/queue3_bench.log 2>&1
echo QUEUE3_DONE $(date +%T)
