#!/bin/bash
# round23 extra gate-B draw (seeds 16-23, n -> 24 per model, the same count as round22's three draws) after the main gates, then the optional 5a queue.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
T=/root/autodl-tmp/train-opt-tmp/round23_fp4_final
LOG=$T/logs
while ! grep -q GATES_DONE $LOG/queue_gates1.log; do sleep 20; done
step() { name=$1; out=$2; shift 2; if [ -n "$out" ] && [ -f "$out" ]; then echo "== $name skipped (exists)"; return; fi
         echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-240 | tr '\n' ' ')"; }
step gb_vitb_d3 "" python tests/gate_b_p3.py --model vitb16 --seeds 16 17 18 19 20 21 22 23 --out profile/gateB_p3_vitb16_draw3.json
step gb_vitl_d3 "" python tests/gate_b_p3.py --model vitl16 --seeds 16 17 18 19 20 21 22 23 --out profile/gateB_p3_vitl16_draw3.json
echo B3_DONE $(date +%T)
bash tests/queue_q8.sh
