#!/bin/bash
# round23 optional 5a: qkv forward MXFP8 on top of P3 (policy ``+qkv.f=mx8`` of the round22 block; r17's MXFP8 qkv forward path, no new kernel).  RECORDED ONLY, not published.
# bench (graph, 3 phases) -> gate S (4 scenarios, graph vs R0) -> gate B (n=16: seeds 0-7 and 8-15, same process pairing R0 / P1 / FINAL / P3_q8)
# usage: queue_q8.sh [wait_for_log]   (waits until <wait_for_log> contains GATES_DONE)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
T=/root/autodl-tmp/train-opt-tmp/round23_fp4_final
LOG=$T/logs
if [ -n "$1" ]; then while ! grep -q GATES_DONE "$1"; do sleep 20; done; fi
step() { name=$1; out=$2; shift 2; if [ -n "$out" ] && [ -f "$out" ]; then echo "== $name skipped (exists)"; return; fi
         echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-240 | tr '\n' ' ')"; }
for m in vitl16 vitb16; do
  step q8_bench_${m}_a "" python run.py --model $m --stack P3_q8 --runner graph --action bench --phases all
  step q8_bench_${m}_f "" python run.py --model $m --stack FINAL --runner graph --action bench --phases all
  step q8_bench_${m}_p "" python run.py --model $m --stack P3 --runner graph --action bench --phases all
  step q8_bench_${m}_b "" python run.py --model $m --stack P3_q8 --runner graph --action bench --phases all
done
for m in vitb16 vitl16; do
  step q8_gs_${m}_a "" python run.py --model $m --stack P3_q8 --runner graph --action gate_s --ref R0 --scenarios schedule compressed
  step q8_gs_${m}_b "" python run.py --model $m --stack P3_q8 --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
done
for m in vitb16 vitl16; do
  step q8_gb_${m}_d1 "" python tests/gate_b_p3.py --model $m --cands P3 P3_q8 --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_q8_${m}_draw1.json
  step q8_gb_${m}_d2 "" python tests/gate_b_p3.py --model $m --cands P3 P3_q8 --seeds 8 9 10 11 12 13 14 15 --out profile/gateB_q8_${m}_draw2.json
done
echo Q8_DONE $(date +%T)
