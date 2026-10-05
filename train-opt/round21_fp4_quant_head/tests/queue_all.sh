#!/bin/bash
# gate A (full) then the full micro-benchmark, with retry (another agent's unlocked process can trip assert_gpu_exclusive); logs in tmp/logs
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1
cd /root/lightly-train/train-opt/round21_fp4_quant_head
L=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs
for step in "gateA:python tests/test_fp4q.py --part all" "bench:python tests/bench_fp4q.py"; do
  name=${step%%:*}; cmd=${step#*:}
  for try in 1 2 3 4 5 6; do
    $cmd > $L/q_$name.log 2>&1
    if grep -q '^DONE' $L/q_$name.log; then break; fi
    echo retry $name $try >> $L/q_retry.log; sleep 20
  done
done
echo ALLDONE > $L/q_done.flag
