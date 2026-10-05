#!/bin/bash
# follow-up bench: headroom A/B + roll monitor cost + full bench_fp4q (tag headroom), retry on assert_gpu_exclusive
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round21_fp4_quant_head
L=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs
rm -f $L/h_done.flag
for step in "hbench:python tests/bench_headroom.py" "fullbench:python tests/bench_fp4q.py --tag headroom"; do
  name=${step%%:*}; cmd=${step#*:}
  for try in 1 2 3 4 5 6; do
    $cmd > $L/h_$name.log 2>&1
    if grep -q '^DONE' $L/h_$name.log; then break; fi
    echo retry $name $try >> $L/h_retry.log; sleep 20
  done
done
echo done > $L/h_done.flag
