#!/bin/bash
# round23 control: P3_RN (RN instead of SR in head and backbone) whole-step bench, interleaved with FINAL / P3 of the same session.  waits for Q8_DONE in queue_b3.log.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
T=/root/autodl-tmp/train-opt-tmp/round23_fp4_final
LOG=$T/logs
while ! grep -q Q8_DONE $LOG/queue_b3.log; do sleep 20; done
for m in vitl16 vitb16; do
  for st in FINAL P3_RN P3 P3_RN FINAL; do
    name=rn_bench_${m}_${st}_$(date +%H%M%S)
    echo "== $name $(date +%T)"
    tests/retry.sh $LOG/$name.log python run.py --model $m --stack $st --runner graph --action bench --phases all
    tail -2 $LOG/$name.log | cut -c1-240
  done
done
echo RN_DONE $(date +%T)
