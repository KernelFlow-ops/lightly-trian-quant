#!/bin/bash
# round19 queue 5 (after queue 3): bf16-only fallback FINAL_noMX under the graph (the best combination that passes the strict gate-B median rule), max batch beyond 14
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
while ! grep -q QUEUE3_DONE $LOG/queue3.log; do sleep 10; done
tests/queue_bench.sh vitb16:FINAL_noMX:graph vitl16:FINAL_noMX:graph > $LOG/queue5_bench.log 2>&1
step() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > $LOG/$name.log 2>&1; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-260 | tr '\n' ' ')"; }
rm -f profile/maxbatch_vitl16_FINAL_graph_B15.json
for B in 15 16 17 18 19 20; do
  step maxbatch_vitl16_B$B python run.py --model vitl16 --stack FINAL --runner graph --action maxbatch --batch $B
  if grep -q '"oom": true' profile/maxbatch_vitl16_FINAL_graph_B$B.json || ! grep -q '"ok": true' profile/maxbatch_vitl16_FINAL_graph_B$B.json; then break; fi
done
echo QUEUE5_DONE $(date +%T)
