#!/bin/bash
# round19: max batch beyond 14 for ViT-L (graph), stops at the first OOM
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
while ! grep -q 'maxbatch_vitl16_B14' $LOG/queue2.log || ! grep -A1 'maxbatch_vitl16_B14' $LOG/queue2.log | grep -q rc=; do sleep 10; done
step() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > $LOG/$name.log 2>&1; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-260 | tr '\n' ' ')"; }
for B in 15 16 17 18 19 20; do
  step maxbatch_vitl16_B$B python run.py --model vitl16 --stack FINAL --runner graph --action maxbatch --batch $B
  if grep -q '"oom": true' profile/maxbatch_vitl16_FINAL_graph_B$B.json || ! grep -q '"ok": true' profile/maxbatch_vitl16_FINAL_graph_B$B.json; then break; fi
done
echo QUEUE4_DONE $(date +%T)
