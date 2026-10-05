#!/usr/bin/env bash
# usage: queue_run.sh <listfile>   each line = arguments for run.py (dual GPU lock is taken inside run.py); prints a compact tail per job
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round16_full_graph
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "=== $line  $(date +%T)"
  tests/locked_run.sh /root/miniconda3/bin/python run.py $line 2>&1 | grep -v "Seed set\|Tensor Cores\|Automatic Mixed" | tail -${TAIL:-60}
done < "$1"
echo ALLDONE
