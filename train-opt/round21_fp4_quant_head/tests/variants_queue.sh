#!/bin/bash
# experiment: build kernel variants (R21_VARIANT + R21_DEFS) and bench a representative subset; logs in tmp/logs/var_*
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round21_fp4_quant_head
L=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/logs
rm -f $L/var_*.log $L/var_done.flag
for spec in "mb6:-DACT_MINB=6" "ph7:-DPHILOX_ROUNDS=7" "mb6ph7:-DACT_MINB=6 -DPHILOX_ROUNDS=7"; do
  v=${spec%%:*}; d=${spec#*:}
  for try in 1 2 3 4 5 6; do
    R21_VARIANT=$v R21_DEFS="$d" python tests/bench_fp4q.py --quick --models vitl16,vitb16 > $L/var_$v.$try.log 2>&1
    if grep -q '^DONE' $L/var_$v.$try.log; then break; fi; sleep 15
  done
done
echo done > $L/var_done.flag
