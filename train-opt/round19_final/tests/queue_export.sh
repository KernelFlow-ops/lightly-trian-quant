#!/bin/bash
# round19: export / checkpoint / val timing + the 200-step proof runs of run_final.py (after the r15 scheduler-state workaround was added to run_final.py)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
CK=/root/autodl-tmp/train-opt-tmp/round19_final/ckpt
step() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > $LOG/$name.log 2>&1; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-260 | tr '\n' ' ')"; }
for m in vitl16 vitb16; do
  rm -rf $CK/$m
  step export2_run_$m python run_final.py --model $m --steps 20 --workers 16 --val --save-ckpt $CK/$m --log-every 10 --warm-steps 5 --json profile/run_final_${m}_export.json
  step export2_check_$m python tests/check_export.py --model $m --ckpt-dir $CK/$m --out profile/export_$m.json
done
for m in vitl16 vitb16; do
  step run200_$m python run_final.py --model $m --steps 200 --workers 16 --persistent --log-every 50 --json profile/run_final_${m}_200.json
done
echo QUEUE_EXPORT_DONE $(date +%T)
