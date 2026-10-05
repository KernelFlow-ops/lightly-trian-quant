#!/bin/bash
# round22 resume queue (after the session interruption): export / resume checks first (required), then the same-session bench extras, then the optional gate B / S arms.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
CK=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/ckpt
B3="all.d=nv4+sr+all.w=nv4+sr"; B4="$B3+nofirst"
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
for m in vitb16 vitl16; do
  if [ $m = vitl16 ]; then pol=$B4; else pol=$B3; fi
  rm -rf $CK/$m
  step export_run_$m python /root/lightly-train/train-opt/round19_final/run_final.py --model $m --stack "FINAL,round22_fp4_backbone:$pol" --steps 20 --workers 16 --val --save-ckpt $CK/$m --log-every 10 --warm-steps 5 --json /root/lightly-train/train-opt/round22_fp4_backbone/profile/run_final_${m}_export.json
  step export_check_$m python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model $m --ckpt-dir $CK/$m --out /root/lightly-train/train-opt/round22_fp4_backbone/profile/export_$m.json
done
echo EXPORT_DONE $(date +%T)
