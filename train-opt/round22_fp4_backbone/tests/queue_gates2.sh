#!/bin/bash
# round22 gates queue 2: reruns after the fixes (graph_equiv ViT-L, S gate ViT-B 'compressed'), headroom B-gate arms, export / resume checks.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
CK=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/ckpt
B3="all.d=nv4+sr+all.w=nv4+sr"; B4="$B3+nofirst"
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
step ge_vitl python tests/graph_equiv22.py --model vitl16 --stack P3B --equiv-steps 8 --out profile/graph_equiv_vitl16_P3B.json
step gs_vitb_c python run.py --model vitb16 --stack P3B_B --runner graph --action gate_s --ref R0 --scenarios compressed
step gb_hr_vitb python tests/gate_b_real.py --model vitb16 --seeds 0 1 2 3 4 5 6 7 --cands "REAL_B3hr16=real:$B3+dyhr=16" "REAL_B3hr32=real:$B3+dyhr=32" "REAL_B3exact=real:$B3+amax=exact" --out profile/gateB_r22_vitb16_hr.json
step gb_hr_vitl python tests/gate_b_real.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --cands "REAL_B4hr16=real:$B4+dyhr=16" "REAL_B4hr32=real:$B4+dyhr=32" --out profile/gateB_r22_vitl16_hr.json
for m in vitl16 vitb16; do
  if [ $m = vitl16 ]; then pol=$B4; else pol=$B3; fi
  rm -rf $CK/$m
  step export_run_$m python /root/lightly-train/train-opt/round19_final/run_final.py --model $m --stack "FINAL,round22_fp4_backbone:$pol" --steps 20 --workers 16 --val --save-ckpt $CK/$m --log-every 10 --warm-steps 5 --json /root/lightly-train/train-opt/round22_fp4_backbone/profile/run_final_${m}_export.json
  step export_check_$m python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model $m --ckpt-dir $CK/$m --out /root/lightly-train/train-opt/round22_fp4_backbone/profile/export_$m.json
done
echo GATES2_DONE $(date +%T)
