#!/bin/bash
# round22 optional arms (run after queue_final.sh): same-session bench extras, extra gate B batches, dY headroom arms.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache
ulimit -n 65536
cd /root/lightly-train/train-opt/round22_fp4_backbone
LOG=/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/logs
B3="all.d=nv4+sr+all.w=nv4+sr"; B4="$B3+nofirst"
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-200 | tr '\n' ' ')"; }
tests/queue_bench.sh vitl16:FINAL:graph vitl16:P3B_B:graph vitl16:FINAL:graph vitb16:FINAL:graph vitb16:P3B_noFirst:graph vitb16:P3B_rnB:graph vitb16:FINAL:graph
step gb_more_vitb python tests/gate_b_real.py --model vitb16 --seeds 8 9 10 11 12 13 14 15 --cands SIM_B3 REAL_B3 REAL_B4 --out profile/gateB_r22_vitb16_seeds8_15.json
step gb_more_vitl python tests/gate_b_real.py --model vitl16 --seeds 8 9 10 11 12 13 14 15 --cands SIM_B4 REAL_B4 --out profile/gateB_r22_vitl16_seeds8_15.json
step gb_hr_vitb python tests/gate_b_real.py --model vitb16 --seeds 0 1 2 3 4 5 6 7 --cands "REAL_B3hr16=real:$B3+dyhr=16" "REAL_B3hr32=real:$B3+dyhr=32" "REAL_B3exact=real:$B3+amax=exact" --out profile/gateB_r22_vitb16_hr.json
step gb_hr_vitl python tests/gate_b_real.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --cands "REAL_B4hr16=real:$B4+dyhr=16" "REAL_B4hr32=real:$B4+dyhr=32" --out profile/gateB_r22_vitl16_hr.json
step gs_vitl_hr16 python run.py --model vitl16 --stack "FINAL,round22_fp4_backbone:$B4+dyhr=16" --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
echo EXTRA_DONE $(date +%T)
