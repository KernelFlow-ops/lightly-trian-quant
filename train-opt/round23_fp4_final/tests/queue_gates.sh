#!/bin/bash
# round23 gates queue (every step takes both GPU locks through tests/retry.sh -> locked_run.sh, each <= ~10 min; a step whose output json already exists is skipped).
# usage: queue_gates.sh <group> [<group> ...]    groups: b, s, ge, exp, e2e, a, nsys
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round23_fp4_final
T=/root/autodl-tmp/train-opt-tmp/round23_fp4_final
LOG=$T/logs
step() { name=$1; out=$2; shift 2; if [ -n "$out" ] && [ -f "$out" ]; then echo "== $name skipped (exists)"; return; fi
         echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-240 | tr '\n' ' ')"; }
for g in "$@"; do
case $g in
b)   # gate B: two independent draws (seeds 0-7, 8-15) per model, same process pairing R0 / P1 / FINAL / P3H / P3B / P3
  step gb_vitb_d1 "" python tests/gate_b_p3.py --model vitb16 --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_p3_vitb16_draw1.json
  step gb_vitl_d1 "" python tests/gate_b_p3.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_p3_vitl16_draw1.json
  step gb_vitb_d2 "" python tests/gate_b_p3.py --model vitb16 --seeds 8 9 10 11 12 13 14 15 --out profile/gateB_p3_vitb16_draw2.json
  step gb_vitl_d2 "" python tests/gate_b_p3.py --model vitl16 --seeds 8 9 10 11 12 13 14 15 --out profile/gateB_p3_vitl16_draw2.json ;;
s)   # gate S in StepGraphRunner: P3 graph vs R0 eager (and vs FINAL graph), 2 scenarios per process
  step gs_vitb_a "" python run.py --model vitb16 --stack P3 --runner graph --action gate_s --ref R0 --scenarios schedule compressed
  step gs_vitb_b "" python run.py --model vitb16 --stack P3 --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
  step gs_vitl_a "" python run.py --model vitl16 --stack P3 --runner graph --action gate_s --ref R0 --scenarios schedule compressed
  step gs_vitl_b "" python run.py --model vitl16 --stack P3 --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late ;;
ge)  # graph_equiv (8 steps, 4 scenarios)
  step ge_vitb profile/graph_equiv_vitb16_P3.json python run.py --model vitb16 --stack P3 --action graph_equiv --equiv-steps 8
  step ge_vitl profile/graph_equiv_vitl16_P3.json python run.py --model vitl16 --stack P3 --action graph_equiv --equiv-steps 8 ;;
exp) # run_final: 20 real graph steps + validation + checkpoint + export, then r19's check_export against the original (R0) path
  for m in vitb16 vitl16; do
    rm -rf $T/ckpt/$m
    step exp_run_$m "" python run_final.py --model $m --stack P3 --steps 20 --val --save-ckpt $T/ckpt/$m --workers 16 --persistent --json profile/run_final_${m}_export.json
    step exp_check_$m "" python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model $m --ckpt-dir $T/ckpt/$m --out profile/export_$m.json
  done ;;
e2e) # end to end, real loader, 200 steps, workers 16 / prefetch 2 / pin / persistent
  step e2e_vitb "" python run.py --model vitb16 --stack P3 --action e2e --steps 200 --loader-settings 16:2:1:1 --tag P3
  step e2e_vitl "" python run.py --model vitl16 --stack P3 --action e2e --steps 200 --loader-settings 16:2:1:1 --tag P3 ;;
a)   # gate A of round21b (head) and round22 (backbone) re-run through ro_run23.py (outputs -> profile/gateA_<round>_*.json)
  step ga_head "" python tests/ro_run23.py /root/lightly-train/train-opt/round21_fp4_quant_head/tests/test_head_fp4.py --part all
  step ga_blk_vitb "" python tests/ro_run23.py /root/lightly-train/train-opt/round22_fp4_backbone/tests/test_block22.py --model vitb16
  step ga_blk_vitl "" python tests/ro_run23.py /root/lightly-train/train-opt/round22_fp4_backbone/tests/test_block22.py --model vitl16 ;;
nsys) # nsys captures (graph mode, mid phase, 5 steps): FINAL and P3, both models
  R16=/root/lightly-train/train-opt/round16_full_graph/tests
  OUT=$T/nsys
  for spec in vitl16:FINAL vitl16:P3 vitb16:FINAL vitb16:P3; do
    model=${spec%%:*}; stack=${spec#*:}; name=step_${model}_${stack}_graph_v1
    for try in 1 2 3; do
      if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then break; fi
      rm -f $OUT/${name}*
      $R16/locked_run.sh bash $R16/nsys_inner.sh $OUT/$name run.py --model $model --stack "$stack" --runner graph --action profile --phases mid --warmup 20 > $LOG/nsys_${name}.log 2>&1
      if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then echo "ok $name try $try"; break; fi
      echo "retry $name ($try)"; sleep 10
    done
    python $R16/trace_busy.py $OUT/$name | tee profile/nsys_busy_${name}.txt
  done ;;
esac
done
echo GATES_DONE "$@" $(date +%T)
