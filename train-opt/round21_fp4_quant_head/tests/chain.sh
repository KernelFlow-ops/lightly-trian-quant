#!/bin/bash
# R21b job chain: runs the listed steps one after another (each step = one process under both GPU locks, <= ~10 min per lock hold), logging to tmp/logs; a step is skipped if its marker file exists.
# usage: chain.sh <step> [<step> ...]     steps: gemmbench gb_l_a gb_l_b gb_b_a gb_b_b gs_l_a gs_l_b gs_b_a gs_b_b ge_l ge_b bench_l bench_b nsys exp_l exp_b
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache QUACK_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/lightly-train/train-opt/round21_fp4_quant_head
R16=/root/lightly-train/train-opt/round16_full_graph/tests
T=/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head
LOG=$T/logs
MARK=$T/marks
mkdir -p $MARK $LOG $T/ckpt
cd $R
LR="$R16/locked_run.sh"

run_step() {  # run_step <name> <cmd...>
  name=$1; shift
  if [ -f $MARK/$name ]; then echo "skip $name (done)"; return 0; fi
  for try in 1 2 3; do
    echo "[$(date +%T)] start $name try $try"
    $LR "$@" > $LOG/chain_$name.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch $MARK/$name; echo "[$(date +%T)] done $name"; return 0; fi
    echo "[$(date +%T)] FAIL $name rc=$rc: $(tail -3 $LOG/chain_$name.log | tr '\n' ' ' | cut -c1-300)"
    sleep 20
  done
  return 1
}

for step in "$@"; do
  case $step in
    gemmbench) run_step $step python tests/bench_head_gemm.py ;;
    gb_l_a) run_step $step python tests/gate_b_real.py --model vitl16 --seeds 0 1 2 3 --cands H3 H1 H2 H3x H3rn --out profile/gateB_r21b_vitl16.json ;;
    gb_l_b) run_step $step python tests/gate_b_real.py --model vitl16 --seeds 0 4 5 6 7 --redo 0 --cands H3 H1 H2 H3x H3rn --out profile/gateB_r21b_vitl16.json ;;
    gb_b_a) run_step $step python tests/gate_b_real.py --model vitb16 --seeds 0 1 2 3 --cands H3 H1 H2 H3x H3rn --out profile/gateB_r21b_vitb16.json ;;
    gb_b_b) run_step $step python tests/gate_b_real.py --model vitb16 --seeds 0 4 5 6 7 --redo 0 --cands H3 H1 H2 H3x H3rn --out profile/gateB_r21b_vitb16.json ;;
    gs_l_a) run_step $step python run.py --model vitl16 --stack P3H --action gate_s --runner graph --steps 50 --scenarios schedule fixed_mid ;;
    gs_l_b) run_step $step python run.py --model vitl16 --stack P3H --action gate_s --runner graph --steps 50 --scenarios compressed fixed_late ;;
    gs_b_a) run_step $step python run.py --model vitb16 --stack P3H --action gate_s --runner graph --steps 50 --scenarios schedule fixed_mid ;;
    gs_b_b) run_step $step python run.py --model vitb16 --stack P3H --action gate_s --runner graph --steps 50 --scenarios compressed fixed_late ;;
    ge_l) run_step $step python run.py --model vitl16 --stack P3H --action graph_equiv --scenarios fixed_mid schedule fixed_late --equiv-steps 8 ;;
    ge_b) run_step $step python run.py --model vitb16 --stack P3H --action graph_equiv --scenarios fixed_mid schedule fixed_late --equiv-steps 8 ;;
    bench_l) bash tests/queue_bench.sh s1 vitl16 FINAL P3H FINAL P3H_H1 P3H_H2 FINAL > $LOG/chain_bench_l.log 2>&1 && touch $MARK/bench_l ;;
    bench_b) bash tests/queue_bench.sh s1 vitb16 FINAL P3H FINAL P3H_H1 P3H_H2 > $LOG/chain_bench_b.log 2>&1 && touch $MARK/bench_b ;;
    nsys) bash tests/queue_nsys.sh t1 vitl16:FINAL vitl16:P3H vitb16:FINAL vitb16:P3H > $LOG/chain_nsys.log 2>&1 && touch $MARK/nsys ;;
    exp_l) run_step $step python tests/run_final_r21.py --model vitl16 --stack P3H --steps 20 --workers 16 --val --save-ckpt $T/ckpt/vitl16 --json profile/run_final_vitl16_P3H.json ;;
    exp_b) run_step $step python tests/run_final_r21.py --model vitb16 --stack P3H --steps 20 --workers 16 --val --save-ckpt $T/ckpt/vitb16 --json profile/run_final_vitb16_P3H.json ;;
    chk_l) run_step $step python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model vitl16 --ckpt-dir $T/ckpt/vitl16 --out profile/export_vitl16_P3H.json ;;
    chk_b) run_step $step python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model vitb16 --ckpt-dir $T/ckpt/vitb16 --out profile/export_vitb16_P3H.json ;;
    *) echo "unknown step $step" ;;
  esac
done
echo CHAIN_DONE "$@" $(date +%T)
