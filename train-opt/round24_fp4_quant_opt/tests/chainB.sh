#!/bin/bash
# chain B: correctness gates, started after chain A (timing) is done.  Every step takes both GPU locks through retry_run.sh (<= ~10 min each).
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
LOG=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs
until grep -q CHAIN_A_DONE $LOG/chainA.log; do sleep 20; done
step() { name=$1; shift; echo "== $name $(date +%T)"; tests/retry_run.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-160 | tr '\n' ' ')"; }
step ge_vitl python tests/graph_equiv24.py --model vitl16 --stack P3B_q2 --equiv-steps 8 --out profile/graph_equiv_vitl16_P3B_q2.json
step ge_vitb python tests/graph_equiv24.py --model vitb16 --stack P3B_B_q2 --equiv-steps 8 --out profile/graph_equiv_vitb16_P3B_B_q2.json
for part in fast philox7 rn sr exact weight state gemm headroom codes div; do
  step gateA_$part python tests/test_fp4q2.py --part $part
done
step gateB_vitl_1 python tests/gate_b_q2.py --model vitl16 --cands REAL_B4 Q2_B4 Q2p7_B4 --seeds 0 1 --out profile/gateB_r24_vitl16.json
step gateB_vitb_1 python tests/gate_b_q2.py --model vitb16 --cands REAL_B3 Q2_B3 Q2p7_B3 --seeds 0 1 2 --out profile/gateB_r24_vitb16.json
step gateS_vitb_a python run.py --model vitb16 --stack P3B_B_q2 --runner graph --action gate_s --ref R0 --scenarios schedule compressed
step gateS_vitb_b python run.py --model vitb16 --stack P3B_B_q2 --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
step gateS_vitl_a python run.py --model vitl16 --stack P3B_q2 --runner graph --action gate_s --ref R0 --scenarios schedule compressed
step gateS_vitl_b python run.py --model vitl16 --stack P3B_q2 --runner graph --action gate_s --ref R0 --scenarios fixed_mid fixed_late
step gateB_vitl_2 python tests/gate_b_q2.py --model vitl16 --cands REAL_B4 Q2_B4 Q2p7_B4 --seeds 2 3 --out profile/gateB_r24_vitl16.json
step gateB_vitb_2 python tests/gate_b_q2.py --model vitb16 --cands REAL_B3 Q2_B3 Q2p7_B3 --seeds 3 4 5 --out profile/gateB_r24_vitb16.json
step gateB_vitl_3 python tests/gate_b_q2.py --model vitl16 --cands REAL_B4 Q2_B4 Q2p7_B4 --seeds 4 5 --out profile/gateB_r24_vitl16.json
step gateB_vitb_3 python tests/gate_b_q2.py --model vitb16 --cands REAL_B3 Q2_B3 Q2p7_B3 --seeds 6 7 --out profile/gateB_r24_vitb16.json
step gateB_vitl_4 python tests/gate_b_q2.py --model vitl16 --cands REAL_B4 Q2_B4 Q2p7_B4 --seeds 6 7 --out profile/gateB_r24_vitl16.json
echo CHAIN_B_DONE $(date +%T)
