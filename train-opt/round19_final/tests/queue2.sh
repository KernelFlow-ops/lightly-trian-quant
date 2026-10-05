#!/bin/bash
# round19 queue 2: nsys (graph), graph-vs-eager equivalence, gate S (graph vs R0), gate B-lite multi (R0/P1/FINAL, 8 batches), Gate A suite, e2e, export, fallback, max batch.
# waits for queue 1 (benches) to finish; every step is a separate GPU process that takes the dual lock itself.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round19_final
LOG=/root/autodl-tmp/train-opt-tmp/round19_final/logs
CK=/root/autodl-tmp/train-opt-tmp/round19_final/ckpt
T=/root/lightly-train/train-opt
while ! grep -q QUEUE_DONE $LOG/queue_bench.log; do sleep 10; done
step() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > $LOG/$name.log 2>&1; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-260 | tr '\n' ' ')"; }
tests/nsys_queue.sh v1 vitl16:FINAL:graph vitb16:FINAL:graph
step equiv_vitb16 python run.py --model vitb16 --stack FINAL --action graph_equiv --equiv-steps 8 --scenarios fixed_mid schedule fixed_late
step equiv_vitl16 python run.py --model vitl16 --stack FINAL --action graph_equiv --equiv-steps 6 --scenarios fixed_mid schedule fixed_late
step gateS_vitb16 python run.py --model vitb16 --stack FINAL --runner graph --ref R0 --action gate_s --steps 50 --scenarios schedule fixed_mid fixed_late compressed
step gateS_vitl16 python run.py --model vitl16 --stack FINAL --runner graph --ref R0 --action gate_s --steps 50 --scenarios schedule fixed_mid fixed_late compressed
step gateB_vitb16 python tests/gate_b_multi.py --model vitb16 --cands FINAL --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_vitb16_FINAL.json
step gateB_vitl16 python tests/gate_b_multi.py --model vitl16 --cands FINAL --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_vitl16_FINAL.json
# Gate A: the existing round tests that cover the ops of the final stack (outputs redirected into round19_final/profile by tests/ro_run.py)
step A_r17_quant python tests/ro_run.py $T/round17_mxfp8_sm120/tests/test_quant.py
step A_r17_gemm python tests/ro_run.py $T/round17_mxfp8_sm120/tests/test_gemm_mx8.py --quick
step A_r17_block python tests/ro_run.py $T/round17_mxfp8_sm120/tests/test_block_mx8.py --variants fc+proj+qkv_dw --json profile/gateA_r17_block_final.json
step A_r14_head python tests/ro_run.py $T/round14_head_sm120/tests/test_gateA.py --out profile/gateA_r14_head.json
step A_r14_graph_l python tests/ro_run.py $T/round14_head_sm120/tests/test_graph.py --model vitl16
step A_r14_graph_b python tests/ro_run.py $T/round14_head_sm120/tests/test_graph.py --model vitb16
step A_r13_block python tests/ro_run.py $T/round13_gemm_fusion_sm120/tests/test_fused_block.py --json profile/gateA_r13_block.json
step A_r15_flat python tests/ro_run.py $T/round15_optimizer_flat/tests/test_flat_synthetic.py
step A_r11_attn python -m pytest -q -p no:cacheprovider $T/round11_bf16_mixed/tests/test_round11_gpu.py
step A_r3_padded python -m pytest -q -p no:cacheprovider $T/round3_loss_matcher/tests/test_padded_graph.py
LS="16:2:1:0 32:2:1:0 64:2:1:0 32:4:1:0 32:8:1:0 32:2:0:0 32:2:1:1"
step e2e_vitl16 python run.py --model vitl16 --stack FINAL --action e2e --steps 200 --warmup 20 --loader-settings $LS --tag scan
step e2e_vitb16 python run.py --model vitb16 --stack FINAL --action e2e --steps 200 --warmup 20 --loader-settings $LS --tag scan
for m in vitl16 vitb16; do
  rm -rf $CK/$m
  step export_run_$m python run_final.py --model $m --steps 20 --workers 16 --val --save-ckpt $CK/$m --log-every 10 --warm-steps 5 --json profile/run_final_${m}_export.json
  step export_check_$m python tests/check_export.py --model $m --ckpt-dir $CK/$m --out profile/export_$m.json
done
step fallback_vitb16 python tests/test_fallback.py --model vitb16 --out profile/fallback_vitb16.json
step fallback_vitl16 python tests/test_fallback.py --model vitl16 --out profile/fallback_vitl16.json
for B in 9 10 11 12 13 14; do
  step maxbatch_vitl16_B$B python run.py --model vitl16 --stack FINAL --runner graph --action maxbatch --batch $B
  if grep -q '"oom": true' profile/maxbatch_vitl16_FINAL_graph_B$B.json; then break; fi
done
echo QUEUE2_DONE $(date +%T)
