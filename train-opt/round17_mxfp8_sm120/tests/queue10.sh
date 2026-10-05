#!/bin/bash
# round17 queue 10 (waits for queue 9): the recommended policy fc+proj+qkv_dw (qkv forward stays bf16, its dgrad / wgrad are MXFP8):
# gate B on fresh batches, step timing, gate S, nsys.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
until grep -q QUEUE9_DONE $LOG/queue9.log; do sleep 10; done
P="MX8_fc+proj+qkv_dw"
echo "gateB vitl $(date +%T)"
python tests/gate_b_multi.py --model vitl16 --cands MX8_fc $P --seeds 21 22 23 24 25 26 --out profile/gateB_vitl16_mx8_rec.json > $LOG/q10_gateB_vitl.log 2>&1
echo "gateB vitb $(date +%T)"
python tests/gate_b_multi.py --model vitb16 --cands MX8_fc $P --seeds 21 22 23 24 25 26 --out profile/gateB_vitb16_mx8_rec.json > $LOG/q10_gateB_vitb.log 2>&1
echo "bench $(date +%T)"
tests/bench_queue.sh vitl16:$P vitb16:$P > $LOG/q10_bench.log 2>&1
cat $LOG/q10_bench.log
echo "gateS $(date +%T)"
python tests/gate_s_multi.py --model vitl16 --cands $P --tag w2 --phases schedule mid --out profile/gateS_vitl16_mx8_rec.json > $LOG/q10_gateS_vitl.log 2>&1
python tests/gate_s_multi.py --model vitb16 --cands $P --tag w2 --phases schedule mid --out profile/gateS_vitb16_mx8_rec.json > $LOG/q10_gateS_vitb.log 2>&1
echo "nsys $(date +%T)"
tests/nsys_queue.sh v2 vitl16:$P vitb16:$P vitb16:TB1_R13 > $LOG/q10_nsys.log 2>&1
cat $LOG/q10_nsys.log | grep -v '^|' | head -20
echo QUEUE10_DONE $(date +%T)
