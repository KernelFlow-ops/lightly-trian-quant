#!/bin/bash
# round17 queue 8: qkv per-GEMM-kind bisection (which of fwd / dgrad / wgrad of qkv costs gradient cosine), ViT-L, 10 new batches.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
echo "blockA qkv kinds $(date +%T)"
python tests/test_block_mx8.py --models vitl16 --variants qkv_f qkv_d qkv_w qkv_fd qkv_dw qkv_fw --joint 0 --spiky 1 > $LOG/q8_blockA.log 2>&1
grep -h 'gcos\|PASS\|BELOW' $LOG/q8_blockA.log | cut -c1-400
echo "kinds vitl $(date +%T)"
B="MX8_fc+proj"
python tests/gate_b_multi.py --model vitl16 --cands $B ${B}+qkv_f ${B}+qkv_d ${B}+qkv_w ${B}+qkv_fd ${B}+qkv_dw ${B}+qkv_fw --seeds 11 12 13 14 15 16 17 18 19 20 --out profile/gateB_vitl16_mx8_qkvkinds.json > $LOG/q8_kinds_vitl.log 2>&1
echo QUEUE8_DONE $(date +%T)
