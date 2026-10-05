#!/bin/bash
# round17 queue 2: re-validate the host-optimized wrappers (gate A quick), then layer-policy bisection with gate B-lite on both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
echo "gemmA quick $(date +%T)"
python tests/test_gemm_mx8.py --quick > $LOG/q2_gemmA.log 2>&1
tail -2 $LOG/q2_gemmA.log
CANDS="MX8_fc MX8_fc+proj MX8_fc+qkv MX8_attn MX8_all+nofirst MX8_all+nojoint MX8_all+nofirst+nolast+nojoint"
echo "bisect vitb $(date +%T)"
python tests/gate_b_multi.py --model vitb16 --cands $CANDS --seeds 0 1 2 --out profile/gateB_vitb16_mx8_bisect.json > $LOG/q2_bisect_vitb.log 2>&1
echo "bisect vitl $(date +%T)"
python tests/gate_b_multi.py --model vitl16 --cands $CANDS --seeds 0 1 2 --out profile/gateB_vitl16_mx8_bisect.json > $LOG/q2_bisect_vitl.log 2>&1
echo QUEUE2_DONE $(date +%T)
