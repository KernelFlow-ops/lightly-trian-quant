#!/bin/bash
# round17 queue 4: more seeds (batches) for the policy decision: pass rate of global grad cos >= 0.99 per policy, both models.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
CANDS="MX8_fc MX8_fc+proj MX8_all"
echo "seeds vitl $(date +%T)"
python tests/gate_b_multi.py --model vitl16 --cands $CANDS --seeds 3 4 5 6 7 8 9 10 --out profile/gateB_vitl16_mx8_seeds.json > $LOG/q4_vitl.log 2>&1
echo "seeds vitb $(date +%T)"
python tests/gate_b_multi.py --model vitb16 --cands $CANDS --seeds 3 4 5 6 7 8 9 10 --out profile/gateB_vitb16_mx8_seeds.json > $LOG/q4_vitb.log 2>&1
echo QUEUE4_DONE $(date +%T)
