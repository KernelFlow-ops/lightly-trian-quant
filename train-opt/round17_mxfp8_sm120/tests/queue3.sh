#!/bin/bash
# round17 queue 3 (waits for queue 2): nsys (TB1_R13 vs MX8_all, ViT-L first), then weighted step timing.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
until grep -q QUEUE2_DONE $LOG/queue2.log; do sleep 10; done
echo "nsys $(date +%T)"
tests/nsys_queue.sh v2 vitl16:TB1_R13 vitl16:MX8_all vitl16:TB1 vitb16:MX8_all > $LOG/q3_nsys.log 2>&1
echo "bench $(date +%T)"
for spec in vitl16:TB1_R13 vitl16:MX8_all vitl16:MX8_fc vitb16:TB1_R13 vitb16:MX8_all vitb16:MX8_fc; do
  model=${spec%%:*}; stack=${spec#*:}
  python run.py --model $model --stack $stack --action bench --phases all --name $stack > $LOG/q3_bench_${model}_${stack}.log 2>&1
  grep -h '"weighted_img_s"' $LOG/q3_bench_${model}_${stack}.log | head -1 | sed "s/^/$model $stack /"
done
echo QUEUE3_DONE $(date +%T)
