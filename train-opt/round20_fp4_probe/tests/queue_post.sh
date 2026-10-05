#!/bin/bash
cd /root/lightly-train/train-opt/round20_fp4_probe
L=/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/logs
until [ -f $L/r20b_queue_extra_done.flag ]; do sleep 10; done
bash tests/retry.sh $L/r20b_test_f2.log python tests/test_fp4_block.py --only f --model vitb16
bash tests/retry.sh $L/r20b_test_f2.log python tests/test_fp4_block.py --only f --model vitl16
echo done > $L/r20b_queue_post_done.flag
