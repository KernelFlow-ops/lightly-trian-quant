#!/usr/bin/env bash
# usage: batchS2.sh <model> <G-stage> [time-limit]   -- resumable full-mode sweep of one config grid
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python make_cfgs.py $2 > /dev/null
bash sweep.sh $1 $2 $2_$1 full ${3:-540}
