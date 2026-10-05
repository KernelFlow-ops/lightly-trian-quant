#!/usr/bin/env bash
# usage: batchFin.sh <model> <preset> <tag> <min-iou> <max-conf> [time-limit]
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python finalize.py --model $1 --preset $2 --tag "$3" --min-iou $4 --max-conf $5 --time-limit ${6:-400} --min-saving-ms 0.0
