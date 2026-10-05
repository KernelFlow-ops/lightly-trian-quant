#!/usr/bin/env bash
# fp16 tier (real mg_f16acc kernel) on both models.  usage: batchF.sh [tuned-json-name] [tag]
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
export MG16_TUNED=${1:-tuned16.json}
TAG=${2:-}
for M in vitb16 vitl16; do
  python make_cfgs.py F $M > /dev/null
  bash sweep.sh $M F_$M F${TAG}_$M full 250
done
