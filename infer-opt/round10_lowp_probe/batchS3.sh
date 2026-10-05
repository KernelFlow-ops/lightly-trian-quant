#!/usr/bin/env bash
# usage: batchS3.sh <model> <time-limit-per-grid> <grids...>   (each sweep is resumable; total job time = n_grids * limit + startup)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
M=$1; TL=$2; shift; shift
for G in "$@"; do
  python make_cfgs.py $G > /dev/null
  bash sweep.sh $M $G ${G}_$M full $TL
done
