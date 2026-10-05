#!/usr/bin/env bash
# usage: sweep.sh <model> <cfgname> <outname> [mode] [time-limit]  (one emu_eval.py sweep, resumable; run under run_locked.sh)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
M=$1; CFG=$2; OUT=$3; MODE=${4:-screen}; TL=${5:-520}
mkdir -p sweeps cfgs
python emu_eval.py sweep --model $M --cfg cfgs/$CFG.json --out sweeps/$OUT.jsonl --mode $MODE --time-limit $TL
