#!/usr/bin/env bash
cd /root/lightly-train/infer-opt/round10_lowp_probe
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export LIGHTLY_TRAIN_EVENTS_DISABLED=1
for m in "$@"; do python emu_eval.py refs --model $m; done
