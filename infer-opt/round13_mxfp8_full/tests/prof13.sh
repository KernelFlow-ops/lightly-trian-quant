#!/usr/bin/env bash
# Usage: tests/prof13.sh <model> <variant> <tag> [iters]   -> /root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full/prof_<tag>.*  (dual GPU lock held for the whole run)
set -euo pipefail
MODEL=$1; VAR=$2; TAG=$3; IT=${4:-20}
H=/root/lightly-train/infer-opt/round13_mxfp8_full
OUT=/root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full/prof_$TAG
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock flock -x /root/autodl-tmp/train-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 TRAIN_OPT_GPU_LOCK_HELD=1 \
  /usr/local/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o "$OUT" \
  /root/miniconda3/bin/python $H/tests/prof13.py --model $MODEL --variant $VAR --iters $IT
/usr/local/bin/nsys stats --force-export=true -r cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" > /dev/null
echo done $OUT
