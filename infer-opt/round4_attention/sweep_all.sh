#!/bin/sh
# Full BLOCK_M/BLOCK_N/threads sweep per (kind, H, N); results in tune/*.json
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd $(dirname $0)
for kind in plain_qkv rope_qkv; do
  for H in 16 12; do
    for N in 1029 1129; do
      python tune_attn.py --kind $kind --H $H --N $N --bm 64,128,256 --bn 32,64 --stages 2 --threads 128,256,512 2>&1 | grep -E 'us  |BEST'
    done
  done
done
python make_tuned.py
