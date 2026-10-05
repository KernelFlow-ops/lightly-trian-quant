#!/bin/bash
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
bash tests/retry.sh quick_cmp3 python tests/quick_cmp.py
bash tests/retry.sh hot_v3 python tests/bench_hot.py --cands r21a,base --models vitl16 --quick --sites dy,x,dual,row --tag v3
