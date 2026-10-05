#!/bin/bash
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
bash tests/retry.sh quick_cmp5 python tests/quick_cmp.py
bash tests/retry.sh hot_v5 python tests/bench_hot.py --cands r21a,base --models vitl16,vitb16 --sites dy,x,dual,row,dysr_row,dysr_col --weights --tag v5
