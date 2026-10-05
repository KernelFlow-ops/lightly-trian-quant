#!/bin/bash
cd /root/lightly-train/train-opt/round24_fp4_quant_opt
export R24_SRC=/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/scratch/exp_srd.cu
bash tests/retry.sh srd_check python tests/sr_dither_check.py
bash tests/retry.sh hot_srd python tests/bench_hot.py --cands base,srd_exact,srd --models vitl16 --quick --sites dy,dysr_row,dysr_col --tag srd
