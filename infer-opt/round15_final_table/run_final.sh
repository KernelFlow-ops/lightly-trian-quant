#!/bin/bash
# Round 15 (final table): one same-session whole-model speed comparison, no accuracy gates.
# baseline r5_v1 | FP16 r12 | MXFP8 r13_mx8best | NVFP4 r14_nv4a | MXFP4 r14_mx4a   (ViT-L then ViT-B)
set -u
ROOT=/root/lightly-train/infer-opt
OUT=$ROOT/round15_final_table
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd $ROOT/round12_lowp_stack
PLUG="--plugin $ROOT/round13_mxfp8_full:plug13 --plugin $ROOT/round14_fp4_full:plug14"
python speed12.py --model vitl16 --variants r5_v1,r12+head16,r13_mx8best,r14_nv4a,r14_mx4a $PLUG \
    --tag final_L --out $OUT/results_final.json > $OUT/logs/final_L.log 2>&1
python speed12.py --model vitb16 --variants r5_v1,r12,r13_mx8best,r14_nv4a,r14_mx4a $PLUG \
    --tag final_B --out $OUT/results_final.json > $OUT/logs/final_B.log 2>&1
echo DONE > $OUT/logs/final.done
