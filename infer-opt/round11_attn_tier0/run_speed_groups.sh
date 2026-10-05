#!/usr/bin/env bash
# attn_matrix speed runs in three candidate groups (one process each; flash in every group as the common reference)
cd /root/lightly-train/infer-opt/round11_attn_tier0
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
S2=2x16x1029,2x16x1129,2x12x1029,2x12x1129
S4=4x16x1029,4x16x1129,4x12x1029,4x12x1129
run() { python attn_matrix.py --cands $2 --shapes $3 --out profile/attn_speed_$1.json > logs/attn_speed_$1.log 2>&1; }
run A_b2 flash,tl,a16k0,a16k1,a16k2,a16k3,a16k4 $S2
run B_b2 flash,sage2_fp8,sage2_fp8_f32,sage2_pv16,sage2_pv16_f32 $S2
run C_b2 flash,sage3,spark_fp8 $S2
run A_b4 flash,tl,a16k0,a16k1,a16k2,a16k3,a16k4 $S4
run B_b4 flash,sage2_fp8,sage2_fp8_f32,sage2_pv16,sage2_pv16_f32 $S4
run C_b4 flash,sage3,spark_fp8 $S4
