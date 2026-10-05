#!/bin/bash
# round25 gate B (relaxed FP4 rule; same process pairing R0 / P1 / FINAL / candidates; each draw = its own process = its own batches)
# ViT-L n=16 (2 draws): P3 and the dY headroom 16 / 32 / 64 / exact variants + the head-H2 candidates;  ViT-B n=24 (3 draws): P3, P3_H2, P3_H2f, P3_hr32
source /root/lightly-train/train-opt/round25_fp4_harden/tests/qlib.sh
for d in 1 2; do
  lo=$(( (d-1)*8 )); seeds=$(seq -s " " $lo $((lo+7)))
  step gb_vitl_d$d "" python tests/gate_b_p25.py --model vitl16 --cands P3 P3_hr16 P3_hr32 P3_hr64 P3_exact P3_H2 P3_H2f --seeds $seeds --out profile/gateB_hr_vitl16_draw$d.json
done
for d in 1 2 3; do
  lo=$(( (d-1)*8 )); seeds=$(seq -s " " $lo $((lo+7)))
  step gb_vitb_d$d "" python tests/gate_b_p25.py --model vitb16 --cands P3 P3_H2 P3_H2f P3_hr32 --seeds $seeds --out profile/gateB_h2_vitb16_draw$d.json
done
echo B_DONE $(date +%T)
