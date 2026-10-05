#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
MG16_TUNED=tuned16_pkt1.json python robust_plans.py --model vitl16 --set fp16_all4,fp16_qkv,fp16_proj,fp16_fc1,fp16_fc2 --seeds 0 --tag _fp16_pkt1
MG16_TUNED=tuned16_pkt1.json python robust_plans.py --model vitb16 --set fp16_all4 --seeds 0 --tag _fp16_pkt1
python robust_plans.py --model vitl16 --set fp16_qkv,fp16_proj,fp16_fc1,fp16_fc2 --seeds 0 --tag _fp16_tuned
