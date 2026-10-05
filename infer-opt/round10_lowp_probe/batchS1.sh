#!/usr/bin/env bash
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python make_cfgs.py A
bash sweep.sh vitb16 A A_vitb16 screen 280
bash sweep.sh vitl16 A A_vitl16 screen 280
