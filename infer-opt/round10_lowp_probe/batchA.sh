#!/usr/bin/env bash
# emulation sanity + overflow scan (GPU, under lock via run_locked.sh)
source /root/lightly-train/infer-opt/round10_lowp_probe/envsetup.sh
python emu_eval.py check --model vitb16
python scan_overflow.py --model vitb16 --frames 4
python scan_overflow.py --model vitl16 --frames 4
