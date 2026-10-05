#!/usr/bin/env bash
# copy local scratchpad r10 -> server round10 dir (whole files only)
cd "/c/Users/LiuJun/AppData/Local/Temp/claude/C--Users-LiuJun-Desktop-ssh-autodl/f17624be-463a-4646-8ebf-cbdefb894c6d/scratchpad/r10"
scp -q -r impl infer-5090:/root/lightly-train/infer-opt/round10_lowp_probe/ || exit 1
scp -q tests/*.py infer-5090:/root/lightly-train/infer-opt/round10_lowp_probe/tests/; for f in *.py *.sh *.cu; do [ -f "$f" ] && scp -q "$f" infer-5090:/root/lightly-train/infer-opt/round10_lowp_probe/; done
echo deployed
