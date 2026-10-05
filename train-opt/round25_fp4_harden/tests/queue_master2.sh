#!/bin/bash
T=/root/autodl-tmp/train-opt-tmp/round25_fp4_harden
cd /root/lightly-train/train-opt/round25_fp4_harden
while ! grep -q MASTER_DONE $T/logs/queue_master.log 2>/dev/null; do sleep 10; done
bash tests/queue_real.sh
echo MASTER2_DONE $(date +%T)
