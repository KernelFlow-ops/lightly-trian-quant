#!/bin/bash
# round25 master queue: waits for the saturation study, then S sweep -> gate B -> benches (alternating A B C .. A B C ..) -> task 3.  Re-runnable (finished steps with an existing output are skipped where the step has one).
T=/root/autodl-tmp/train-opt-tmp/round25_fp4_harden
cd /root/lightly-train/train-opt/round25_fp4_harden
while ! grep -q SAT_DONE $T/logs/queue_sat.log 2>/dev/null; do sleep 10; done
bash tests/queue_s.sh
bash tests/queue_b.sh
bash tests/queue_bench.sh vitl16:FINAL vitl16:P3 vitl16:P3_hr16 vitl16:P3_hr32 vitl16:P3_ex vitl16:P3_H2 vitl16:FINAL vitl16:P3 vitl16:P3_hr16 vitl16:P3_hr32 vitl16:P3_ex vitl16:P3_H2
bash tests/queue_bench.sh vitb16:FINAL vitb16:P3 vitb16:P3_H2 vitb16:P3_H2f vitb16:P3_ex vitb16:FINAL vitb16:P3 vitb16:P3_H2 vitb16:P3_H2f vitb16:P3_ex
bash tests/queue_t3.sh
echo MASTER_DONE $(date +%T)
