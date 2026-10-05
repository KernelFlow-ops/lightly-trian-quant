#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round16_full_graph
L=/root/autodl-tmp/train-opt-tmp/round16_full_graph/logs
TAIL=14 tests/queue_run.sh tests/list_a.txt
tests/nsys_queue.sh v2 vitl16:P2,r15:graph vitb16:P2,r15:graph
TAIL=14 tests/queue_run.sh tests/list_b.txt
tests/locked_run.sh python tests/test_fallback_and_swap.py vitb16 > $L/fallback_vitb16.log 2>&1
tests/locked_run.sh python tests/probe_r10_capture_failure.py vitb16 P2 > profile/r10_failure_reproduced_vitb16.log 2>&1
tests/locked_run.sh python tests/probe_r10_capture_failure.py vitb16 P2 --fix > profile/r10_failure_fixed_vitb16.log 2>&1
tests/locked_run.sh python ../round15_optimizer_flat/tests/check_real_flat.py vitl16 P2,r15 > ../round15_optimizer_flat/profile/check_real_flat_vitl16.log 2>&1
echo MISC_DONE
TAIL=14 tests/queue_run.sh tests/list_c.txt
echo DRIVER3_DONE
