#!/usr/bin/env bash
B=/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build
echo "== other gpu procs"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader
$B/peak_extra 200
echo ----
$B/bench_peak --iters=5 --ms=200
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader
