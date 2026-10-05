#!/bin/bash
# waits for queue 5 (and thereby 4), then runs the step benches with the retry wrapper. usage: queue7.sh <model>:<stack> ...
LOG=/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/logs
until grep -q QUEUE5_DONE $LOG/queue5.log; do sleep 10; done
cd /root/lightly-train/train-opt/round17_mxfp8_sm120
exec tests/bench_queue.sh "$@"
