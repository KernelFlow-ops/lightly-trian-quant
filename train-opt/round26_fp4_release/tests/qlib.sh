#!/bin/bash
# round26 queue helpers (source me).  Every step holds both GPU locks through tests/retry.sh -> round16 locked_run.sh; a step whose output json already exists is skipped.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65535 2>/dev/null || true      # this container: soft 1024, hard 65535 (65536 fails)
R=/root/lightly-train/train-opt/round26_fp4_release
cd $R
T=/root/autodl-tmp/train-opt-tmp/round26_fp4_release
LOG=$T/logs
mkdir -p $LOG $T/ckpt
step() { name=$1; out=$2; shift 2; if [ -n "$out" ] && ls $out >/dev/null 2>&1; then echo "== $name skipped (exists)"; return; fi
         echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-240 | tr '\n' ' ')"; }
