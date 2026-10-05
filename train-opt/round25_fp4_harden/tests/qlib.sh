#!/bin/bash
# round25 queue helpers (source me).  Every step holds both GPU locks through tests/retry.sh -> round16 locked_run.sh (each step <= ~10 min); a step whose output json already exists is skipped.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
ulimit -n 65536
cd /root/lightly-train/train-opt/round25_fp4_harden
T=/root/autodl-tmp/train-opt-tmp/round25_fp4_harden
LOG=$T/logs
step() { name=$1; out=$2; shift 2; if [ -n "$out" ] && ls $out >/dev/null 2>&1; then echo "== $name skipped (exists)"; return; fi
         echo "== $name $(date +%T)"; tests/retry.sh $LOG/$name.log "$@"; echo "   rc=$? $(tail -n 2 $LOG/$name.log | cut -c1-240 | tr '\n' ' ')"; }
