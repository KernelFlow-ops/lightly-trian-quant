#!/usr/bin/env bash
# Launch one long training job with both GPU locks; never replace an existing log.
# Usage: gpu_run.sh <log_path> <command...>
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    log=$1
    shift
    mkdir -p /root/autodl-tmp/infer-opt-tmp /root/autodl-tmp/train-opt-tmp
    exec 9>>/root/autodl-tmp/infer-opt-tmp/gpu.lock
    flock 9
    exec 8>>/root/autodl-tmp/train-opt-tmp/gpu.lock
    flock 8
    export TRAIN_OPT_GPU_LOCK_HELD=1
    printf 'START %s\n' "$(date -u +%FT%TZ)"
    if ! PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src \
        /root/miniconda3/bin/python -c 'from common.bench import assert_gpu_exclusive; assert_gpu_exclusive()'; then
        printf 'EXIT_CODE=1 END=%s\n' "$(date -u +%FT%TZ)"
        exit 1
    fi
    status=0
    "$@" || status=$?
    printf 'EXIT_CODE=%s END=%s\n' "$status" "$(date -u +%FT%TZ)"
    exit "$status"
fi
if (( $# < 2 )); then
    echo "usage: $0 <log_path> <command...>" >&2
    exit 2
fi
log=$1
shift
available_kb=$(df -Pk /root/autodl-tmp | awk 'END {print $4}')
if (( available_kb < 15 * 1024 * 1024 )); then
    echo 'at least 15 GB free space is required before a long GPU run' >&2
    exit 1
fi
mkdir -p "$(dirname "$log")"
if [[ -e $log ]]; then
    echo "refusing to overwrite existing log: $log" >&2
    exit 1
fi
: > "$log"
nohup bash "$0" --worker "$log" "$@" >> "$log" 2>&1 < /dev/null &
printf 'pid=%s log=%s\n' "$!" "$log"
