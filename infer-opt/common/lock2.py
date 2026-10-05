"""Dual GPU lock for infer-opt phase 2 (round10+, see PLAN_v2.md section 0 / 4).

Takes the infer-opt lock, then the train-opt lock -- the same order as
train-opt/common/bench.py::gpu_lock -- so infer-opt and train-opt jobs never
share the GPU.  While held, INFER_OPT_GPU_LOCK_HELD=1 and TRAIN_OPT_GPU_LOCK_HELD=1
are set, so nested helpers (infer-opt common/bench.py::gpu_lock used by time_fn /
profile_fn, train-opt gpu_lock) skip re-locking instead of deadlocking on a
second flock of the same file from this process.  A lock already marked held by
a parent (e.g. a shell flock wrapper) is not taken again.

Python:
    from lock2 import gpu_lock2, other_gpu_procs
    with gpu_lock2():
        ...

Shell (whole command under both locks):
    flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock flock -x /root/autodl-tmp/train-opt-tmp/gpu.lock \
        env INFER_OPT_GPU_LOCK_HELD=1 TRAIN_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python script.py
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
from pathlib import Path
from typing import Iterator

INFER_LOCK = Path("/root/autodl-tmp/infer-opt-tmp/gpu.lock")
TRAIN_LOCK = Path("/root/autodl-tmp/train-opt-tmp/gpu.lock")
INFER_ENV = "INFER_OPT_GPU_LOCK_HELD"
TRAIN_ENV = "TRAIN_OPT_GPU_LOCK_HELD"


@contextlib.contextmanager
def gpu_lock2() -> Iterator[None]:
    """Exclusive infer-opt + train-opt GPU lock for timing, nsys and accuracy runs."""
    old = {v: os.environ.get(v) for v in (INFER_ENV, TRAIN_ENV)}
    files = []
    try:
        for path, var in ((INFER_LOCK, INFER_ENV), (TRAIN_LOCK, TRAIN_ENV)):
            if os.environ.get(var) == "1":
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open("a")
            fcntl.flock(f, fcntl.LOCK_EX)
            files.append(f)
            os.environ[var] = "1"
        yield
    finally:
        for var, value in old.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        for f in reversed(files):
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


def other_gpu_procs() -> list[str]:
    """Compute processes on the GPU other than this one (log it next to timings to spot contamination)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return [f"unavailable: {e}"]
    me = str(os.getpid())
    return [line.strip() for line in out.splitlines() if line.strip() and line.split(",")[0].strip() != me]
