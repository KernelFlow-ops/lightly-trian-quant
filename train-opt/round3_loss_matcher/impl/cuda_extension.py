"""Lazy loader for the round3 CUDA C++/CUDA kernels.

The extension is intentionally built outside the repository.  This keeps the
source tree reproducible and lets the ordinary CPU unit tests import round3
without requiring nvcc.  Set ``TRAIN_OPT_CUDA_BUILD=0`` to disable loading.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_MODULE: Any | None = None
_ERROR: Exception | None = None


def load_cuda_ops() -> Any:
    global _MODULE, _ERROR
    if _MODULE is not None:
        return _MODULE
    if _ERROR is not None:
        raise RuntimeError("round3 CUDA extension failed to build") from _ERROR
    if os.environ.get("TRAIN_OPT_CUDA_BUILD", "1") == "0":
        raise RuntimeError("round3 CUDA extension disabled by TRAIN_OPT_CUDA_BUILD=0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("round3 CUDA extension requires CUDA")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    build = Path(os.environ.get("TRAIN_OPT_CUDA_BUILD_DIR", "/root/autodl-tmp/train-opt-tmp/C/cuda_build"))
    build.mkdir(parents=True, exist_ok=True)
    try:
        # The target is fixed by RULES: RTX 5090 (compute capability 12.0).
        # MAX_JOBS is capped to avoid starving the other agents during builds.
        old_jobs = os.environ.get("MAX_JOBS")
        os.environ["MAX_JOBS"] = min(int(old_jobs or "6"), 6).__str__()
        old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        _MODULE = load(
            name="train_opt_round3_cuda",
            sources=[str(root / "cuda_ops.cpp"), str(root / "cuda_ops.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            build_directory=str(build),
            verbose=bool(os.environ.get("TRAIN_OPT_CUDA_VERBOSE")),
        )
        return _MODULE
    except Exception as exc:
        _ERROR = exc
        raise
