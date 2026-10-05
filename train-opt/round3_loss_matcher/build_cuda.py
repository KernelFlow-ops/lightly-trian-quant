"""Build the round3 CUDA C++ extension into the C temporary directory."""
from __future__ import annotations

import os
from pathlib import Path

from impl.cuda_extension import load_cuda_ops


if __name__ == "__main__":
    os.environ.setdefault(
        "TRAIN_OPT_CUDA_BUILD_DIR",
        "/root/autodl-tmp/train-opt-tmp/C/cuda_build",
    )
    module = load_cuda_ops()
    print(module.__file__)
