"""JIT loader for the round-11 FP16-accumulate FA2 prototype (impl/attn16.cu)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXTS: dict = {}
DEFAULT = dict(CW=8, STAGES=4, MINB=1)


def ext(**cfg):
    c = dict(DEFAULT)
    c.update(cfg)
    key = tuple(sorted(c.items()))
    if key not in _EXTS:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"     # arch-specific only: one compile per config
        from torch.utils.cpp_extension import load

        tag = hashlib.md5(repr(key).encode()).hexdigest()[:8]
        build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_attn16_" + tag)
        build.mkdir(parents=True, exist_ok=True)
        _EXTS[key] = load(name="r11_attn16_" + tag, sources=[str(_HERE / "attn16.cu")], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-std=c++20", "-lineinfo"]
                          + [f"-D{k}={v}" for k, v in c.items()], extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _EXTS[key]
