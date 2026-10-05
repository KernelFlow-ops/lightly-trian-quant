"""JIT loader for impl/postfuse_ext11.cu (round-11 post_a2); same build cache / tag scheme as tests/test_post.py."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXTS: dict = {}
DEFAULT = dict(NG=8)


def ext(**cfg):
    c = dict(DEFAULT)
    c.update(cfg)
    key = tuple(sorted(c.items()))
    if key not in _EXTS:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        tag = hashlib.md5(repr(sorted(c.items())).encode()).hexdigest()[:8]
        build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_post_" + tag)
        build.mkdir(parents=True, exist_ok=True)
        _EXTS[key] = load(name="r11_post_" + tag, sources=[str(_HERE / "postfuse_ext11.cu")], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a"] + [f"-D{k}={v}" for k, v in c.items()], verbose=False)
    return _EXTS[key]
