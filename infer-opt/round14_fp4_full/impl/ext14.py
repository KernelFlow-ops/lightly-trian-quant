"""Round 14 JIT loaders (torch cpp_extension, own build dir under infer-opt-tmp/round14_fp4_full/build)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round14_fp4_full")
_EXT: dict = {}
for _p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + ":" + os.environ.get("PATH", "")


def load(name, src, flags=()):
    if name not in _EXT:
        from torch.utils.cpp_extension import load as _load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"       # arch-specific only: cvt.e2m1x2 / block-scaled mma need sm_120a, a plain compute_120 pass would fail in ptxas

        src = Path(src)
        tag = hashlib.md5((src.read_bytes() + repr(flags).encode())).hexdigest()[:8]
        build = TMP / "build" / f"{name}_{tag}"
        build.mkdir(parents=True, exist_ok=True)
        _EXT[name] = _load(name=f"r14_{name}_{tag}", sources=[str(src)], build_directory=str(build),
                           extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-std=c++20"] + list(flags),
                           extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _EXT[name]


def lnq_ext():
    return load("lnq14", HERE / "lnq14.cu")


def ln_ext():
    return load("ln14", HERE / "ln14.cu")


def rope_ext():
    return load("rope14", HERE / "rope14.cu")


def attn_bf16_ext():
    """round-11 attn16 (fp16 in, bf16 out), namespace a14 (step 1)."""
    return load("attn14", HERE / "attn14.cu", ("-DCW=8", "-DSTAGES=4", "-DMINB=1"))


def attn_f16_ext():
    """round-12 attn16o (fp16 in, fp16 out), namespace a14o (step 2)."""
    return load("attn14o", HERE / "attn14o.cu", ("-DCW=8", "-DSTAGES=4", "-DMINB=1"))
