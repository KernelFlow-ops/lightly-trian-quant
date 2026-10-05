"""JIT build/load of the round12 CUDA extension (sm_120a).  Build products live in the round's tmp directory."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path("/root/autodl-tmp/train-opt-tmp/round12_attention_sm120/build")
SOURCES = ("attn_ext.cpp", "attn_fwd.cu")  # attn_bwd.cu / attn_misc.cu appended when present

_ext = None


def load_ext(verbose: bool = False, ptxas_info: bool = False, name: str = "r12_attn", defines: tuple = ()):
    global _ext
    if _ext is not None and not ptxas_info and not defines:
        return _ext
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    from torch.utils.cpp_extension import load

    srcs = [HERE / s for s in ("attn_ext.cpp", "attn_fwd.cu", "attn_bwd.cu", "attn_misc.cu") if (HERE / s).exists()]
    BUILD.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20", "-gencode", "arch=compute_120a,code=sm_120a"]
    if ptxas_info:
        flags += ["-Xptxas", "-v"]
    flags += [f"-D{d}" for d in defines]
    mod = load(
        name=name,
        sources=[str(s) for s in srcs],
        extra_include_paths=[str(HERE)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        build_directory=str(BUILD),
        verbose=verbose,
    )
    if not ptxas_info and not defines:
        _ext = mod
    return mod
