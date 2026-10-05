"""Round 10: price of the epilogue that GEMMs with a plain bf16 / int32 output still need (separate bandwidth-bound pass; see epi_kernels.cu)."""
from __future__ import annotations

import torch

from bench10 import MODELS, Cand, TMP
import cands_base as cb

BF = torch.bfloat16
_E = None


def ext():
    global _E
    if _E is None:
        from torch.utils.cpp_extension import load
        bd = TMP / "build" / "epi"
        bd.mkdir(parents=True, exist_ok=True)
        _E = load(name="r10_epi", sources=[str(__import__("pathlib").Path(__file__).resolve().parent / "epi_kernels.cu")],
                  extra_cuda_cflags=["-O3", "-arch=sm_120a"], build_directory=str(bd), verbose=False)
    return _E


def make(i32: bool):
    def mk(s):
        e = ext()
        M, N, K, kind, model = s["M"], s["N"], s["K"], s["kind"], s["model"]
        if kind not in ("qkv", "proj", "fc1", "fc2"):
            return None
        B, Nt = cb._batch(M)
        d, H = MODELS[model]["d"], MODELS[model]["H"]
        y = torch.randint(-1000, 1000, (M, N), device="cuda", dtype=torch.int32) if i32 else torch.randn(M, N, device="cuda").to(BF)
        sx = torch.rand(M, device="cuda") + 0.5 if i32 else torch.empty(0, device="cuda")
        sw = torch.rand(N, device="cuda") + 0.5 if i32 else torch.empty(0, device="cuda")
        bias = torch.randn(N, device="cuda") * 0.1
        if kind == "qkv":
            sin = torch.randn(1024, 64, device="cuda"); cos = torch.randn(1024, 64, device="cuda")
            out = torch.empty(3, B, Nt, H, 64, device="cuda", dtype=BF)
            return [lambda: e.epi_rope(y, bias, sx, sw, sin, cos, out, Nt, Nt - 1024, d, H)]
        if kind == "fc1":
            out = torch.empty(M, N, device="cuda", dtype=BF)
            return [lambda: e.epi_gelu(y, bias, sx, sw, out)]
        res = torch.randn(M, N, device="cuda"); gamma = torch.rand(N, device="cuda"); out = torch.empty(M, N, device="cuda")
        return [lambda: e.epi_res(y, res, gamma, bias, sx, sw, out)]
    return mk


def cands(names):
    allc = [Cand("epi_bf16", make(False), "stand-alone epilogue pass over a bf16 GEMM output (qkv: bias+RoPE+layout, fc1: bias+GELU, proj/fc2: bias+gamma+fp32 residual)", repeat=16),
            Cand("epi_i32", make(True), "stand-alone dequant+epilogue pass over an int32 GEMM output (row/col scales + the same epilogues)", repeat=16)]
    return [c for c in allc if names is None or c.name in names]
