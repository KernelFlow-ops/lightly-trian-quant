"""Round 13: MXFP8 building blocks of the step-1 variant r13_mx8a (cuBLASLt `torch._scaled_mm` GEMMs, E4M3 + UE8M0 1x32).

  ext()                 mx13.cu  (ln_mx8 / quant_rows / gelu_quant / epi_rope / epi_res)
  act_buf(M, K)         persistent zero-initialised MX8 activation buffer (q [M,K] e4m3 + blocked scales); layers share one buffer per shape
                        (the stream is sequential; the padded-row scale bytes stay 0 forever)
  MXW(W [N,K])          weight quantised once (K-major, scale blocks along K), operand B of `_scaled_mm(x, W.qt, scale_a, scale_b)`
  mm(buf, W)            bf16 [M,N] = dequant(buf) @ dequant(W)^T (fp32 accumulate, cuBLASLt block-scaled MXFP8; no epilogue)
Scale layout and scale rule: see mx13.cu header (bit-identical to train-opt r17 quant_dual / round10 quant_ref.mx8_quantize).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full")
F16, BF, F32 = torch.float16, torch.bfloat16, torch.float32
_EXT: dict = {}


def _load(name, src, flags=()):
    if name not in _EXT:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        tag = hashlib.md5((Path(src).read_bytes() + repr(flags).encode())).hexdigest()[:8]
        build = TMP / f"build_{name}_{tag}"
        build.mkdir(parents=True, exist_ok=True)
        _EXT[name] = load(name=f"r13_{name}_{tag}", sources=[str(src)], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a", "--expt-relaxed-constexpr", "-std=c++20"] + list(flags),
                          extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _EXT[name]


def ext():
    first = "mx13" not in _EXT
    e = _load("mx13", _HERE / "mx13.cu")
    if first and os.environ.get("MX13_PDL", "1") == "0":      # MX13_PDL=0 / MGX_PDL=0: plain stream-ordered launches (serialised nsys profiles)
        e.set_pdl(False)
    return e


def pad128(n):
    return -(-n // 128) * 128


class ActBuf:
    __slots__ = ("M", "K", "q", "sf", "qv", "sfv")

    def __init__(self, M, K, device="cuda"):
        self.M, self.K = M, K
        self.q = torch.zeros(M, K, dtype=torch.uint8, device=device)
        self.sf = torch.zeros(pad128(M) // 128 * (K // 128) * 512, dtype=torch.uint8, device=device)
        self.qv = self.q.view(torch.float8_e4m3fn)
        self.sfv = self.sf.view(torch.float8_e8m0fnu)


_POOL: dict = {}


def act_buf(M, K, device="cuda", slot=0):
    key = (M, K, slot, str(device))
    if key not in _POOL:
        _POOL[key] = ActBuf(M, K, device)
    return _POOL[key]


class MXW:
    """[N, K] weight quantised once; N, K multiples of 128 / 256 (every backbone and head weight here is)."""

    def __init__(self, W):
        N, K = W.shape
        self.N, self.K = N, K
        self.q = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
        self.sf = torch.zeros(pad128(N) // 128 * (K // 128) * 512, dtype=torch.uint8, device=W.device)
        ext().quant_rows(W.float().contiguous(), self.q, self.sf)
        self.qt = self.q.view(torch.float8_e4m3fn).t()
        self.sfv = self.sf.view(torch.float8_e8m0fnu)


def mm(buf: ActBuf, W: MXW, out_dtype=BF):
    return torch._scaled_mm(buf.qv, W.qt, scale_a=buf.sfv, scale_b=W.sfv, out_dtype=out_dtype)
