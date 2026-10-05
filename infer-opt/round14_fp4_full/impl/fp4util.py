"""Round 14 FP4 helpers shared by the step-1 (r14_*a: QuACK / fp4q2 pieces) and step-2 (own kernel) stacks.

Formats (bit spec = train-opt round20 fp4_ref.py, == infer-opt round10 quant_ref.py):
  nv4  e2m1 + e4m3 block scale per 16 + fp32 per-tensor scale ts = amax/(6*448)
  mx4  e2m1 + ue8m0 block scale per 32 (RCEIL), no per-tensor scale
Operands are K-major packed nibbles q [R, K/2] u8 + blocked scales sf [ceil(R/128), K/(4 vec), 32, 4, 4] u8.

Pieces here
  Site        static per-tensor scale of one activation quantisation site; calibrated device-side from the first (eager) call, never synced to the host
  FQ          a quantised activation (q, sf, ts, M)
  FP4Linear   weight quantised once (1D scales along K, exact ts) + QuACK GEMMs with ts-aware fused epilogues (bias / gamma+fp32 residual / bias+GELU / plain)
Nothing is written under train-opt-tmp: QuACK's disk cache and the fp4q2 build go to infer-opt-tmp/round14_fp4_full.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TMP = Path("/root/autodl-tmp/infer-opt-tmp/round14_fp4_full")
TMP.mkdir(parents=True, exist_ok=True)
os.environ["QUACK_CACHE_DIR"] = str(TMP / "quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
for _p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + ":" + os.environ.get("PATH", "")
for _p in ("/root/lightly-train/train-opt", "/root/lightly-train/src"):
    if _p not in sys.path:
        sys.path.append(_p)

import torch  # noqa: E402

from round20_fp4_probe.impl import fp4_cands as C  # noqa: E402
from round20_fp4_probe.impl import fp4_ref as R  # noqa: E402
from round24_fp4_quant_opt.impl import fp4q2  # noqa: E402

fp4q2._BUILD = TMP / "build" / "fp4_kernels2"          # own build dir (the train-opt one is read-only for us)

VEC = R.VEC
HEADROOM = 2.0


_CLEAR = [lambda: None]      # set by the stack: ln14 ext's clear_cuda_error (see ln14.cu)


class Calib:
    """Shared flag: True during the first eager call of a model_fn (scales are measured), False afterwards (static scales)."""

    def __init__(self):
        self.on = True


class FQ:
    __slots__ = ("q", "sf", "ts", "M", "K")

    def __init__(self, q, sf, ts, M, K):
        self.q, self.sf, self.ts, self.M, self.K = q, sf, ts, M, K


class Site:
    """One activation quantisation site.  nv4: static ts = HEADROOM * amax / 2688 measured on the calibration call (device-side, no host sync)."""

    def __init__(self, fmt, calib: Calib, device="cuda"):
        self.fmt, self.calib = fmt, calib
        self.ts2 = torch.ones(2, dtype=torch.float32, device=device)        # (row, col) copy; only [0] is used here

    def quant(self, x2d: torch.Tensor) -> FQ:
        M, K = x2d.shape
        if self.fmt == "nv4":
            if self.calib.on:
                amax = x2d.float().abs().amax()
                self.ts2.copy_(torch.where(amax > 0, amax * (HEADROOM / 2688.0), torch.ones_like(amax)).expand(2))
            r = fp4q2.quant_act(x2d, "nv4", row=True, col=False, amax="ts", ts=self.ts2)
            return FQ(r.q, r.sf, self.ts2[:1], M, K)
        r = fp4q2.quant_act(x2d, "mx4", row=True, col=False)
        return FQ(r.q, r.sf, None, M, K)


_EPI = None


def epilogues():
    """QuACK block-scaled GEMM epilogues with the per-tensor scale ``ts = ts_a * ts_w`` as a device scalar (mx4 passes 1.0); one output each."""
    global _EPI
    if _EPI is None:
        from quack.activation import gelu_erf
        from quack.epilogue import gemm_epilogue

        @gemm_epilogue()
        def bias_ts(acc, bias, ts):
            return {"D": acc * ts + bias}

        @gemm_epilogue()
        def res_ts(acc, c, bias, gamma, ts):
            return {"D": c + gamma * (acc * ts + bias)}

        @gemm_epilogue()
        def gelu_ts(acc, bias, ts):
            return {"D": gelu_erf(acc * ts + bias)}

        @gemm_epilogue()
        def plain_ts(acc, ts):
            return {"D": acc * ts}

        _EPI = dict(bias=bias_ts, res=res_ts, gelu=gelu_ts, plain=plain_ts)
    return _EPI


def _f4(q):
    return q.view(torch.float4_e2m1fn_x2)


class FP4Linear:
    """y = x @ W^T (W [N, K] quantised once along K, 1D block scales).  All GEMMs are TN, K-major on both sides, bf16 / fp32 outputs."""

    def __init__(self, W: torch.Tensor, bias, fmt: str, calib: Calib, gamma=None):
        self.fmt, self.calib = fmt, calib
        self.N, self.K = W.shape
        self.q, self.sf, ts = R.quantize_matrix(W.float().contiguous(), fmt)
        self.ts = ts if ts is not None else torch.ones(1, device=W.device)
        self.tsp = self.ts.clone()                                           # ts_a * ts_w, updated by the calibration call
        self.qb = _f4(self.q).t()
        self.bias = None if bias is None else bias.float().contiguous().reshape(1, -1)
        self.gamma = None if gamma is None else gamma.float().contiguous().reshape(1, -1)
        self.dev = W.device

    def _kw(self, x: FQ):
        if self.fmt == "nv4" and self.calib.on:
            self.tsp.copy_(x.ts * self.ts)
        return dict(ts=self.tsp, **C.epi_kw(x.sf, self.sf, self.fmt))

    def _done(self):
        if self.calib.on:
            _CLEAR[0]()

    def fwd_bias(self, x: FQ) -> torch.Tensor:
        y = torch.empty(x.M, self.N, device=self.dev, dtype=torch.bfloat16)
        epilogues()["bias"](_f4(x.q), self.qb, out={"D": y}, bias=self.bias, **self._kw(x))
        self._done()
        return y

    def fwd_res(self, x: FQ, resid: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.M, self.N, device=self.dev, dtype=torch.float32)
        epilogues()["res"](_f4(x.q), self.qb, resid, out={"D": out}, bias=self.bias, gamma=self.gamma, **self._kw(x))
        self._done()
        return out

    def fwd_gelu(self, x: FQ) -> torch.Tensor:
        y = torch.empty(x.M, self.N, device=self.dev, dtype=torch.bfloat16)
        epilogues()["gelu"](_f4(x.q), self.qb, out={"D": y}, bias=self.bias, **self._kw(x))
        self._done()
        return y

    def fwd_plain(self, x: FQ) -> torch.Tensor:
        y = torch.empty(x.M, self.N, device=self.dev, dtype=torch.bfloat16)
        epilogues()["plain"](_f4(x.q), self.qb, out={"D": y}, **self._kw(x))
        self._done()
        return y

    def dequant(self) -> torch.Tensor:
        """fp32 [N, K] exactly what the tensor core multiplies (ts applied)."""
        return R.dequant_copy(self.q, self.sf, self.ts if self.fmt == "nv4" else None, self.fmt, self.N, self.K)


def dequant_fq(x: FQ) -> torch.Tensor:
    return R.dequant_copy(x.q, x.sf, x.ts, "nv4" if x.ts is not None else "mx4", x.M, x.K)
