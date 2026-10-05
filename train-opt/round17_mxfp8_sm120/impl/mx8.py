"""MXFP8 tensors for the round17 block: E4M3 data + UE8M0 scales over 1x32 blocks, QuACK / cuBLAS blocked (128x4) scale layout.

``MX8`` is the CONTRACTS ``MX8Tensor``: one logical bf16/fp32 matrix ``x [R, C]`` quantized in up to two directions from a single
read (``mx8_kernels.cu::quant_dual``):

    row copy  : ``qr [R, C]`` e4m3 (K-major for a GEMM that contracts over C), scale blocks along C,
                ``sfr [Rpad/128, C/128, 32, 4, 4]`` u8
    col copy  : ``qc [Rpad, C]`` e4m3 (same row-major layout; rows >= R are zero), scale blocks along R,
                ``sfc [C/128, Rpad/128, 32, 4, 4]`` u8      (the SF of the transposed matrix: blocks run down the columns)

Operands for quack.gemm (``Y[M,N] = A[M,K] @ B[K,N]``, scale blocks always along K):

    fwd    Y  = X Wt   : A = X.a_row()          B = W.b_row_t()        (both K-major)
    dgrad  dX = dY W   : A = dY.a_row()         B = W.b_col()          (W [N,K] seen as (K=N, N=K): N-major B)
    wgrad  dW = dYt X  : A = dY.a_col_t()       B = X.b_col()          (M-major A, N-major B, contraction over padded tokens)

The scale rule is the NVIDIA MXFP8 recipe (E4M3, RCEIL), bit-identical to ``quack.blockscaled.quantize.to_mx(.., 'rceil')``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_BUILD = Path("/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/build/mx8_kernels")
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "8")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _BUILD.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="trainopt_r17_mx8_kernels",
            sources=[str(_HERE / "mx8_kernels.cu")],
            build_directory=str(_BUILD),
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xptxas", "-v"],
            verbose=bool(int(os.environ.get("R17_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def pad128(n: int) -> int:
    return -(-n // 128) * 128


_FMT = None


def fmt():
    global _FMT
    if _FMT is None:
        from quack.blockscaled import MXFP8_E4M3

        _FMT = MXFP8_E4M3
    return _FMT


def fmt_name() -> str:
    return fmt().name


def _sf(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float8_e8m0fnu)


@dataclass
class MX8:
    rows: int
    cols: int
    mpad: int
    qr: Optional[torch.Tensor] = None
    sfr: Optional[torch.Tensor] = None
    qc: Optional[torch.Tensor] = None
    sfc: Optional[torch.Tensor] = None

    # ---- quack operand containers (views only) ----
    def a_row(self):
        """(R, C) operand, scale blocks along C (A of fwd / dgrad)."""
        from quack.blockscaled import BlockScaledOperand

        return BlockScaledOperand.from_parts(self.qr, _sf(self.sfr), fmt(), quant_dim=-1)

    def b_row_t(self):
        """(C, R) operand, scale blocks along C (B of fwd for a weight [N=R, K=C])."""
        return self.a_row().mT

    def b_col(self):
        """(Rpad, C) operand with scale blocks down the rows (B of wgrad; B of dgrad for a weight [N=R, K=C])."""
        from quack.blockscaled import BlockScaledOperand

        return BlockScaledOperand.from_parts(self.qc, _sf(self.sfc), fmt(), quant_dim=-2)

    def a_col_t(self):
        """(C, Rpad) operand, scale blocks along Rpad (A of wgrad, contraction over the tokens)."""
        return self.b_col().mT

    # ---- raw views for the epilogue frontend (no container construction / validation on the hot path) ----
    @property
    def sfr_v(self):
        return _sf(self.sfr)

    @property
    def sfc_v(self):
        return _sf(self.sfc)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qr, self.sfr, self.qc, self.sfc) if t is not None)


def quantize(x: torch.Tensor, *, row: bool = True, col: bool = True, mpad: Optional[int] = None) -> MX8:
    """x [R, C] contiguous bf16/fp32 (C % 128 == 0) -> MX8 with the requested copies (one read of x)."""
    R, C = x.shape
    mp = pad128(R) if mpad is None else mpad
    qr, sfr, qc, sfc = extension().quant_dual(x, mp, row, col)
    return MX8(R, C, mp, qr if row else None, sfr if row else None, qc if col else None, sfc if col else None)
