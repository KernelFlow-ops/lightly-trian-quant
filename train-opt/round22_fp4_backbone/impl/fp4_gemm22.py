"""Round22 FP4 GEMM layer on ``round21 fp4q.FP4Q`` operands (nv4 only): thin wrappers over round20's QuACK dispatch (``fp4_gemm``).

All GEMMs are TN with K-major 4-bit operands (CONTRACTS 11.2); the per-tensor scale ``ts_a * ts_b`` is folded in as a device-scalar alpha
(``BlockScaledOperand.per_tensor_scale``, no host sync):

    fwd    (not used by R22: forward stays mx8 / bf16)
    dgrad  dX [M,K] bf16 = dY.q . W.qt^T                  ``G4.dgrad``       (round20, plain QuACK nv4)
    dgrad  fc2: dz = (dY.q . W.qt^T) * GELU'(z), db        ``G4.dgrad_dgelu`` (round20 ``fp4_epi.dgelu_ts`` epilogue)
    wgrad  dW [N,K] fp32 = dY.qt . X.qt^T                 ``wgrad`` below: contraction over the zero padded token axis; small outputs
                                                         (proj: N*K <= 1.2e6) use QuACK's autotuned split-K (R20a matrix: 46 -> 21-25 us)

``FP4Q`` (round21) has the attributes round20's wrappers duck-type on (fmt, rows, cols, q, sf, ts, qt, sft, tst).
"""
from __future__ import annotations

import os

import torch

from round20_fp4_probe.impl import fp4_cands as C
from round20_fp4_probe.impl import fp4_gemm as G4

SPLITK_MAX_OUT = int(os.environ.get("R22_SPLITK_MAX_OUT", 1_200_000))

dgrad = G4.dgrad
dgrad_dgelu = G4.dgrad_dgelu


def _gemm():
    from quack.gemm_interface import gemm

    return gemm


def wgrad(dy, x, split_k=None):
    """``dW [N, K]`` fp32 = dY^T X over the padded token axis.  ``dy.qt`` [N, Kt/2], ``x.qt`` [K, Kt/2]; both copies must have the same Kt (padding rows are 0).

    ``split_k``: None = by output size (SPLITK_MAX_OUT), True / False to force."""
    assert dy.kt == x.kt and dy.kt > 0, (dy.kt, x.kt)
    N, K = dy.cols, x.cols
    dw = torch.empty(N, K, device=dy.qt.device, dtype=torch.float32)
    ao = C.operand((dy.qt, dy.sft, dy.tst), "nv4")
    bo = C.operand((x.qt, x.sft, x.tst), "nv4")
    use = (N * K <= SPLITK_MAX_OUT) if split_k is None else bool(split_k)
    if use:
        _gemm()(ao, bo.mT, out=dw, split_k=None)
    else:
        _gemm()(ao, bo.mT, out=dw)
    return dw
