"""GPTQ weight rounding for the block-scaled / int8 formats (Round 10).

Hessian H = sum over calibration tokens of x x^T (non-prefix rows of the 16 calibration frames, P0 activations, see emu.Calib).  The weight W [N, K] (rows = output channels)
is rounded column by column; the error of every column is pushed onto the not-yet-rounded columns through the upper Cholesky factor of H^-1 (Frantar et al.).
Block scales (mx8/mx4: ue8m0 per 32, nv4: e4m3 per 16 with a static per-tensor scale) are fixed when the first column of a block is reached, from the already-updated
weights of that block (so the final tensor is a valid MX/NV tensor); int8 uses one per-row scale fixed from the original row.
With a Hadamard rotation T (block diagonal, orthonormal) on K the Hessian of the rotated activations is T^T H T.  SmoothQuant + GPTQ is not combined (H would need D^-1 H D^-1).
"""
from __future__ import annotations

import torch

import quant_ref as qr


def install(net):
    net.gptq_fn = lambda i, kind, Wm, spec, ml: gptq_dequant(net, i, kind, Wm, spec)


def _hess(net, i, kind, had, K, device):
    H = net.calib.hess[(i, kind)] / max(net.calib.n_hess[(i, kind)], 1)
    if had:
        T = qr.block_hadamard(torch.eye(K, device=device), had)       # rows: x' = x @ T   (X' = X T)
        H = T.t() @ H @ T
    return H


def _scales(Wblk: torch.Tensor, fmt: str, ts: torch.Tensor):
    """per-row dequantisation scale (fp32 [N]) for the block Wblk [N, vec]."""
    amax = Wblk.abs().amax(1)
    if fmt == "mx8":
        return torch.exp2(qr._rceil_byte(amax, qr.F8_MAX, 255).float() - 127.0)
    if fmt == "mx4":
        return torch.exp2(qr._rceil_byte(amax, 6.0, 254).float() - 127.0)
    if fmt == "nv4":
        return qr._nv4_sf(amax, ts).float() * ts
    raise ValueError(fmt)


def _round_col(w: torch.Tensor, sc: torch.Tensor, fmt: str) -> torch.Tensor:
    sc1 = torch.where(sc > 0, sc, torch.ones_like(sc))
    v = w / sc1
    if fmt == "mx8":
        q = v.clamp(-qr.F8_MAX, qr.F8_MAX).to(torch.float8_e4m3fn).float()
    else:
        q = qr.e2m1_value(qr.e2m1_rn(v))
    return torch.where(sc > 0, q * sc1, torch.zeros_like(w))


@torch.no_grad()
def gptq_dequant(net, i: int, kind: str, Wm: torch.Tensor, spec, damp: float = 0.01) -> torch.Tensor:
    fmt = spec.wf
    N, K = Wm.shape
    W = Wm.clone().float()
    H = _hess(net, i, kind, spec.had, K, W.device).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    if fmt == "int8":
        rs = torch.where(W.abs().amax(1) > 0, W.abs().amax(1) / 127.0, torch.ones(N, device=W.device))
        vec = 128
    else:
        vec = qr.VEC[fmt]
        ts = qr._nv4_ts(Wm.float()) if fmt == "nv4" else None
    for j0 in range(0, K, vec):
        j1 = min(j0 + vec, K)
        W1 = W[:, j0:j1].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[j0:j1, j0:j1]
        sc = None if fmt == "int8" else _scales(W1, fmt, ts)
        for j in range(j1 - j0):
            w = W1[:, j]
            if fmt == "int8":
                q = torch.round(w / rs).clamp(-127, 127) * rs
            else:
                q = _round_col(w, sc, fmt)
            Q[:, j0 + j] = q
            err = (w - q) / Hi[j, j]
            W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
            Err[:, j] = err
        if j1 < K:
            W[:, j1:] -= Err @ Hinv[j0:j1, j1:]
    return Q
