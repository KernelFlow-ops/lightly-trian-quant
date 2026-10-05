"""round14 fused EoMT head: ``ops.predict`` override.

predict(x [B,N,C] fp32 residual) =
    xn = backbone.norm(x)                                  torch (fp32 stats, tiny)
    class_logits = class_head(xn[:, :100])                 torch
    E = mask_head(xn[:, :100])                             torch autocast (bf16 [B,100,C])
    mask_logits = Upscale(xn[:, 105:], E)                  ONE autograd.Function (this file), fp32 [B,100,128,128]

Upscale forward (per ScaleBlock stage, ConvT(k2,s2) -> GELU -> dwconv3x3 -> LN2D):
    Z  = tokens @ Wm^T                      GEMM [B*h*w, C] x [C, 4C]  (pixel shuffle is NOT materialized: the Triton kernels read it)
    Y  = LN2D(dwconv3x3(gelu(shuffle(Z) + bias)))      fused Triton kernel (head_triton.fwd)
  logits[b] = E[b] @ Y2[b]^T                batched GEMM, fp32 output
Backward: einsum dgrads (2 bmm) -> per stage: bwd1 (LN backward) -> bwd2 (dwconv backward + GELU') -> dgrad / wgrad GEMMs.
Parameters are passed to the Function directly, so autograd accumulates their fp32 gradients as usual.
"""
from __future__ import annotations

import os

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round14_head_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

import torch

KERNELS = os.environ.get("R14_KERNELS", "cuda")  # "cuda" (head_kernels.cu) | "triton" (head_triton.py)
if KERNELS == "cuda":
    from . import head_cuda as ht
else:
    from . import head_triton as ht

# per-GEMM-kind backend (from tests/bench_gemm.py: QuACK is ~8% faster for fwd/dgrad; cuBLAS wins the fp32-out wgrad; QuACK wins the einsum bwd 2.5x)
POLICY = {"fwd": "quack", "dgrad": "quack", "wgrad": "torch", "e_fwd": "torch", "e_bwd": "quack"}
for _k in list(POLICY):
    POLICY[_k] = os.environ.get(f"R14_{_k.upper()}", POLICY[_k])
POLICY["mx8"] = bool(int(os.environ.get("R14_MX8", "0")))  # ConvT GEMMs (fwd/dgrad/wgrad) in MXFP8 via round17's quantizer + QuACK block-scaled GEMM
DEFAULT_POLICY = dict(POLICY)
_Q = None


def reset_policy() -> None:
    POLICY.clear()
    POLICY.update(DEFAULT_POLICY)


def _quack():
    global _Q
    if _Q is None:
        from quack.gemm_interface import gemm as qgemm

        _Q = qgemm
    return _Q


# ---------------------------------------------------------------------------------------------- GEMM wrappers (row-major bf16)
def mm_fwd(x, wm, backend=None):
    """Z[M,N] = x[M,K] @ wm[N,K]^T (bf16)."""
    backend = backend or POLICY["fwd"]
    out = torch.empty(x.shape[0], wm.shape[0], device=x.device, dtype=torch.bfloat16)
    if backend == "quack":
        _quack()(x, wm.t(), out=out)
    else:
        torch.mm(x, wm.t(), out=out)
    return out


def mm_dgrad(dz, wm, backend=None):
    """dX[M,K] = dZ[M,N] @ wm[N,K] (bf16)."""
    backend = backend or POLICY["dgrad"]
    out = torch.empty(dz.shape[0], wm.shape[1], device=dz.device, dtype=torch.bfloat16)
    if backend == "quack":
        _quack()(dz, wm, out=out)
    else:
        torch.mm(dz, wm, out=out)
    return out


def mm_wgrad(dz, x, backend=None):
    """dWm[N,K] (fp32) = dZ[M,N]^T @ x[M,K]."""
    backend = backend or POLICY["wgrad"]
    if backend == "quack":
        out = torch.empty(dz.shape[1], x.shape[1], device=dz.device, dtype=torch.float32)
        _quack()(dz.t(), x, out=out)
        return out
    return torch.mm(dz.t(), x, out_dtype=torch.float32)


def einsum_fwd(e, y, backend=None):
    """logits[B,Q,HW] fp32 = e[B,Q,C] @ y[B,HW,C]^T."""
    backend = backend or POLICY["e_fwd"]
    if backend == "quack":
        out = torch.empty(e.shape[0], e.shape[1], y.shape[1], device=e.device, dtype=torch.float32)
        _quack()(e, y.transpose(1, 2), out=out)
        return out
    return torch.bmm(e, y.transpose(1, 2), out_dtype=torch.float32)


def einsum_bwd(dl, e, y, backend=None):
    """dl [B,Q,HW] bf16 -> (dy [B,HW,C] bf16 = dl^T @ e,  de [B,Q,C] bf16 = dl @ y)."""
    backend = backend or POLICY["e_bwd"]
    if backend == "quack":
        dy = torch.empty(y.shape, device=y.device, dtype=torch.bfloat16)
        de = torch.empty(e.shape, device=e.device, dtype=torch.bfloat16)
        _quack()(dl.transpose(1, 2), e, out=dy)
        _quack()(dl, y, out=de)
        return dy, de
    return torch.bmm(dl.transpose(1, 2), e), torch.bmm(dl, y)


# ---------------------------------------------------------------------------------------------- weights
def mx_modes() -> frozenset:
    """Which ConvT GEMM kinds run in MXFP8: POLICY['mx8'] = False | True/'all' | '+'-joined subset of fwd, dgrad, wgrad ('bwd' = dgrad+wgrad)."""
    v = POLICY.get("mx8", False)
    if v is True or v == "all":
        return frozenset(("fwd", "dgrad", "wgrad"))
    if not v:
        return frozenset()
    out = set()
    for tok in str(v).replace(",", "+").split("+"):
        out |= {"bwd": {"dgrad", "wgrad"}}.get(tok, {tok})
    return frozenset(out)


class _ConvT:
    """One ConvT(k2s2)-as-GEMM layer: Z = X Wm^T, dX = dZ Wm, dWm = dZ^T X; each kind in bf16 (QuACK/cuBLAS) or MXFP8 (round17)."""

    def __init__(self, x, wm, modes):
        self.x, self.wm, self.modes = x, wm, modes
        self.xq = None
        self.wq = None

    def _mx(self):
        from round17_mxfp8_sm120.impl import gemm_mx8, mx8

        return gemm_mx8, mx8

    def fwd(self):
        if "fwd" not in self.modes:
            return mm_fwd(self.x, self.wm)
        gemm_mx8, mx8 = self._mx()
        self.xq = mx8.quantize(self.x, row=True, col="wgrad" in self.modes)
        self.wq = mx8.quantize(self.wm, row=True, col="dgrad" in self.modes)
        return gemm_mx8.fwd_bias(self.xq, self.wq, None)

    def bwd(self, dz):
        """-> (dWm fp32 [4C, C], dX bf16 [M, C])."""
        m = self.modes
        if not (("dgrad" in m) or ("wgrad" in m)):
            return mm_wgrad(dz, self.x), mm_dgrad(dz, self.wm)
        gemm_mx8, mx8 = self._mx()
        dzq = mx8.quantize(dz, row="dgrad" in m, col="wgrad" in m)
        if "wgrad" in m:
            if self.xq is None or self.xq.qc is None:
                self.xq = mx8.quantize(self.x, row=False, col=True)
            dwm = gemm_mx8.wgrad(dzq, self.xq)
        else:
            dwm = mm_wgrad(dz, self.x)
        if "dgrad" in m:
            if self.wq is None or self.wq.qc is None:
                self.wq = mx8.quantize(self.wm, row=False, col=True)
            dx = gemm_mx8.dgrad(dzq, self.wq)
        else:
            dx = mm_dgrad(dz, self.wm)
        return dwm, dx


def _convt_to_gemm(w):
    """ConvTranspose2d weight [Cin,Cout,2,2] fp32 -> Wm [4*Cout, Cin] bf16 with Wm[(dy*2+dx)*Cout+co, ci] = w[ci,co,dy,dx]."""
    cin, cout = w.shape[0], w.shape[1]
    wm = torch.empty(4 * cout, cin, device=w.device, dtype=torch.bfloat16)
    wm.view(2, 2, cout, cin).copy_(w.permute(2, 3, 1, 0))
    return wm


def _gemm_grad_to_convt(dwm, cin, cout):
    """dWm [4*Cout, Cin] fp32 -> grad of the ConvTranspose2d weight [Cin,Cout,2,2] fp32."""
    return dwm.view(2, 2, cout, cin).permute(3, 2, 0, 1).contiguous()


def _dw_taps(w):
    """depthwise weight [C,1,3,3] -> [9, C] fp32 (tap-major, channel contiguous)."""
    return w.reshape(w.shape[0], 9).t().contiguous()


class _Upscale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tok, e, w1, b1, d1, g1, be1, w2, b2, d2, g2, be2, eps1, eps2, grid, nvtx=False):
        if nvtx:
            torch.cuda.nvtx.range_push("r14_up_fwd")
        ctx.nvtx = nvtx
        B, T, C = tok.shape
        h1 = grid
        h2 = 2 * grid
        tok2d = tok.reshape(B * T, C)
        wm1, wm2 = _convt_to_gemm(w1), _convt_to_gemm(w2)
        dwt1, dwt2 = _dw_taps(d1), _dw_taps(d2)
        modes = mx_modes()
        L1, L2 = _ConvT(tok2d, wm1, modes), None
        z1 = L1.fwd()
        y1, m1, r1 = ht.fwd(z1, b1, dwt1, g1, be1, B, h1, h1, eps1)
        L2 = _ConvT(y1, wm2, modes)
        z2 = L2.fwd()
        y2, m2, r2 = ht.fwd(z2, b2, dwt2, g2, be2, B, h2, h2, eps2)
        hw = (2 * h2) * (2 * h2)
        logits = einsum_fwd(e, y2.view(B, hw, C))
        ctx.layers = (L1, L2)
        ctx.save_for_backward(tok, e, dwt1, dwt2, b1, b2, g1, g2, z1, m1, r1, z2, y2, m2, r2)
        ctx.dims = (B, C, h1, h2, hw, w1.dtype)
        if nvtx:
            torch.cuda.nvtx.range_pop()
        return logits.view(B, e.shape[1], 2 * h2, 2 * h2)

    @staticmethod
    def backward(ctx, glogits):
        if ctx.nvtx:
            torch.cuda.nvtx.range_push("r14_up_bwd")
        tok, e, dwt1, dwt2, b1, b2, g1, g2, z1, m1, r1, z2, y2, m2, r2 = ctx.saved_tensors
        L1, L2 = ctx.layers
        ctx.layers = None
        B, C, h1, h2, hw, _ = ctx.dims
        dl = glogits.reshape(B, e.shape[1], hw).to(torch.bfloat16)
        dy2, de = einsum_bwd(dl, e, y2.view(B, hw, C))
        du2, dg2, dbe2 = ht.bwd1(z2, b2, dwt2, g2, m2, r2, dy2.view(-1, C), B, h2, h2)
        dz2, ddw2, db2 = ht.bwd2(z2, b2, dwt2, du2, B, h2, h2)
        dwm2, dy1 = L2.bwd(dz2)
        du1, dg1, dbe1 = ht.bwd1(z1, b1, dwt1, g1, m1, r1, dy1, B, h1, h1)
        dz1, ddw1, db1 = ht.bwd2(z1, b1, dwt1, du1, B, h1, h1)
        dwm1, dtok = L1.bwd(dz1)
        dtok = dtok.view(tok.shape)
        dw2c = _gemm_grad_to_convt(dwm2, C, C)
        dw1c = _gemm_grad_to_convt(dwm1, C, C)
        if ctx.nvtx:
            torch.cuda.nvtx.range_pop()
        return (dtok, de, dw1c, db1, ddw1.t().reshape(C, 1, 3, 3), dg1, dbe1,
                dw2c, db2, ddw2.t().reshape(C, 1, 3, 3), dg2, dbe2, None, None, None, None)


def upscale(model, tok, e, nvtx=False):
    """tok [B,1024,C] bf16 (normalised patch tokens), e [B,100,C] bf16 -> fp32 mask logits [B,100,128,128]."""
    s1, s2 = model.upscale[0], model.upscale[1]
    return _Upscale.apply(
        tok, e,
        s1.conv1.weight, s1.conv1.bias, s1.conv2.weight, s1.norm.weight, s1.norm.bias,
        s2.conv1.weight, s2.conv1.bias, s2.conv2.weight, s2.norm.weight, s2.norm.bias,
        float(s1.norm.eps), float(s2.norm.eps), 32, nvtx,
    )


def eligible(model, x, grid_size) -> bool:
    if not x.is_cuda or tuple(grid_size) != (32, 32):
        return False
    up = model.upscale
    if len(up) != 2:
        return False
    C = up[0].conv1.weight.shape[0]
    return C in (768, 1024) and up[0].conv1.kernel_size == (2, 2) and x.shape[1] == model.num_queries + 1 + model.backbone.n_storage_tokens + 1024


def predict(ctx, x, grid_size):
    model = ctx.train_model.model
    xn = model.backbone.norm(x)
    q = xn[:, : model.num_queries]
    class_logits = model.class_head(q)
    e = model.mask_head(q)
    prefix = model.num_queries + 1 + model.backbone.n_storage_tokens
    tok = xn[:, prefix:].to(torch.bfloat16).contiguous()
    mask_logits = upscale(model, tok, e.to(torch.bfloat16), bool(getattr(ctx, "nvtx", False)))
    return mask_logits, class_logits.float()
