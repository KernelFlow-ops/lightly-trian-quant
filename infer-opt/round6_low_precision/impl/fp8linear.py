"""Real FP8 (e4m3, per-token x per-output-channel scaled) Linear layers for the EoMT backbone.

Per FP8 unit (layer, type):
  qkv / fc1 : input comes from LayerNorm -> fused `ln_quant` kernel (norm1 / norm2 patched)
  proj      : `quant_rows` on the attention output
  fc2       : input comes from GELU -> fused `gelu_quant` (Mlp.act patched)
  GEMM      : torch._scaled_mm(fp8 x, fp8 W^T, rowwise scales, bias, out bf16)
All launches stay on the current stream, no host sync, static shapes (CUDA-graph capturable).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None
FP8 = torch.float8_e4m3fn


def ext():
    global _EXT
    if _EXT is None:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        build = Path("/root/autodl-tmp/infer-opt-tmp/postfuse/build_fp8")
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(name="fp8_ext", sources=[str(_HERE / "fp8_ext.cu")],
                    extra_cuda_cflags=["-O3", "-arch=sm_120a"], build_directory=str(build), verbose=False)
    return _EXT


def quant_weight(w: torch.Tensor):
    """Static per-output-channel e4m3 quantization. w: [N,K] -> (fp8 [N,K], scale fp32 [1,N])."""
    w = w.detach().float()
    s = (w.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
    q = (w / s).clamp(-448, 448).to(FP8).contiguous()
    return q, s.t().contiguous()


class _Ctx:
    """Scratch buffers per (M, K) so nothing is allocated twice for the same shape."""

    def __init__(self):
        self.buf = {}

    def get(self, M, K, dev):
        key = (M, K)
        if key not in self.buf:
            self.buf[key] = (torch.empty((M, K), dtype=FP8, device=dev), torch.empty((M, 1), dtype=torch.float32, device=dev))
        return self.buf[key]


def _scaled_linear(qx, sx, wq, sw, bias, out_shape):
    y = torch._scaled_mm(qx, wq.t(), scale_a=sx, scale_b=sw, bias=bias, out_dtype=torch.bfloat16)
    return y.view(*out_shape)


def enable(model, units) -> None:
    """Patch the given (layer, type) units of model.backbone.blocks with real FP8 GEMMs."""
    e = ext()
    units = set(map(tuple, units))
    ctx = _Ctx()
    stash: dict = {}
    x_lead: dict = {}
    blocks = model.backbone.blocks
    patched = []
    for i, blk in enumerate(blocks):
        types = {t for (li, t) in units if li == i}
        if not types:
            continue

        def make_lin(lin, key):
            wq, sw = quant_weight(lin.weight)
            bias = lin.bias.detach().to(torch.bfloat16).contiguous() if lin.bias is not None else None
            N = lin.weight.shape[0]

            def fwd(x):
                if key in stash:
                    qx, sx = stash.pop(key)
                    lead = x_lead.pop(key)
                else:
                    K = x.shape[-1]
                    lead = x.shape[:-1]
                    x2 = x.reshape(-1, K)
                    qx, sx = ctx.get(x2.shape[0], K, x.device)
                    e.quant_rows(x2.contiguous(), qx, sx)
                return _scaled_linear(qx, sx, wq, sw, bias, (*lead, N))

            lin.forward = fwd
            patched.append(lin)

        def make_ln(norm, key):
            g, b, eps = norm.weight.detach().float().contiguous(), norm.bias.detach().float().contiguous(), norm.eps

            def fwd(x):
                K = x.shape[-1]
                x2 = x.reshape(-1, K).contiguous()
                qx, sx = ctx.get(x2.shape[0], K, x.device)
                e.ln_quant(x2, g, b, qx, sx, eps)
                stash[key] = (qx, sx)
                x_lead[key] = x.shape[:-1]
                return x.new_empty(0)  # placeholder; the FP8 linear consumes the stash

            norm.forward = fwd
            patched.append(norm)

        def make_gelu(act, key):
            def fwd(y):
                K = y.shape[-1]
                y2 = y.reshape(-1, K).contiguous()
                qx, sx = ctx.get(y2.shape[0], K, y.device)
                # GELU+quant reads two sets of buffers: use a distinct scratch (fc2 input is 4K wide)
                e.gelu_quant(y2, qx, sx)
                stash[key] = (qx, sx)
                x_lead[key] = y.shape[:-1]
                return y.new_empty(0)

            act.forward = fwd
            patched.append(act)

        if "qkv" in types:
            make_ln(blk.norm1, (i, "qkv"))
            make_lin(blk.attn.qkv, (i, "qkv"))
        if "proj" in types:
            make_lin(blk.attn.proj, (i, "proj"))
        if "fc1" in types:
            make_ln(blk.norm2, (i, "fc1"))
            make_lin(blk.mlp.fc1, (i, "fc1"))
        if "fc2" in types:
            make_gelu(blk.mlp.act, (i, "fc2"))
            make_lin(blk.mlp.fc2, (i, "fc2"))
    model._fp8_patched = patched


def disable(model) -> None:
    for m in getattr(model, "_fp8_patched", []):
        m.__dict__.pop("forward", None)
    model._fp8_patched = []
