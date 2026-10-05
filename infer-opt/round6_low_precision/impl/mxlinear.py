"""Real MXFP8 E4M3 Linear layers with 1x32 E8M0 block scales on sm_120.

Weights are quantized once; activations use a CUDA kernel that writes FP8 values
and cuBLAS's swizzled scale layout in one pass. LayerNorm and GELU can feed the
quantized Linear without materializing their bf16 outputs. The patched model is
CUDA-graph capturable for the fixed batch shapes used by video inference.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None
FP8 = torch.float8_e4m3fn
SCALE = torch.float8_e8m0fnu


def ext():
    global _EXT
    if _EXT is None:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        build = Path("/root/autodl-tmp/infer-opt-tmp/postfuse/build_mx")
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(name="mx_ext", sources=[str(_HERE / "mx_ext.cu")],
                    extra_cuda_cflags=["-O3", "-arch=sm_120a"], build_directory=str(build), verbose=False)
    return _EXT


def scale_len(M: int, K: int) -> int:
    if K % 256:
        raise ValueError("MXFP8 requires K divisible by 256")
    return ((M + 127) // 128) * 128 * (K // 32)


def empty_quant(M: int, K: int, device: torch.device):
    q = torch.empty((M, K), dtype=FP8, device=device)
    s = torch.empty(scale_len(M, K), dtype=SCALE, device=device)
    s.view(torch.uint8).fill_(127)  # padded rows/columns carry the E8M0 value 1
    return q, s


def quant_weight(w: torch.Tensor):
    w = w.detach().float().contiguous()
    q, s = empty_quant(*w.shape, w.device)
    ext().quant_rows(w, q, s)
    return q, s


class _Ctx:
    def __init__(self):
        self.buf = {}

    def get(self, M, K, device):
        key = (M, K, device)
        if key not in self.buf:
            self.buf[key] = empty_quant(M, K, device)
        return self.buf[key]


def enable(model, units) -> None:
    """Patch selected (layer, qkv|proj|fc1|fc2) units of model.backbone.blocks."""
    e = ext()
    units = set(map(tuple, units))
    ctx = _Ctx()
    stash = {}
    leads = {}
    patched = []
    for i, blk in enumerate(model.backbone.blocks):
        types = {t for li, t in units if li == i}
        if not types:
            continue

        def make_lin(lin, key):
            wq, sw = quant_weight(lin.weight)
            bias = lin.bias.detach().to(torch.bfloat16).contiguous() if lin.bias is not None else None
            N = lin.weight.shape[0]

            def fwd(x):
                if key in stash:
                    qx, sx = stash.pop(key)
                    lead = leads.pop(key)
                else:
                    K = x.shape[-1]
                    lead = x.shape[:-1]
                    x2 = x.reshape(-1, K).contiguous()
                    qx, sx = ctx.get(x2.shape[0], K, x.device)
                    e.quant_rows(x2, qx, sx)
                y = torch._scaled_mm(qx, wq.t(), scale_a=sx, scale_b=sw,
                                     bias=bias, out_dtype=torch.bfloat16)
                return y.view(*lead, N)

            lin.forward = fwd
            patched.append(lin)

        def make_ln(norm, key):
            g = norm.weight.detach().float().contiguous()
            b = norm.bias.detach().float().contiguous()

            def fwd(x):
                K = x.shape[-1]
                x2 = x.reshape(-1, K).contiguous()
                qx, sx = ctx.get(x2.shape[0], K, x.device)
                e.ln_quant(x2, g, b, qx, sx, norm.eps)
                stash[key] = (qx, sx)
                leads[key] = x.shape[:-1]
                return x.new_empty(0)

            norm.forward = fwd
            patched.append(norm)

        def make_gelu(act, key):
            def fwd(x):
                K = x.shape[-1]
                x2 = x.reshape(-1, K).contiguous()
                qx, sx = ctx.get(x2.shape[0], K, x.device)
                e.gelu_quant(x2, qx, sx)
                stash[key] = (qx, sx)
                leads[key] = x.shape[:-1]
                return x.new_empty(0)

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
    model._mx_patched = patched


def disable(model) -> None:
    for m in getattr(model, "_mx_patched", []):
        m.__dict__.pop("forward", None)
    model._mx_patched = []
