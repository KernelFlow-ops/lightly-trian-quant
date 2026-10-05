"""Round 11 fused mask head  (copy-and-modify of round7_head_postprocess/impl/head.py).

Changes vs round 7:
  * dw_ln kernel from head_ext11.cu (channel-pass halo tiles, sliding-window dwconv; bit-identical output with exact GELU).
  * `tail=True`: one kernel (tail_ln_kernel) does bb.norm + bf16 cast + row gather (q rows / patch rows) + class_head; mask_head runs on
    pre-cast bf16 weights (no per-call autocast weight casts); no strided .to(bf16) copy, no cutlass align1 class-head GEMM.
    Numerically equal to  autocast(model._predict(bb.norm(t)))  up to LN summation order.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXTS: dict = {}


def ext(**cfg):
    key = tuple(sorted(cfg.items()))
    if key not in _EXTS:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        tag = hashlib.md5(repr(sorted(cfg.items())).encode()).hexdigest()[:8]   # same tag scheme as tests/test_dwln.py (shared build cache)
        build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_dwln_" + tag)
        build.mkdir(parents=True, exist_ok=True)
        _EXTS[key] = load(name="r11_dwln_" + tag, sources=[str(_HERE / "head_ext11.cu")], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a"] + [f"-D{k}={v}" for k, v in cfg.items()], verbose=False)
    return _EXTS[key]


DEFAULT_DWLN = dict(TH=4, TW=8, PPW=2, MINB=2)   # best of tests/test_dwln.py sweep (logs/dwln_sweep2.log)


class FastHead11:
    def __init__(self, model, dwln_cfg=None):
        self.model = model
        self.ext = ext(**(dwln_cfg or DEFAULT_DWLN))
        self.stages = []
        for blk in model.upscale:
            c1, c2, nrm = blk.conv1, blk.conv2, blk.norm
            C = c1.weight.shape[0]
            wr = c1.weight.detach().permute(0, 2, 3, 1).reshape(C, 4 * C).to(torch.bfloat16).contiguous()
            self.stages.append({
                "wr": wr,
                "bias": c1.bias.detach().to(torch.bfloat16).float().contiguous(),
                "dww": c2.weight.detach().reshape(C, 9).t().to(torch.bfloat16).contiguous(),
                "g": nrm.weight.detach().float().contiguous(),
                "b": nrm.bias.detach().float().contiguous(),
                "eps": nrm.eps,
            })
        self.C = C
        bf = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731
        self.cls_w, self.cls_b = bf(model.class_head.weight), bf(model.class_head.bias).float().contiguous()
        mh = [m for m in model.mask_head if isinstance(m, torch.nn.Linear)]
        self.mh = [(bf(m.weight), bf(m.bias)) for m in mh]
        self.Q = model.num_queries
        self.npre = 1 + model.backbone.n_storage_tokens
        self.ln_g = model.backbone.norm.weight.detach().float().contiguous()
        self.ln_b = model.backbone.norm.bias.detach().float().contiguous()
        self.ln_eps = model.backbone.norm.eps

    def upscale_nhwc(self, x_tok: torch.Tensor, B: int, Hs: int, Ws: int) -> torch.Tensor:
        z = x_tok
        for st in self.stages:
            g = torch.mm(z, st["wr"])
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=torch.bfloat16, device=x_tok.device)
            self.ext.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out

    def predict_tail(self, t: torch.Tensor, grid_size):
        """t: fp32 final residual stream [B, Q+5+gy*gx, C] (before bb.norm)  ->  (mask_logits [B,Q,4gy,4gx] bf16, class_logits [B,Q,3] bf16)."""
        B, ntok, C = t.shape
        Q, Hs, Ws = self.Q, grid_size[0], grid_size[1]
        dev = t.device
        qb = torch.empty(B * Q, C, device=dev, dtype=torch.bfloat16)
        xb = torch.empty(B * Hs * Ws, C, device=dev, dtype=torch.bfloat16)
        cl = torch.empty(B, Q, 3, device=dev, dtype=torch.bfloat16)
        self.ext.tail_ln_launch(t, self.ln_g, self.ln_b, qb, xb, self.cls_w, self.cls_b, cl, Q, self.npre, self.ln_eps)
        u = self.upscale_nhwc(xb, B, Hs, Ws)
        h = qb
        for i, (w, b) in enumerate(self.mh):
            h = F.linear(h, w, b)
            if i < len(self.mh) - 1:
                h = F.gelu(h)
        ml = torch.bmm(h.view(B, Q, C), u.view(B, -1, C).transpose(1, 2))
        return ml.view(B, Q, u.shape[1], u.shape[2]), cl
