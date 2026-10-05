"""P3: fused EoMT mask-head upscale (2x ScaleBlock) + mask einsum for bf16 inference.

ScaleBlock = ConvT(k2,s2) -> GELU -> DWConv3x3 -> LayerNorm2D.  Here:
  1. ConvT as GEMM  X[B*Hs*Ws, C] @ Wr[C, 4C]  (torch.mm / cuBLAS, bf16)
  2. one CUDA kernel: +bias, GELU, pixel-shuffle, DWConv3x3, LN over C  -> NHWC bf16
Final einsum('bqc,bchw->bqhw') becomes bmm(mask_head(q), U^T) on the NHWC tensor.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        build = Path("/root/autodl-tmp/infer-opt-tmp/postfuse/build_head")
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(name="head_ext", sources=[str(_HERE / "head_ext.cu")],
                    extra_cuda_cflags=["-O3", "-arch=sm_120a"], build_directory=str(build),
                    verbose=False)
    return _EXT


class FastHead:
    def __init__(self, model):
        self.model = model
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

    def upscale_nhwc(self, x_tok: torch.Tensor, B: int, Hs: int, Ws: int) -> torch.Tensor:
        """x_tok: [B*Hs*Ws, C] bf16 patch tokens -> [B, 4Hs, 4Ws, C] bf16 (NHWC)."""
        e = ext()
        z = x_tok
        for st in self.stages:
            g = torch.mm(z, st["wr"])  # [B*Hs*Ws, 4C] pre-shuffle
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=torch.bfloat16, device=x_tok.device)
            e.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out

    def predict(self, x: torch.Tensor, grid_size: tuple[int, int]):
        """Drop-in for task_model._predict (bf16 / autocast inference only)."""
        m = self.model
        q = x[:, : m.num_queries, :]
        class_logits = m.class_head(q)
        xt = x[:, m.num_queries + 1 + m.backbone.n_storage_tokens:, :]
        B, N, C = xt.shape
        Hs, Ws = grid_size
        xt = xt.to(torch.bfloat16).reshape(B * N, C)
        u = self.upscale_nhwc(xt, B, Hs, Ws)  # [B,H,W,C]
        mq = m.mask_head(q).to(torch.bfloat16)  # [B,Q,C]
        ml = torch.bmm(mq, u.view(B, -1, C).transpose(1, 2))  # [B,Q,H*W]
        return ml.view(B, mq.shape[1], u.shape[1], u.shape[2]), class_logits


def enable(model) -> FastHead:
    fh = FastHead(model)
    model._predict = fh.predict  # instance attribute shadows the bound method
    model._fast_head = fh
    return fh


def disable(model) -> None:
    if "_predict" in model.__dict__:
        del model.__dict__["_predict"]
