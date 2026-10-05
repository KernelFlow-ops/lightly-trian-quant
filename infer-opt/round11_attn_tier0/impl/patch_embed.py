"""Round 11: patch-embed implicit-im2col GEMM wrapper (kernel: patch_embed.cu)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXTS: dict = {}
DEFAULT_CFG = dict(BM=64, BN=64, BK=32, ST=3, WM=2, WN=2, MINB=3)


def ext(**cfg):
    c = dict(DEFAULT_CFG)
    c.update(cfg)
    key = tuple(sorted(c.items()))
    if key not in _EXTS:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        tag = hashlib.md5(repr(key).encode()).hexdigest()[:8]
        build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_patch_" + tag)
        build.mkdir(parents=True, exist_ok=True)
        _EXTS[key] = load(name="r11_patch_" + tag, sources=[str(_HERE / "patch_embed.cu")], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a"] + [f"-D{k}={v}" for k, v in c.items()], verbose=False)
    return _EXTS[key]


class PatchEmbedGemm:
    """backbone -> x0 fp32 [B, 1+n_reg+gy*gx, C] (== bb.prepare_tokens_with_masks(x).float() under bf16 autocast)."""

    def __init__(self, bb, **cfg):
        pe = bb.patch_embed
        assert pe.patch_size == (16, 16) and pe.proj.in_channels == 3 and bb.n_storage_tokens == 4
        C = pe.proj.out_channels
        self.C = C
        with torch.no_grad():
            self.W = pe.proj.weight.detach().to(torch.bfloat16).reshape(C, 768).contiguous()
            self.bias = pe.proj.bias.detach().to(torch.bfloat16).float().contiguous()
            cls = bb.cls_token.detach().float() + 0 * bb.mask_token.detach().float()      # [1, C] (see prepare_tokens_with_masks)
            self.pre = torch.cat([cls.reshape(1, C), bb.storage_tokens.detach().float().reshape(-1, C)], 0).contiguous()
        self.ext = ext(**cfg)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        out = torch.empty(B, 5 + (H // 16) * (W // 16), self.C, device=x.device, dtype=torch.float32)
        self.ext.patch_embed_launch(x, self.W, self.bias, self.pre, out)
        return out
