"""Round 3 fused transformer-block ops (bf16, fp32 accumulate).

    add_layernorm(x, r, w, b)                       K1  CUDA C++ (one warp / row, 128-bit; x,r bf16 or fp32; w,b fp32)
    qkv_rope(h, W, b, sin, cos, prefix, heads)      K2  TileLang GEMM + RoPE/layout epilogue
    gemm_bias_residual(h, W, b, r)                  K3  TileLang GEMM + bias + residual epilogue
    gemm_bias_gelu(h, W, b)                         K4  TileLang GEMM + bias + GELU(erf) epilogue
    prepare_block_weights(block) -> dict            gamma folding, masked bias, bf16
    fused_block(x, w, sin, cos, prefix, heads)      full block, attention = torch SDPA (round4 replaces)

All kernels launch on the current torch stream, allocate only their outputs and cache compiled
kernels per shape (CUDA-Graph friendly).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ln_mod = None

# per-shape tile configs found by bench_kernels.py sweeps (see NOTES.md); default 128x128x32, 3 stages
_WS = dict(ws=True, tma=True)  # TMA + warp-specialized producer (sm_120 handles it fine, see NOTES)
TILE_CFG: dict[str, dict] = {
    # keys: gelu_{N}_{K} / res_{N}_{K} / qkv_{d}; values from tune_gemm.py sweeps (M=2058; ViT-L and ViT-B each tuned separately)
    "gelu_4096_1024": dict(bm=128, bn=64, bk=32, stages=4, threads=256, **_WS),
    "gelu_3072_768": dict(bm=64, bn=64, bk=32, stages=4, threads=256, **_WS),
    "res_1024_4096": dict(bm=128, bn=128, bk=32, stages=4, threads=256, **_WS),
    "res_768_3072": dict(bm=64, bn=64, bk=32, stages=3, threads=128, ws=False, tma=False),
    "res_1024_1024": dict(bm=64, bn=128, bk=32, stages=3, threads=256, **_WS),
    "res_768_768": dict(bm=64, bn=64, bk=32, stages=4, threads=128, ws=False, tma=False),
    "qkv_1024": dict(bm=128, bn=64, bk=32, stages=3, threads=128, ws=False, tma=False),
    "qkv_768": dict(bm=64, bn=128, bk=32, stages=4, threads=128, ws=False, tma=False),
}


def _ln():
    global _ln_mod
    if _ln_mod is None:
        from torch.utils.cpp_extension import load

        build = "/root/autodl-tmp/infer-opt-tmp/blockfuse/build"
        os.makedirs(build, exist_ok=True)
        _ln_mod = load(
            name="r3_add_ln", sources=[str(_HERE / "ln_kernels.cu")], build_directory=build,
            extra_cuda_cflags=["-arch=sm_120a", "-O3", "--use_fast_math"], verbose=False,
        )
    return _ln_mod


def add_layernorm(x, r, w, b, eps: float = 1e-5):
    """returns (s, y): s = x + r (bf16, new residual; == x when r is None), y = LN(s)."""
    return tuple(_ln().add_layernorm(x, r, w, b, eps))


def gemm_bias_gelu(h, W, b, cfg=None):
    from .gemm_tl import gemm_bias_gelu_kernel

    shp = h.shape
    h2 = h.reshape(-1, shp[-1])
    cfg = cfg or TILE_CFG.get(f"gelu_{W.shape[0]}_{W.shape[1]}", {})
    k = gemm_bias_gelu_kernel(h2.shape[0], W.shape[0], W.shape[1], **cfg)
    return k(h2, W, b).view(*shp[:-1], W.shape[0])


_ONES: dict = {}


def gemm_bias_residual(h, W, b, r, gamma=None, cfg=None):
    """out = r + gamma * (h @ W^T + b); r bf16 or fp32 (fp32 residual stream), output has r's dtype.
    gamma: fp32 [N] LayerScale applied in the fp32 epilogue (None -> 1)."""
    from .gemm_tl import gemm_bias_residual_kernel

    shp = h.shape
    h2 = h.reshape(-1, shp[-1])
    cfg = dict(cfg or TILE_CFG.get(f"res_{W.shape[0]}_{W.shape[1]}", {}))
    cfg["res_dtype"] = "float32" if r.dtype == torch.float32 else "bfloat16"
    k = gemm_bias_residual_kernel(h2.shape[0], W.shape[0], W.shape[1], **cfg)
    if gamma is None:
        key = (W.shape[0], W.device)
        gamma = _ONES.get(key)
        if gamma is None:
            gamma = _ONES[key] = torch.ones(W.shape[0], device=W.device, dtype=torch.float32)
    return k(h2, W, b, gamma, r.reshape(-1, W.shape[0])).view(*shp[:-1], W.shape[0])


def qkv_rope(h, W, b, sin, cos, prefix: int, num_heads: int, cfg=None, round_bf16: bool = False,
             layout: str = "bnhd"):
    """h [B,N,d] -> logical out [3,B,H,N,64] (out[0]=q, out[1]=k roped for tokens >= prefix, out[2]=v).
    layout="bnhd" (default): physical [3,B,N,H,64], returned as a permuted view (SDPA-friendly, no copies);
    layout="bhnd": physical [3,B,H,N,64] contiguous."""
    from .gemm_tl import qkv_rope_kernel

    B, N, d = h.shape
    cfg = cfg or TILE_CFG.get(f"qkv_{d}", {})
    k = qkv_rope_kernel(B, N, d, num_heads, prefix, round_bf16=round_bf16, layout=layout, **cfg)
    out = k(h.reshape(B * N, d), W, b, sin, cos)
    return out.transpose(2, 3) if layout == "bnhd" else out


@torch.no_grad()
def prepare_block_weights(block, fold_gamma: bool = False) -> dict:
    """SelfAttentionBlock -> bf16 contiguous weights, K-bias masked (bias * bias_mask precomputed).

    fold_gamma=True : LayerScale folded into Wproj/bproj/W2/b2 (bf16 rounding of gamma*W; simplest, but the
                      weight-rounding error is visible on the massive-activation outlier tokens, see NOTES).
    fold_gamma=False: raw weights + fp32 gamma vectors g1/g2 applied in the GEMM epilogue (default)."""
    qkv = block.attn.qkv
    bq = qkv.bias
    if hasattr(qkv, "bias_mask"):
        bq = bq * qkv.bias_mask.to(bq.dtype)
    g1 = block.ls1.gamma.float()
    g2 = block.ls2.gamma.float()
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731
    f32 = lambda t: t.detach().float().contiguous()  # noqa: E731  (LN affine kept in fp32)
    one = torch.ones_like(g1)
    fg1, fg2 = (g1, g2) if fold_gamma else (one, one)
    return {
        "g1": (one if fold_gamma else g1).contiguous(), "g2": (one if fold_gamma else g2).contiguous(),
        "ln1_w": f32(block.norm1.weight), "ln1_b": f32(block.norm1.bias), "ln1_eps": block.norm1.eps,
        "Wqkv": bf(qkv.weight), "bqkv": bf(bq),
        "Wproj": bf(fg1[:, None] * block.attn.proj.weight.float()),
        "bproj": bf(fg1 * block.attn.proj.bias.float()),
        "ln2_w": f32(block.norm2.weight), "ln2_b": f32(block.norm2.bias), "ln2_eps": block.norm2.eps,
        "W1": bf(block.mlp.fc1.weight), "b1": bf(block.mlp.fc1.bias),
        "W2": bf(fg2[:, None] * block.mlp.fc2.weight.float()),
        "b2": bf(fg2 * block.mlp.fc2.bias.float()),
    }


def fused_block(x, w: dict, sin, cos, prefix: int, num_heads: int, rope_round_bf16: bool = False):
    """x [B,N,d] -> x_out (same dtype).  7 kernels: LN1, qkv+rope, SDPA, proj+res, LN2, fc1+gelu, fc2+res.
    x should be an fp32 residual stream (what autocast does); a bf16 stream fails the model-level accuracy gate on ViT-L."""
    B, N, d = x.shape
    _, h = add_layernorm(x, None, w["ln1_w"], w["ln1_b"], w["ln1_eps"])
    qkv = qkv_rope(h, w["Wqkv"], w["bqkv"], sin, cos, prefix, num_heads, round_bf16=rope_round_bf16)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # [B,H,N,64] view of [B,N,H,64] storage
    o = o.transpose(1, 2).reshape(B, N, d)  # free view: flash writes [B,N,H,64]
    x = gemm_bias_residual(o, w["Wproj"], w["bproj"], x, w["g1"])
    _, h = add_layernorm(x, None, w["ln2_w"], w["ln2_b"], w["ln2_eps"])
    a = gemm_bias_gelu(h, w["W1"], w["b1"])
    return gemm_bias_residual(a, w["W2"], w["b2"], x, w["g2"])
