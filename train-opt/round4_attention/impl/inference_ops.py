# copied from infer-opt/round4_attention/impl/ops.py (forward-only prototype; not used by training patch)
"""Round 4 attention ops (TileLang, sm_120). Interfaces:

  attention(q, k, v)                              q,k,v [B,H,N,64] bf16 contiguous -> [B,H,N,64]
  attention_qkv(qkv)                              qkv [B,N,3,H,64]                 -> [B,N,H*64]
  attention_rope(q_raw, k_raw, v, sin, cos, prefix)   RoPE (fp32) fused into the Q/K tile loads
  attention_rope_qkv(qkv, sin, cos, prefix)       same on the packed QKV layout
sin/cos: [HW, 64] fp32 as produced by backbone.rope_embed (rows are tokens prefix..N-1).
All kernels launch on the current stream, no host sync, compiled once per shape/config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_attn_tl as attn_tl  # noqa: E402

import json

# Tuned configs: impl/tuned_cfg.json {"<kind>|H|N": {bm,bn,stages,threads}} written by make_tuned.py from
# the tune_attn.py sweeps; bhnd variants reuse the packed-layout config.
DEFAULT_CFG = dict(bm=128, bn=32, stages=2, threads=256)
_TUNED_PATH = Path(__file__).resolve().parent / "tuned_cfg.json"
CFG: dict[str, dict] = json.loads(_TUNED_PATH.read_text()) if _TUNED_PATH.exists() else {}


def _cfg(kind: str, N: int, H: int, cfg: dict | None) -> dict:
    if cfg is not None:
        return cfg
    base = kind.replace("rope_qkv", "rope").replace("plain_qkv", "plain")
    for key in (f"{kind}|{H}|{N}", f"{kind.split('_')[0]}_qkv|{H}|{N}"):
        if key in CFG:
            return CFG[key]
    return DEFAULT_CFG


_DUMMY: dict = {}


def _dummy(hw: int, device):
    key = (hw, device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(hw, 64, dtype=torch.float32, device=device)
    return _DUMMY[key]


def attention(q, k, v, cfg: dict | None = None):
    B, H, N, D = q.shape
    assert D == 64 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    kern = attn_tl.build(B, H, N, D, layout="bhnd", rope=False, **_cfg("plain_qkv", N, H, cfg))
    return kern(q, k, v, _dummy(1024, q.device), _dummy(1024, q.device))


def attention_qkv(qkv, cfg: dict | None = None):
    B, N, three, H, D = qkv.shape
    assert three == 3 and D == 64 and qkv.is_contiguous()
    kern = attn_tl.build(B, H, N, D, layout="qkv", rope=False, **_cfg("plain_qkv", N, H, cfg))
    return kern(qkv, qkv, qkv, _dummy(1024, qkv.device), _dummy(1024, qkv.device)).view(B, N, H * D)


def attention_rope(q_raw, k_raw, v, sin, cos, prefix: int, cfg: dict | None = None):
    B, H, N, D = q_raw.shape
    hw = sin.shape[0]
    assert prefix == N - hw and sin.dtype == torch.float32 and sin.is_contiguous() and cos.is_contiguous()
    kern = attn_tl.build(B, H, N, D, layout="bhnd", rope=True, prefix=prefix, hw=hw, **_cfg("rope_qkv", N, H, cfg))
    return kern(q_raw, k_raw, v, sin, cos)


def attention_rope_qkv(qkv, sin, cos, prefix: int, cfg: dict | None = None):
    B, N, three, H, D = qkv.shape
    hw = sin.shape[0]
    assert three == 3 and prefix == N - hw and qkv.is_contiguous()
    kern = attn_tl.build(B, H, N, D, layout="qkv", rope=True, prefix=prefix, hw=hw, **_cfg("rope_qkv", N, H, cfg))
    return kern(qkv, qkv, qkv, sin, cos).view(B, N, H * D)


def attention_3bnhd(qkv, rope=None, cfg: dict | None = None):
    """qkv [3,B,N,H,64] (worker B's K2 physical layout) -> [B,N,H*64]. rope=(sin,cos,prefix) fuses RoPE."""
    three, B, N, H, D = qkv.shape
    assert three == 3 and D == 64 and qkv.is_contiguous()
    if rope is None:
        kern = attn_tl.build(B, H, N, D, layout="3bnhd", rope=False, **_cfg("plain_qkv", N, H, cfg))
        return kern(qkv, qkv, qkv, _dummy(1024, qkv.device), _dummy(1024, qkv.device)).view(B, N, H * D)
    sin, cos, prefix = rope
    kern = attn_tl.build(B, H, N, D, layout="3bnhd", rope=True, prefix=prefix, hw=sin.shape[0], **_cfg("rope_qkv", N, H, cfg))
    return kern(qkv, qkv, qkv, sin, cos).view(B, N, H * D)
