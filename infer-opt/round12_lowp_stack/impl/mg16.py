# Round 12: verbatim copy of round10_lowp_probe/impl/mg16.py (fp16-accumulate GEMM wrappers).
"""Round 10 `mg_f16acc` prototype wrappers: fp16 operands, fp16-accumulate MMA with fp32 promotion, R5 epilogues.

Same call shape as round5_gemm_sm120/impl/ops.py (qkv_rope / gemm_bias_residual / gemm_bias_gelu) but A and W are fp16 tensors,
`oscale` is the power-of-two factor that undoes the operand pre-scaling (acc * oscale in the epilogue) and EPI 0/2 outputs are fp16.
Tile configs: `tuned16.json` (written by tune16.py), else the round5 defaults.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import mg16b  # noqa: E402

F16 = torch.float16
BF = torch.bfloat16

_EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"
DEFAULT = {  # round5 _DEFAULT (ViT-L M=2058 sweep) -- starting point for the sweep
    "fc1": dict(epi=0, bm=96, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, extra=_EX + " -DSTG=1"),
    "fc2": dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, splitk=2, extra=_EX),
    "proj": dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=1, wn=4, minb=3, extra=_EX),
    "qkv": dict(epi=2, bm=64, bn=128, bk=32, st=2, wm=2, wn=2, minb=3, extra=_EX + " -DSTG=1"),
}
for _k in ("plain", "ct1", "ct2", "einsum", "patch"):
    DEFAULT[_k] = dict(epi=4, bm=128, bn=128, bk=32, st=2, wm=2, wn=2, minb=2, extra=_EX)
import os
_T = _HERE / os.environ.get("MG16_TUNED", "tuned16.json")   # tuned16_pkt1.json: only configs that promote to fp32 every k-tile (fp16 accumulation over BK = 32/64 k)
TUNED: dict = json.loads(_T.read_text()) if _T.exists() else {}


def pick(kind, N, K, M):
    key = f"{kind}|{N}|{K}|{M}"
    if key in TUNED:
        return TUNED[key]["cfg"]
    best = None
    for k, v in TUNED.items():
        kk, nn, kd, mm = k.split("|")
        if kk == kind and int(nn) == N and int(kd) == K:
            dm = abs(int(mm) - M)
            if best is None or dm < best[0]:
                best = (dm, v["cfg"])
    return best[1] if best else DEFAULT[kind]


def nfull(cfg, N):
    ex = cfg.get("extra", "")
    if "-DNFULL=1" in ex and N % cfg["bn"]:
        cfg = dict(cfg, extra=ex.replace("-DNFULL=1", ""))
    return cfg


def gemm_bias_gelu16(h, W, b, cfg=None, oscale=1.0, out_dtype=F16):
    """a = GELU(h @ W^T * oscale + b), fp16 (or bf16 with -DOUTF16=0 via cfg outf16=0)."""
    h2 = h.reshape(-1, h.shape[-1])
    M, K = h2.shape
    N = W.shape[0]
    cfg = dict(cfg if cfg is not None else pick("fc1", N, K, M))
    cfg["outf16"] = 1 if out_dtype == F16 else 0
    out = torch.empty(M, N, device=h.device, dtype=out_dtype)
    mg16b.get(nfull(cfg, N))(h2, W, b, C=out, oscale=oscale)
    return out.view(*h.shape[:-1], N)


def gemm_bias_residual16(h, W, b, r, gamma, cfg=None, oscale=1.0):
    """out = r + gamma * (h @ W^T * oscale + b); r fp32 residual stream, fp32 out."""
    h2 = h.reshape(-1, h.shape[-1])
    M, K = h2.shape
    N = W.shape[0]
    kind = "fc2" if K > N else "proj"
    cfg = dict(cfg if cfg is not None else pick(kind, N, K, M))
    r2 = r.reshape(-1, N)
    out = torch.empty_like(r2)
    mg16b.get(nfull(dict(cfg, rf32=1), N))(h2, W, b, gamma, r2, out, oscale=oscale)
    return out.view(*h.shape[:-1], N)


def qkv_rope16(h, W, b, sin, cos, prefix, num_heads, cfg=None, oscale=1.0, out_dtype=F16):
    """h [B,N,d] fp16 -> physical [3,B,N,H,64] fp16 (RoPE fp32 in the epilogue), returned as the [3,B,H,N,64] permuted view like r5."""
    B, N, d = h.shape
    cfg = dict(cfg if cfg is not None else pick("qkv", 3 * d, d, B * N))
    cfg["outf16"] = 1 if out_dtype == F16 else 0
    out = torch.empty(3, B, N, num_heads, 64, device=h.device, dtype=out_dtype)
    mg16b.get(nfull(cfg, 3 * d))(h.reshape(B * N, d), W, b, C=out, sin=sin, cos=cos, Ntok=N, prefix=prefix, d=d, H=num_heads, oscale=oscale)
    return out.transpose(2, 3)


def gemm_plain16(h, W, cfg=None, oscale=1.0, out_dtype=BF, kind="plain"):
    """out = h @ W^T * oscale (no bias, no activation), bf16/fp16 out: head ConvT GEMMs, patch-embed, mask-einsum shapes."""
    M, K = h.shape
    N = W.shape[0]
    proxy = {"ct1": "fc1", "ct2": "fc1", "einsum": "proj", "patch": "proj"}.get(kind, kind)   # tile of the same-shape backbone GEMM as a starting point
    cfg = dict(cfg if cfg is not None else pick(proxy, N, K, M))
    cfg["epi"] = 4
    cfg["extra"] = cfg.get("extra", "").replace("-DSTG=1", "")     # STG staging only exists for the bf16-output epilogues 0 / 2
    cfg["outf16"] = 1 if out_dtype == F16 else 0
    out = torch.empty(M, N, device=h.device, dtype=out_dtype)
    mg16b.get(nfull(cfg, N))(h, W, None, C=out, oscale=oscale)
    return out


def run_kind(kind, X, W, b, *, cfg=None, aux=None, oscale=1.0, B=2, heads=16):
    """one call per GEMM kind with the benchmark-style arguments (aux = dict(sin, cos, prefix, res, gamma))."""
    if kind not in ("qkv", "proj", "fc1", "fc2"):
        return gemm_plain16(X, W, cfg, oscale, BF, kind)
    if kind == "qkv":
        return qkv_rope16(X.view(B, -1, X.shape[-1]), W, b, aux["sin"], aux["cos"], aux["prefix"], heads, cfg, oscale)
    if kind == "fc1":
        return gemm_bias_gelu16(X, W, b, cfg, oscale)
    return gemm_bias_residual16(X, W, b, aux["res"], aux["gamma"], cfg, oscale)
