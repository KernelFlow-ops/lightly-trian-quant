"""Round 10 `mg_i8` prototype wrappers: int8 W8A8 (per-token activation scale sx [M], per-output-channel weight scale sw [N]), s32 accumulate, R5 epilogues.

Same call shape as mg16.py / round5 ops (qkv_rope / gemm_bias_residual / gemm_bias_gelu) with extra (sx, sw).  Tile configs: tuned8.json, else the defaults below.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import mg8b  # noqa: E402

BF = torch.bfloat16
_EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"
DEFAULT = {
    "fc1": dict(epi=0, bm=128, bn=128, bk=32, st=3, wm=2, wn=2, minb=1, extra=_EX + " -DSTG=1"),
    "fc2": dict(epi=1, bm=128, bn=64, bk=32, st=3, wm=2, wn=2, minb=2, extra=_EX),
    "proj": dict(epi=1, bm=128, bn=64, bk=32, st=3, wm=2, wn=2, minb=2, extra=_EX),
    "qkv": dict(epi=2, bm=128, bn=128, bk=32, st=3, wm=2, wn=2, minb=1, extra=_EX + " -DSTG=1"),
}
for _k in ("plain", "ct1", "ct2", "einsum", "patch"):
    DEFAULT[_k] = dict(epi=4, bm=128, bn=128, bk=32, st=3, wm=2, wn=2, minb=1, extra=_EX)
_T = _HERE / os.environ.get("MG8_TUNED", "tuned8.json")
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


def gemm_bias_gelu8(X8, sx, W8, sw, b, cfg=None):
    M, K = X8.shape
    N = W8.shape[0]
    cfg = dict(cfg if cfg is not None else pick("fc1", N, K, M))
    out = torch.empty(M, N, device=X8.device, dtype=BF)
    mg8b.get(nfull(cfg, N))(X8, W8, b, C=out, cs=sw, stats_in=sx)
    return out


def gemm_bias_residual8(X8, sx, W8, sw, b, r, gamma, cfg=None):
    M, K = X8.shape
    N = W8.shape[0]
    kind = "fc2" if K > N else "proj"
    cfg = dict(cfg if cfg is not None else pick(kind, N, K, M))
    r2 = r.reshape(-1, N)
    out = torch.empty_like(r2)
    mg8b.get(nfull(dict(cfg, rf32=1), N))(X8, W8, b, gamma, r2, out, cs=sw, stats_in=sx)
    return out.view_as(r)


def qkv_rope8(X8, sx, W8, sw, b, sin, cos, B, Ntok, prefix, num_heads, cfg=None):
    M, d = X8.shape
    cfg = dict(cfg if cfg is not None else pick("qkv", 3 * d, d, M))
    out = torch.empty(3, B, Ntok, num_heads, 64, device=X8.device, dtype=BF)
    mg8b.get(nfull(cfg, 3 * d))(X8, W8, b, C=out, sin=sin, cos=cos, Ntok=Ntok, prefix=prefix, d=d, H=num_heads, cs=sw, stats_in=sx)
    return out.transpose(2, 3)


def gemm_plain8(X8, sx, W8, sw, cfg=None, kind="plain"):
    M, K = X8.shape
    N = W8.shape[0]
    proxy = {"ct1": "fc1", "ct2": "fc1", "einsum": "proj", "patch": "proj"}.get(kind, kind)   # tile of the same-shape backbone GEMM as a starting point
    cfg = dict(cfg if cfg is not None else pick(proxy, N, K, M))
    cfg["epi"] = 4
    cfg["extra"] = cfg.get("extra", "").replace("-DSTG=1", "")     # STG staging only exists for the bf16-output epilogues 0 / 2
    out = torch.empty(M, N, device=X8.device, dtype=BF)
    mg8b.get(nfull(cfg, N))(X8, W8, None, C=out, cs=sw, stats_in=sx)
    return out


def quant_rows(x):
    """per-row symmetric int8 (reference torch implementation, setup only)."""
    amax = x.float().abs().amax(1, keepdim=True)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    return torch.round(x.float() / s).clamp(-127, 127).to(torch.int8).contiguous(), s.view(-1).contiguous()


def run_kind(kind, X8, sx, W8, sw, b, *, cfg=None, aux=None, B=2, heads=16):
    if kind not in ("qkv", "proj", "fc1", "fc2"):
        return gemm_plain8(X8, sx, W8, sw, cfg, kind)
    if kind == "qkv":
        return qkv_rope8(X8, sx, W8, sw, b, aux["sin"], aux["cos"], B, X8.shape[0] // B, aux["prefix"], heads, cfg)
    if kind == "fc1":
        return gemm_bias_gelu8(X8, sx, W8, sw, b, cfg)
    return gemm_bias_residual8(X8, sx, W8, sw, b, aux["res"], aux["gamma"], cfg)
