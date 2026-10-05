"""Round 10 GEMM candidate: the `mg_i8` prototype (int8 W8A8, s32 accumulate, per-token x per-channel dequant + R5 epilogues).  Quantisation of the activation NOT included."""
from __future__ import annotations

import torch

from bench10 import MODELS, Cand, R_FOR
import cands_base as cb

BF = torch.bfloat16


def make_mg8(cfg_override=None):
    def make(s: dict):
        import mg8
        M, N, K, kind, model = s["M"], s["N"], s["K"], s["kind"], s["model"]
        B, Nt = cb._batch(M)
        H = MODELS[model]["H"]
        R = R_FOR(N * K)
        X8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
        sx = torch.rand(M, device="cuda") * 0.01 + 0.001
        Ws = [torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8) for _ in range(R)]
        sws = [torch.rand(N, device="cuda") * 1e-3 + 1e-4 for _ in range(R)]
        bs = [cb._rand(N, scale=0.1) for _ in range(R)]
        aux = dict(sin=torch.randn(1024, 64, device="cuda").contiguous(), cos=torch.randn(1024, 64, device="cuda").contiguous(),
                   prefix=Nt - 1024, res=torch.randn(B, Nt, N, device="cuda", dtype=torch.float32), gamma=torch.rand(N, device="cuda", dtype=torch.float32))
        cfg = cfg_override(s) if callable(cfg_override) else cfg_override
        return [(lambda W=W, sw=sw, b=b: mg8.run_kind(kind, X8, sx, W, sw, b, cfg=cfg, aux=aux, B=B, heads=H)) for W, sw, b in zip(Ws, sws, bs)]
    return make


def cands(names):
    allc = [Cand("mg8_tuned", make_mg8(None), "mg_i8 prototype WITH R5 epilogue + per-token/per-channel dequant (tuned8.json tile), int8 operands; activation quantisation NOT included")]
    return [c for c in allc if names is None or c.name in names]
