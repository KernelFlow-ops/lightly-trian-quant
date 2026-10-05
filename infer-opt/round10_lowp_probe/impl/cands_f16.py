"""Round 10 GEMM candidates, part 2: the `mg_f16acc` prototype (fp16 operands, fp16-accumulate MMA + fp32 promotion, R5 epilogues)."""
from __future__ import annotations

import torch

from bench10 import MODELS, Cand, R_FOR
import cands_base as cb

F16, BF = torch.float16, torch.bfloat16


def make_mg16(cfg_override=None, label=""):
    def make(s: dict):
        import mg16
        M, N, K, kind, model = s["M"], s["N"], s["K"], s["kind"], s["model"]
        B, Nt = cb._batch(M)
        H = MODELS[model]["H"]
        R = R_FOR(N * K * 2)
        X = cb._rand(M, K, dtype=F16, scale=0.5)
        Ws = [cb._rand(N, K, dtype=F16, scale=0.03) for _ in range(R)]
        bs = [cb._rand(N, scale=0.1) for _ in range(R)]
        aux = dict(sin=torch.randn(1024, 64, device="cuda").contiguous(), cos=torch.randn(1024, 64, device="cuda").contiguous(),
                   prefix=Nt - 1024, res=torch.randn(B, Nt, N, device="cuda", dtype=torch.float32), gamma=torch.rand(N, device="cuda", dtype=torch.float32))
        cfg = cfg_override(s) if callable(cfg_override) else cfg_override
        return [(lambda W=W, b=b: mg16.run_kind(kind, X, W, b, cfg=cfg, aux=aux, B=B, heads=H)) for W, b in zip(Ws, bs)]
    return make


def cands(names):
    allc = [Cand("mg16_tuned", make_mg16(None), "mg_f16acc prototype WITH R5 epilogue (tuned16.json tile per shape), fp16 operands/outputs; operand pre-scale = free (oscale in epilogue)")]
    return [c for c in allc if names is None or c.name in names]
