import sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.bench import gpu_lock
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of
from round14_head_sm120.impl import head as r14head
B, C = MODELS["vitb16"]
with gpu_lock():
    ns = build_head(C)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0).requires_grad_(True)
    go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
    go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
    outs = []
    for flag in (False, True):
        ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns, nvtx=flag)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, c = r14head.predict(ctx, x, GRID)
        g = torch.autograd.grad([m, c], [x, *params_of(ns)], [go_m, go_c], allow_unused=True)
        outs.append((m.detach(), [t.detach() for t in g if t is not None]))
    print("nvtx on/off identical:", torch.equal(outs[0][0], outs[1][0]), all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1])))
