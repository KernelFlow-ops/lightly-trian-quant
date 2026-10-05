"""Host (CPU enqueue) time of one predict fwd / bwd in eager mode, per variant (GPU synchronized between iterations)."""
import copy, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.bench import gpu_lock
from round11_bf16_mixed.impl import head as head_impl
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of, source_predict
from round14_head_sm120.impl import head as r14head

model = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
B, C = MODELS[model]
with gpu_lock():
    base = build_head(C)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0).requires_grad_(True)
    go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
    go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
    res = {}
    for name, mx in (("torch TB1 head", None), ("r14 bf16", False), ("r14 mx8 bwd", "bwd"), ("r14 mx8 all", True)):
        ns = copy.deepcopy(base)
        ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
        if mx is None:
            head_impl.install_ln2d_bf16(ctx)
            fn = lambda x: source_predict(ns, x)
        else:
            r14head.POLICY["mx8"] = mx
            fn = lambda x: r14head.predict(ctx, x, GRID)
        pl = [x, *params_of(ns)]
        tf, tb = [], []
        for it in range(25):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m, c = fn(x)
            t1 = time.perf_counter()
            torch.autograd.grad([m, c], pl, [go_m.to(m.dtype), go_c.to(c.dtype)], allow_unused=True)
            t2 = time.perf_counter()
            if it >= 5:
                tf.append((t1 - t0) * 1e3); tb.append((t2 - t1) * 1e3)
        tf.sort(); tb.sort()
        print(f"{model} {name:16s} host enqueue: fwd {tf[len(tf)//2]:.2f} ms  bwd {tb[len(tb)//2]:.2f} ms  total {tf[len(tf)//2]+tb[len(tb)//2]:.2f} ms", flush=True)
