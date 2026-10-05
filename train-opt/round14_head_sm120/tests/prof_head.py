"""torch.profiler kernel table (self CUDA time per kernel, per predict fwd+bwd) of the round14 head variants."""
import argparse, copy, sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.bench import gpu_lock
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of
from round14_head_sm120.impl import head as r14head
from torch.profiler import profile, ProfilerActivity

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitl16")
ap.add_argument("--mx8", default="0")
ap.add_argument("--rows", type=int, default=30)
args = ap.parse_args()
B, C = MODELS[args.model]
mx = {"0": False, "1": True}.get(args.mx8, args.mx8)
with gpu_lock():
    base = build_head(C)
    ns = base
    ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
    r14head.POLICY["mx8"] = mx
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0).requires_grad_(True)
    go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
    go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
    plist = [x, *params_of(ns)]
    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, c = r14head.predict(ctx, x, GRID)
        torch.autograd.grad([m, c], plist, [go_m.to(m.dtype), go_c.to(c.dtype)], allow_unused=True)
    for _ in range(8): run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): run()
        torch.cuda.synchronize()
    ka = [e for e in prof.key_averages() if e.device_time_total > 0]
    ka.sort(key=lambda e: -e.device_time_total)
    tot = sum(e.device_time_total for e in ka) / 5
    print(f"== {args.model} mx8={mx}: total kernel time per predict {tot/1e3:.3f} ms")
    for e in ka[: args.rows]:
        print(f"{e.device_time_total/5:9.1f} us  x{e.count/5:4.1f}  {e.key[:110]}")
