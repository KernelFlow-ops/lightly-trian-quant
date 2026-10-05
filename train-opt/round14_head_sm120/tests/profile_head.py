"""One-predict nsys capture (fwd + bwd) of the TB1 head (torch path + ln_bf16) at the real shapes.

Usage: bash common/profile_nsys.sh <prefix> round14_head_sm120/tests/profile_head.py --model vitl16 [--variant torch|r14|r14_mx8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import bench  # noqa: E402
from round11_bf16_mixed.impl import head as head_impl  # noqa: E402
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of, source_predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--variant", default="torch")
    args = ap.parse_args()
    B, dim = MODELS[args.model]
    ns = build_head(dim)
    head_impl.install_ln2d_bf16(SimpleNamespace(model=ns))
    if args.variant == "torch":
        fn = lambda x: source_predict(ns, x)
    else:
        from round14_head_sm120.impl import head as r14head
        from round11_bf16_mixed.tests.bench_head import GRID as _G
        r14head.POLICY["mx8"] = {"r14": False, "r14_mx8": True}[args.variant]
        ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
        fn = lambda x: r14head.predict(ctx, x, _G)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(B, TOKENS, dim, device="cuda", generator=gen) * 2.0).requires_grad_(True)
    go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
    go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
    plist = [x, *params_of(ns)]

    def step(i: int) -> None:
        with bench.nvtx("fwd"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ml, cl = fn(x)
        with bench.nvtx("bwd"):
            torch.autograd.grad([ml, cl], plist, [go_m.to(ml.dtype), go_c.to(cl.dtype)], allow_unused=True)

    bench.profile_steps(step, warmup=20, steps=5)


if __name__ == "__main__":
    main()
