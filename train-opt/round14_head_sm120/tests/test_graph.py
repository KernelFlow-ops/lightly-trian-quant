"""CUDA-graph capturability of the round14 head (fwd + bwd): capture after warmup, replay with new inputs, compare with eager."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of  # noqa: E402
from round14_head_sm120.impl import head as r14head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--mx8", default="0")
    args = ap.parse_args()
    B, C = MODELS[args.model]
    mx = {"0": False, "1": True}.get(args.mx8, args.mx8)
    with gpu_lock():
        ns = build_head(C)
        ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
        r14head.POLICY["mx8"] = mx
        params = params_of(ns)
        gen = torch.Generator(device="cuda").manual_seed(1)
        x = (torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0).requires_grad_(True)
        go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
        go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
        for p in params:
            p.grad = torch.zeros_like(p)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m, c = r14head.predict(ctx, x, GRID)
            gx = torch.autograd.grad([m, c], [x, *params], [go_m, go_c], allow_unused=True)
            return m, c, gx

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m_s, c_s, gx_s = step()
        torch.cuda.synchronize()
        # replay with a new input, compare with eager on the same input
        with torch.no_grad():
            x.copy_(torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0)
        g.replay()
        torch.cuda.synchronize()
        m_g, c_g = m_s.clone(), c_s.clone()
        gx_g = [t.clone() for t in gx_s if t is not None]
        m_e, c_e, gx_e = step()
        gx_e = [t for t in gx_e if t is not None]
        torch.cuda.synchronize()

        def cos(a, b):
            a, b = a.float().reshape(-1), b.float().reshape(-1)
            return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))

        print(f"{args.model} mx8={mx}: graph replay vs eager  mask cos={cos(m_g, m_e):.8f} class cos={cos(c_g, c_e):.8f} "
              f"grad cos min={min(cos(a, b) for a, b in zip(gx_g, gx_e)):.8f} max|dmask|={float((m_g - m_e).abs().max()):.3e}")
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(20):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        print(f"graph replay fwd+bwd: {t0.elapsed_time(t1) / 20:.3f} ms per predict")


if __name__ == "__main__":
    main()
