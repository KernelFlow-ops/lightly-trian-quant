"""One predict (fwd+bwd, fwd only) at the real shapes: TB1 torch head (head_ln_bf16) vs the round14 fused head variants.

Candidates are alternated in one process (round11 microbench.compare protocol: warmup, 3 repeats median).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.impl import head as head_impl  # noqa: E402
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of, source_predict  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare, fmt  # noqa: E402
from round14_head_sm120.impl import head as r14head  # noqa: E402


def with_policy(policy, fn):
    def run(*a, **k):
        old = dict(r14head.POLICY)
        r14head.POLICY.update(policy)
        try:
            return fn(*a, **k)
        finally:
            r14head.POLICY.clear()
            r14head.POLICY.update(old)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    report = {}
    with gpu_lock():
        for model in args.models:
            B, C = MODELS[model]
            base = build_head(C)
            gen = torch.Generator(device="cuda").manual_seed(1)
            x = (torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0).requires_grad_(True)
            go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
            go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
            cands = {}
            ns0 = copy.deepcopy(base)
            head_impl.install_ln2d_bf16(SimpleNamespace(model=ns0))
            cands["TB1 torch head_ln_bf16"] = (ns0, lambda x, ns=ns0: source_predict(ns, x), None)
            variants = {
                "r14 fused (quack fwd/dgrad, torch wgrad, quack e_bwd)": {},
                "r14 fused all-torch GEMMs": dict(fwd="torch", dgrad="torch", wgrad="torch", e_fwd="torch", e_bwd="torch"),
                "r14 fused all-quack GEMMs": dict(fwd="quack", dgrad="quack", wgrad="quack", e_fwd="quack", e_bwd="quack"),
                "r14 fused + MXFP8 ConvT all": dict(mx8=True),
                "r14 fused + MXFP8 ConvT bwd only": dict(mx8="bwd"),
                "r14 fused + MXFP8 ConvT wgrad only": dict(mx8="wgrad"),
            }
            for name, pol in variants.items():
                ns = copy.deepcopy(base)
                ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
                cands[name] = (ns, lambda x, ctx=ctx: r14head.predict(ctx, x, GRID), pol)

            def fwdbwd(ns, fn, pol):
                def run():
                    old = dict(r14head.POLICY)
                    if pol:
                        r14head.POLICY.update(pol)
                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            m, c = fn(x)
                        torch.autograd.grad([m, c], [x, *params_of(ns)], [go_m.to(m.dtype), go_c.to(c.dtype)], allow_unused=True)
                    finally:
                        r14head.POLICY.clear()
                        r14head.POLICY.update(old)
                return run

            def fwd_only(ns, fn, pol):
                def run():
                    old = dict(r14head.POLICY)
                    if pol:
                        r14head.POLICY.update(pol)
                    try:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            fn(x)
                    finally:
                        r14head.POLICY.clear()
                        r14head.POLICY.update(old)
                return run

            t = compare({k: fwdbwd(*v) for k, v in cands.items()}, iters=args.iters)
            tf = compare({k: fwd_only(*v) for k, v in cands.items()}, iters=args.iters)
            print(f"=== {model} B={B} C={C}: one predict fwd+bwd", flush=True)
            print(fmt(t), flush=True)
            print(f"--- {model}: fwd only", flush=True)
            print(fmt(tf), flush=True)
            report[model] = {"fwd_bwd_ms": {k: v.get("ms") for k, v in t.items()}, "fwd_ms": {k: v.get("ms") for k, v in tf.items()},
                             "errors": {k: v["error"] for k, v in {**t, **tf}.items() if "error" in v}}
            del cands, base, x
            torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
