"""Categorize GPU kernels of an nsys profile by NVTX range (fwd / bwd of one block) from the exported sqlite.

  python tests/nsys_categorize.py <prefix> [--iters 5] [--json out.json]

Kernel -> NVTX range: the host launch time (CUPTI runtime record with the kernel's correlation id) falls inside the
range's [start, end] host interval (autograd backward threads launch inside the ``backward()`` call).  Categories are
regexes on the kernel name; everything not matched is ``other`` and listed.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

CATS = [
    ("attention / RoPE", re.compile(r"flash|fmha|sdpa|cudnn.*(attn|mha|sdpa)|attention|mha|rope|_rotate|bwd_dot_do_o|convert_dq", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|nvjet|xmma|cublas|sm80_|sm90_|sm100_|sm120_|quack|turing_|ampere_bf16|splitK", re.I)),
    ("norm/residual (fused CUDA)", re.compile(r"ln_fwd_kernel|rowbwd_kernel|colsum_bf16|finalize_kernel")),
    ("elementwise/copy/reduce", re.compile(r"elementwise|vectorized|copy|cast|reduce|add|mul|fill|index|gelu|catArray|transpose|layer_norm|LayerNorm|GammaBeta|multi_tensor", re.I)),
]


def category(name: str) -> str:
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "other"


def short(name: str) -> str:
    name = re.sub(r"\(.*", "", name)
    return name[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    con = sqlite3.connect(args.prefix + ".sqlite")
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    runtime = {r[0]: r[1] for r in con.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    kernels = list(con.execute("SELECT correlationId, start, end, demangledName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    nvtx = [(r[0], r[1], r[2]) for r in con.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL") if r[1] is not None]
    nvtx = [(s, e, t) for s, e, t in nvtx if re.search(r"/(fwd|bwd)$", t)]
    out = defaultdict(lambda: {"count": 0, "us": 0.0, "cats": defaultdict(lambda: [0, 0.0]), "kernels": defaultdict(lambda: [0, 0.0])})
    for corr, ks, ke, nid, gx, gy, gz in kernels:
        t = runtime.get(corr)
        if t is None:
            continue
        rng = next((tag.split("/")[-1] for s, e, tag in nvtx if s <= t <= e and not tag.startswith("warm")), None)
        if rng is None:
            continue
        nm = names.get(nid, str(nid))
        us = (ke - ks) / 1e3
        o = out[rng]
        o["count"] += 1
        o["us"] += us
        c = category(nm)
        o["cats"][c][0] += 1
        o["cats"][c][1] += us
        o["kernels"][short(nm)][0] += 1
        o["kernels"][short(nm)][1] += us
    res = {}
    for rng in ("fwd", "bwd"):
        o = out.get(rng)
        if not o:
            continue
        n = args.iters
        res[rng] = {
            "kernels_per_iter": o["count"] / n, "gpu_us_per_iter": o["us"] / n,
            "categories": {c: {"kernels": v[0] / n, "us": v[1] / n} for c, v in sorted(o["cats"].items(), key=lambda kv: -kv[1][1])},
            "top_kernels": [{"name": k, "count": v[0] / n, "us": v[1] / n} for k, v in sorted(o["kernels"].items(), key=lambda kv: -kv[1][1])[:30]],
        }
        print(f"== {rng}: {res[rng]['kernels_per_iter']:.1f} kernels/iter, {res[rng]['gpu_us_per_iter']:.0f} us GPU/iter (sum of kernel durations)")
        for c, v in res[rng]["categories"].items():
            print(f"   {c:28s} {v['kernels']:5.1f} kernels {v['us']:8.1f} us")
        for k in res[rng]["top_kernels"]:
            print(f"      {k['count']:5.1f} x {k['us']:8.1f} us  {k['name']}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
