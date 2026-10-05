"""R24 HOT-input micro-benchmark of the FP4 quantizers (CUDA-graph replay, candidates interleaved in one process, median of >= 3 rounds, dual GPU lock).

Hot input = the tensor was written by the previous kernel of the same graph (a producer kernel ``x.uniform_()`` precedes every quantizer launch), so it sits in L2 like the output of
the producer GEMM / LN kernel in the real step.  Per candidate: t_quant = (T[producer + quantizer] - T[producer only]) / reps (both graphs hold ``reps`` back-to-back copies).

  python tests/bench_hot.py --cands r21a,base[,name:-DFOO=1 ...] [--models vitl16,vitb16] [--sites dy,x,w,x8,xd] [--shapes quick] --tag NAME     -> profile/bench_hot_<tag>.json

Call sites = the real R22 (P3B / P3B_B) step: dY (dual row+col, SR, headroom 8), X (col only, nv4), weights (fp32 master, 2D scale), r17 mx8 X row (control / merge target),
xd = merged X kernel candidate (mx8 row + nv4 col; only candidates that have it).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
import torch

from common.bench import assert_gpu_exclusive, gpu_lock

dev = "cuda"
# per-step call sites of the real step: (model) -> list of (kind, R, C, count_per_step)   (counts from R22's nsys: 20 plain FP4 blocks (L, nofirst) / 9 (B) + 3 joint)
SITES = {
    "vitl16": {"plain": (7203, 20, [("dy", 4096, 1), ("dy", 3072, 1), ("dy", 1024, 2), ("x", 4096, 1), ("x", 1024, 3)]),
               "joint": (7903, 3, [("dy", 4096, 1), ("dy", 3072, 1), ("dy", 1024, 2), ("x", 4096, 1), ("x", 1024, 3)])},
    "vitb16": {"plain": (14406, 9, [("dy", 3072, 1), ("dy", 2304, 1), ("dy", 768, 2), ("x", 3072, 1), ("x", 768, 3)]),
               "joint": (15806, 3, [("dy", 3072, 1), ("dy", 2304, 1), ("dy", 768, 2), ("x", 3072, 1), ("x", 768, 3)])},
}
WEIGHTS = {"vitl16": [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)], "vitb16": [(2304, 768), (768, 768), (3072, 768), (768, 3072)]}
WCOUNT = {"vitl16": 23, "vitb16": 12}


def load_mod(name: str, defs: str = ""):
    """r21a -> round21's fp4q; base -> round24 fp4q2 default build; other -> fresh copy of fp4q2 bound to build variant `name` with nvcc defs."""
    if name == "r21a":
        from round21_fp4_quant_head.impl import fp4q as m
        return m
    spec = importlib.util.spec_from_file_location(f"fp4q2_{name}", ROUND / "impl" / ("fp4q2_srd.py" if name.startswith("srd") else "fp4q2.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    m._VARIANT = "" if name in ("base", "p7", "wr0") else name  # noqa: SLF001
    if name == "wr0":
        m.WEIGHT_ROW = False  # same build as base, weight quantizer without the (unused) row copy
    if name == "p7":
        m.PHILOX_ROUNDS = 7  # same build as base, Philox4x32-7 SR noise
    if name.startswith("srd"):
        m.SR_MODE = "exact" if name == "srd_exact" else "dither"  # srd_*: build of the scratch .cu (R24_SRC), srd = dither rule, srd_exact = the exact rule of the same build
    m._DEFS = defs.split()  # noqa: SLF001
    m._EXT = None  # noqa: SLF001
    return m


def capture(fn, reps, prod=None):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            if prod:
                prod()
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    keep = []
    with torch.cuda.graph(g):
        for _ in range(reps):
            if prod:
                prod()
            keep.append(fn())
    g._keep = keep  # noqa: SLF001
    return g


def interleave(graphs, rounds=5, reps=10):
    times = {n: [] for n in graphs}
    order = list(graphs)
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for r in range(rounds):
        for n in order if r % 2 == 0 else order[::-1]:
            g = graphs[n]
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            ev[0].record()
            for _ in range(reps):
                g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            times[n].append(ev[0].elapsed_time(ev[1]) * 1000.0 / reps)
    return {n: statistics.median(t) for n, t in times.items()}


def make_ops(mods, mx8, kind, x, states):
    """name -> callable() for one call-site kind on tensor x (all candidates)."""
    ops = {}
    for name, m in mods.items():
        st = states[name]
        if kind == "dy":
            hr = {"headroom": 8.0} if "headroom" in m.quant_act.__code__.co_varnames else {}
            ops[name] = lambda m=m, st=st, hr=hr: m.quant_act(x, "nv4", row=True, col=True, sr_row=True, sr_col=True, sr_offset=3, state=st, site=0, **hr)
        elif kind == "x":
            ops[name] = lambda m=m, st=st: m.quant_act(x, "nv4", row=False, col=True, state=st, site=1)
        elif kind == "dual":
            ops[name] = lambda m=m, st=st: m.quant_act(x, "nv4", row=True, col=True, state=st, site=0)
        elif kind == "row":
            ops[name] = lambda m=m, st=st: m.quant_act(x, "nv4", row=True, col=False, state=st, site=0)
        elif kind == "dysr_row":
            ops[name] = lambda m=m, st=st: m.quant_act(x, "nv4", row=True, col=False, sr_row=True, sr_offset=3, state=st, site=0)
        elif kind == "dysr_col":
            ops[name] = lambda m=m, st=st: m.quant_act(x, "nv4", row=False, col=True, sr_col=True, sr_offset=3, state=st, site=0)
        elif kind == "xd" and hasattr(m, "quant_x_dual"):
            ops[name] = lambda m=m, st=st: m.quant_x_dual(x, state=st, site=1)
    if kind == "x8":
        ops["r17_mx8_row"] = lambda: mx8.quantize(x, row=True, col=False)
    if kind == "xsum":  # control: r17 mx8 row + (each candidate) nv4 col, two passes
        for name, m in mods.items():
            st = states[name]
            ops[name + "+mx8row"] = lambda m=m, st=st: (mx8.quantize(x, row=True, col=False), m.quant_act(x, "nv4", row=False, col=True, state=st, site=1))
    return ops


def bench_cold(mods, mx8, states, kind, R, C, rounds):
    """cold input: the quantizer is replayed over ceil(4.2 * L2 / bytes) distinct buffers (DRAM resident), us per launch (no producer kernel)."""
    L2 = 96 * 2 ** 20
    nbuf = max(2, int(-(-4.2 * L2 // (R * C * 2 * 1.4))))
    xs = [torch.empty(R, C, device=dev, dtype=torch.bfloat16).uniform_(-3.0, 3.0)]
    xs += [xs[0].clone() for _ in range(nbuf - 1)]
    out = {}
    graphs = {}
    for n, m in mods.items():
        st = states[n]
        if kind == "dy":
            hr = {"headroom": 8.0} if "headroom" in m.quant_act.__code__.co_varnames else {}
            fns = lambda x, m=m, st=st, hr=hr: m.quant_act(x, "nv4", row=True, col=True, sr_row=True, sr_col=True, sr_offset=3, state=st, site=0, **hr)  # noqa: E731
        else:
            fns = lambda x, m=m, st=st: m.quant_act(x, "nv4", row=False, col=True, state=st, site=1)  # noqa: E731
        for _ in range(3):
            fns(xs[0])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        keep = []
        with torch.cuda.graph(g):
            for x in xs:
                keep.append(fns(x))
        g._keep = keep  # noqa: SLF001
        graphs[n] = g
    t = interleave(graphs, rounds=rounds)
    for n, v in t.items():
        out[n] = round(v / nbuf, 2)
    out["__nbuf"] = nbuf
    del xs, graphs
    torch.cuda.empty_cache()
    return out


def bench_seq(mods, mx8, states, R, C, reps, rounds):
    """real-step order of the X call site: producer kernel -> r17 mx8 row quantizer -> nv4 col quantizer; reports the col quantizer time (graph with / without it, same prefix)."""
    x = torch.empty(R, C, device=dev, dtype=torch.bfloat16).uniform_(-3.0, 3.0)

    def pre():
        x.uniform_(-3.0, 3.0)
        return mx8.quantize(x, row=True, col=False)

    graphs = {"__pre": capture(lambda: None, reps, prod=pre)}
    for n, m in mods.items():
        st = states[n]
        graphs[n] = capture(lambda m=m, st=st: m.quant_act(x, "nv4", row=False, col=True, state=st, site=1), reps, prod=pre)
    t = interleave(graphs, rounds=rounds)
    base = t.pop("__pre")
    out = {n: round((v - base) / reps, 2) for n, v in t.items()}
    out["__pre_us"] = round(base / reps, 2)
    del x, graphs
    torch.cuda.empty_cache()
    return out


def bench_site(mods, mx8, states, kind, R, C, reps, rounds):
    if kind == "xseq":
        return bench_seq(mods, mx8, states, R, C, reps, rounds)
    x = torch.empty(R, C, device=dev, dtype=torch.bfloat16).uniform_(-3.0, 3.0)
    prod = lambda: x.uniform_(-3.0, 3.0)  # noqa: E731
    ops = make_ops(mods, mx8, kind, x, states)
    if not ops:
        return {}
    graphs = {"__prod": capture(lambda: None, reps, prod=prod)}
    for n, fn in ops.items():
        graphs[n] = capture(fn, reps, prod=prod)
    t = interleave(graphs, rounds=rounds)
    base = t.pop("__prod")
    out = {n: round((v - base) / reps, 2) for n, v in t.items()}
    out["__prod_us"] = round(base / reps, 2)
    del x, graphs
    torch.cuda.empty_cache()
    return out


def bench_weight(mods, states, N, K, reps, rounds):
    ws = [torch.randn(N, K, device=dev, dtype=torch.float32) * 0.02]
    graphs = {}
    for name, m in mods.items():
        st = states[name]
        graphs[name] = capture(lambda m=m, st=st: m.quant_weight(ws[0], "nv4", amax="delayed", state=st, site=2), reps)
    t = interleave(graphs, rounds=rounds)
    return {n: round(v / reps, 2) for n, v in t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="r21a,base")
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--sites", default="dy,x")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--quick", action="store_true", help="plain shapes only, no joint")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tag", default="hot")
    ap.add_argument("--cold", action="store_true", help="cold-input mode for the dy / x sites (DRAM resident; no producer kernel)")
    a = ap.parse_args()
    outp = ROUND / "profile" / f"bench_hot_{a.tag}.json"
    mods, defs = {}, {}
    for c in a.cands.split(","):
        name, _, d = c.partition(":")
        mods[name] = load_mod(name, d)
        defs[name] = d
    from round17_mxfp8_sm120.impl import mx8

    with gpu_lock():
        for _ in range(60):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError as e:
                print("retry:", str(e)[:100], flush=True)
                time.sleep(20)
        states = {n: m.FP4State(8) for n, m in mods.items()}
        # init delayed sites eagerly (exact pass)
        out = {"round": "round24_fp4_quant_opt", "cands": defs, "reps": a.reps, "rounds": a.rounds, "sites": {}, "per_step_us": {}}
        sites = a.sites.split(",")
        for model in a.models.split(","):
            tot = {n: 0.0 for n in mods}
            res = out["sites"].setdefault(model, {})
            for grp, (R, cnt, lst) in SITES[model].items():
                if a.quick and grp == "joint":
                    continue
                seen = {}
                for kind, C, mult in lst:
                    for sk in sites:
                        if sk != kind and not (sk in ("x8", "xd", "xsum", "xseq", "dual", "row", "dysr_row", "dysr_col") and kind in ("x", "dy")):
                            continue
                        key = f"{sk}[{R},{C}]"
                        if key not in seen:
                            t0 = time.time()
                            seen[key] = bench_cold(mods, mx8, states, sk, R, C, a.rounds) if (a.cold and sk in ('dy', 'x')) else bench_site(mods, mx8, states, sk, R, C, a.reps, a.rounds)
                            res[key] = seen[key]
                            print(f"[{model}] {key} ({time.time() - t0:.0f}s): " + " ".join(f"{k}={v}" for k, v in seen[key].items()), flush=True)
                        if sk in ("dy", "x"):
                            for n in mods:
                                if n in seen[key]:
                                    tot[n] += cnt * mult * seen[key][n]
            if a.weights:
                for (N, K) in WEIGHTS[model]:
                    key = f"w[{N},{K}]"
                    res[key] = bench_weight(mods, states, N, K, a.reps, a.rounds)
                    print(f"[{model}] {key}: " + " ".join(f"{k}={v}" for k, v in res[key].items()), flush=True)
                    for n in mods:
                        tot[n] += WCOUNT[model] * res[key][n]
            out["per_step_us"][model] = {n: round(v, 1) for n, v in tot.items()}
            print(f"[{model}] per-step total (dy + x [+ w]) us: {out['per_step_us'][model]}", flush=True)
            outp.write_text(json.dumps(out, indent=1))
    print("DONE", outp)


if __name__ == "__main__":
    main()
