"""R21a micro-benchmark of the fused FP4 quantizers (CUDA-graph replay, candidates interleaved in one process, median of 3 rounds, dual GPU lock).

  python tests/bench_fp4q.py [--which act|weight|all] [--models vitl16,vitb16] [--quick] [--tag NAME]     -> profile/bench_fp4q<_tag>.json (merged)

"cold" numbers rotate over enough distinct input buffers that the working set exceeds the 96 MB L2 (the DRAM-floor comparison); "warm" numbers re-quantize the same input
(L2 resident for small tensors, like a producer -> quantizer hand-off in the real step).  Floor = (bytes read + written) / measured device-to-device copy bandwidth.
Controls: r17 ``mx8.quantize`` (quant_dual: MXFP8 row + col copy), R20a ``profile/fp4_quant_cost.json`` (Spark / QuTLASS / torch transpose, listed in the NOTES).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4q
from common.bench import assert_gpu_exclusive, gpu_lock

dev = "cuda"
L2_BYTES = 96 * 2 ** 20
# (name, R, C, class)  class: X = input activation of a Linear (wgrad needs its transposed copy), dY = gradient (row copy for dgrad + transposed copy for wgrad)
SHAPES = {
    "vitl16": [("X[7203,1024]", 7203, 1024, "X"), ("h[7203,4096]", 7203, 4096, "X"), ("dY[7203,3072]", 7203, 3072, "dY"), ("dY[7203,1024]", 7203, 1024, "dY"), ("dY[7203,4096]", 7203, 4096, "dY"),
               ("X[7903,1024]", 7903, 1024, "X"), ("h[7903,4096]", 7903, 4096, "X"), ("dY[7903,3072]", 7903, 3072, "dY"), ("dY[7903,1024]", 7903, 1024, "dY"), ("dY[7903,4096]", 7903, 4096, "dY"),
               ("dZ[28672,4096]", 28672, 4096, "dY"), ("dZ[7168,4096]", 7168, 4096, "dY"), ("convT_in[28672,1024]", 28672, 1024, "X"), ("convT_in[7168,1024]", 7168, 1024, "X")],
    "vitb16": [("X[14406,768]", 14406, 768, "X"), ("h[14406,3072]", 14406, 3072, "X"), ("dY[14406,2304]", 14406, 2304, "dY"), ("dY[14406,768]", 14406, 768, "dY"), ("dY[14406,3072]", 14406, 3072, "dY"),
               ("X[15806,768]", 15806, 768, "X"), ("h[15806,3072]", 15806, 3072, "X"), ("dY[15806,2304]", 15806, 2304, "dY"), ("dY[15806,768]", 15806, 768, "dY"), ("dY[15806,3072]", 15806, 3072, "dY"),
               ("dZ[57344,3072]", 57344, 3072, "dY"), ("dZ[14336,3072]", 14336, 3072, "dY"), ("convT_in[57344,768]", 57344, 768, "X"), ("convT_in[14336,768]", 14336, 768, "X")],
}
PREFIX = {7203: (7, 1029, 5), 7903: (7, 1129, 105), 14406: (14, 1029, 5), 15806: (14, 1129, 105)}
WEIGHTS = {"vitl16": [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)], "vitb16": [(2304, 768), (768, 768), (3072, 768), (768, 3072)]}
HEAD_W = {"vitl16": (4096, 1024), "vitb16": (3072, 768)}


def pad128(n):
    return -(-n // 128) * 128


def capture(fns, xs, inner_reps=1):
    """CUDA graph of one launch of ``fn`` per input buffer (outputs kept alive so that no buffer is reused inside the graph)."""
    fn = fns
    for _ in range(3):
        fn(xs[0])
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(xs[0])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    keep = []
    with torch.cuda.graph(g):
        for _ in range(inner_reps):
            for x in xs:
                keep.append(fn(x))
    g._keep = keep  # noqa: SLF001
    return g, len(xs) * inner_reps


def interleave(cands, rounds=3, reps=10):
    """cands: {name: (graph, launches_per_replay)} -> median us per launch."""
    times = {n: [] for n in cands}
    order = list(cands)
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for r in range(rounds):
        for n in order if r % 2 == 0 else order[::-1]:
            g, k = cands[n]
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            ev[0].record()
            for _ in range(reps):
                g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            times[n].append(ev[0].elapsed_time(ev[1]) * 1000.0 / (reps * k))
    return {n: statistics.median(t) for n, t in times.items()}


def copy_bandwidth():
    big = torch.empty(512 * 2 ** 20 // 2, device=dev, dtype=torch.bfloat16)
    big2 = torch.empty_like(big)
    g, k = capture(lambda x: big2.copy_(x), [big], inner_reps=4)
    us = interleave({"copy": (g, k)})["copy"]
    return 2 * big.numel() * 2 / us / 1e3  # GB/s (read + write)


def act_bytes(R, C, fmt, row, col, prefix, in_bytes=2):
    v = 16 if fmt == "nv4" else 32
    per = 0.5 + 1.0 / v
    Kt = pad128(R) if prefix is None else prefix[0] * (prefix[1] - prefix[2])
    b = R * C * in_bytes
    if row:
        b += R * C * per
    if col:
        b += Kt * C * per
    if prefix is not None:
        b += prefix[0] * prefix[2] * C * 2  # bf16 xp
    return b


def act_cands(R, C, cls, state, which):
    """mode name -> (callable(x), (fmt, row, col, prefix))"""
    pre = PREFIX.get(R)
    seed = state.seed
    d = {}
    for fmt in ("nv4", "mx4"):
        kw = dict(amax="delayed", state=state, site=0) if fmt == "nv4" else {}
        d[f"{fmt}_row"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, row=True, col=False, **kw), (fmt, True, False, None))
        d[f"{fmt}_col"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, row=False, col=True, **kw), (fmt, False, True, None))
        d[f"{fmt}_dual"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, row=True, col=True, **kw), (fmt, True, True, None))
        d[f"{fmt}_dual_rhtcol"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, rht_col=True, **kw), (fmt, True, True, None))
        d[f"{fmt}_dual_sr"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, sr_row=True, sr_col=True, seed=seed, sr_offset=1, **kw), (fmt, True, True, None))
        d[f"{fmt}_dual_rhtcol_sr"] = (lambda x, fmt=fmt, kw=kw: fp4q.quant_act(x, fmt, rht_col=True, sr_row=True, sr_col=True, seed=seed, sr_offset=1, **kw), (fmt, True, True, None))
        if pre is not None:
            d[f"{fmt}_col_prefix_rht"] = (lambda x, fmt=fmt, kw=kw, pre=pre: fp4q.quant_act(x, fmt, row=False, col=True, rht_col=True, prefix=pre, **kw), (fmt, False, True, pre))
            d[f"{fmt}_dual_prefix"] = (lambda x, fmt=fmt, kw=kw, pre=pre: fp4q.quant_act(x, fmt, prefix=pre, **kw), (fmt, True, True, pre))
            d[f"{fmt}_full_prefix_rhtcol_sr"] = (lambda x, fmt=fmt, kw=kw, pre=pre: fp4q.quant_act(x, fmt, rht_col=True, sr_row=True, sr_col=True, seed=seed, sr_offset=1, prefix=pre, **kw),
                                                 (fmt, True, True, pre))
    d["nv4_dual_exact"] = (lambda x: fp4q.quant_act(x, "nv4", amax="exact"), ("nv4", True, True, None))
    return d


def bench_act(models, quick, bw, out):
    from round17_mxfp8_sm120.impl import mx8

    state = fp4q.FP4State(4)
    for model in models:
        res = out.setdefault("act", {}).setdefault(model, {})
        shapes = SHAPES[model]
        if quick:
            shapes = [s for s in shapes if s[0] in ("X[7203,1024]", "dY[7203,4096]", "dZ[28672,4096]", "dY[7203,1024]", "X[14406,768]")]
        for name, R, C, cls in shapes:
            t0 = time.time()
            foot = R * C * 2 * 1.4
            n_buf = max(2, int(-(-4.2 * L2_BYTES // foot)))
            xs = [torch.randn(R, C, device=dev, dtype=torch.bfloat16)]
            for _ in range(n_buf - 1):
                xs.append(xs[0].clone())
            cands = act_cands(R, C, cls, state, "all")
            for k in ("nv4_row", "nv4_col", "nv4_dual", "nv4_dual_rhtcol_sr", "nv4_col_prefix_rht", "nv4_dual_prefix", "nv4_full_prefix_rhtcol_sr", "mx4_dual", "mx4_dual_rhtcol_sr", "nv4_dual_exact"):
                pass
            sel = {k: v for k, v in cands.items() if k in ("nv4_row", "nv4_col", "nv4_dual", "nv4_dual_rhtcol", "nv4_dual_sr", "nv4_dual_rhtcol_sr", "nv4_col_prefix_rht", "nv4_dual_prefix", "nv4_full_prefix_rhtcol_sr", "nv4_dual_exact",
                                                           "mx4_dual", "mx4_dual_rhtcol_sr")}
            built, errs = {}, {}
            for k, (fn, meta) in sel.items():
                try:
                    built[k] = capture(fn, xs)
                except Exception as e:  # noqa: BLE001
                    errs[k] = f"{type(e).__name__}: {str(e)[:200]}"
            # r17 controls (C % 128 == 0 only)
            if C % 128 == 0:
                built["r17_mx8_dual"] = capture(lambda x: mx8.quantize(x, row=True, col=True), xs)
                built["r17_mx8_row"] = capture(lambda x: mx8.quantize(x, row=True, col=False), xs)
                sel["r17_mx8_dual"] = (None, ("mx8", True, True, None))
                sel["r17_mx8_row"] = (None, ("mx8", True, False, None))
            built["torch_x.t().contiguous()"] = capture(lambda x: x.t().contiguous(), xs)
            cold = interleave(built)
            # warm: the same input (L2-resident for small tensors)
            warm_built = {k: capture(sel[k][0], [xs[0]], inner_reps=5) for k in ("nv4_dual", "nv4_dual_rhtcol_sr") if k in sel and sel[k][0] is not None}
            if C % 128 == 0:
                warm_built["r17_mx8_dual"] = capture(lambda x: mx8.quantize(x, row=True, col=True), [xs[0]], inner_reps=5)
            warm = interleave(warm_built)
            rec = {"R": R, "C": C, "class": cls, "n_cold_buffers": n_buf, "cold_us": {}, "warm_us": warm, "errors": errs}
            for k, us in cold.items():
                meta = sel.get(k, (None, None))[1]
                if meta is None:
                    rec["cold_us"][k] = {"us": us}
                    continue
                fmt, row, col, pre = meta
                if fmt == "mx8":
                    nb = R * C * 2 + (R * C * (1 + 1 / 32) if row else 0) + (pad128(R) * C * (1 + 1 / 32) if col else 0)
                else:
                    nb = act_bytes(R, C, fmt, row, col, pre)
                floor = nb / (bw * 1e3)
                rec["cold_us"][k] = {"us": round(us, 2), "bytes_MB": round(nb / 1e6, 2), "floor_us": round(floor, 2), "x_floor": round(us / floor, 3)}
            res[name] = rec
            print(f"[act {model}] {name} ({time.time() - t0:.0f}s) cold: " + " ".join(f"{k}={v['us'] if isinstance(v, dict) else v}" for k, v in {k: rec['cold_us'][k] for k in rec['cold_us']}.items() if k in ("nv4_row", "nv4_col", "nv4_dual", "nv4_dual_rhtcol", "nv4_dual_sr", "nv4_dual_rhtcol_sr", "nv4_col_prefix_rht", "nv4_full_prefix_rhtcol_sr", "mx4_dual", "r17_mx8_dual")) + " | warm " + " ".join(f"{k}={v:.1f}" for k, v in warm.items()), flush=True)
            del xs, built, warm_built
            torch.cuda.empty_cache()


def bench_weight(models, quick, bw, out):
    from round17_mxfp8_sm120.impl import mx8

    state = fp4q.FP4State(2)
    for model in models:
        res = out.setdefault("weight", {}).setdefault(model, {})
        shapes = WEIGHTS[model] + [HEAD_W[model]]
        if quick:
            shapes = shapes[:1]
        for (N, K) in shapes:
            n_buf = max(2, int(-(-4.2 * L2_BYTES // (N * K * 4 * 1.2))))
            ws = [torch.randn(N, K, device=dev, dtype=torch.float32) * 0.02 for _ in range(1)]
            ws += [ws[0].clone() for _ in range(n_buf - 1)]
            built = {}
            built["nv4_w_delayed"] = capture(lambda w: fp4q.quant_weight(w, "nv4", amax="delayed", state=state, site=0), ws)
            built["mx4_w"] = capture(lambda w: fp4q.quant_weight(w, "mx4"), ws)
            built["nv4_w_exact"] = capture(lambda w: fp4q.quant_weight(w, "nv4", amax="exact"), ws)
            built["r17_mx8_dual(fp32 w)"] = capture(lambda w: mx8.quantize(w, row=True, col=True), ws)
            cold = interleave(built)
            nbytes = N * K * 4 + 2 * N * K * (0.5 + 1 / 16)
            nbytes_mx = N * K * 4 + 2 * N * K * (1 + 1 / 32)
            rec = {}
            for k, us in cold.items():
                nb = nbytes_mx if k.startswith("r17") else nbytes
                floor = nb / (bw * 1e3)
                rec[k] = {"us": round(us, 2), "floor_us": round(floor, 2), "x_floor": round(us / floor, 3)}
            res[f"w[{N},{K}]"] = {"N": N, "K": K, "cold": rec}
            print(f"[weight {model}] w[{N},{K}] " + " ".join(f"{k}={v['us']}({v['x_floor']}x)" for k, v in rec.items()), flush=True)
            del ws, built
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    outp = ROUND / "profile" / f"bench_fp4q{'_' + a.tag if a.tag else ''}.json"
    models = a.models.split(",")
    with gpu_lock():
        for _ in range(30):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError as e:
                print("retry:", str(e)[:100], flush=True)
                time.sleep(20)
        out = json.loads(outp.read_text()) if outp.exists() and not a.quick else {}
        out["round"] = "round21_fp4_quant_head"
        bw = copy_bandwidth()
        out["hbm_copy_GBps"] = round(bw, 1)
        print(f"copy bandwidth {bw:.0f} GB/s (read+write)", flush=True)
        if a.which in ("act", "all"):
            bench_act(models, a.quick, bw, out)
        if a.which in ("weight", "all"):
            bench_weight(models, a.quick, bw, out)
        if not a.quick:
            outp.write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
