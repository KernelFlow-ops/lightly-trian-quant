"""FP16-tier overflow scan (PLAN_v2 R10 step 4): for every layer and GEMM kind, on real activations of the P0 model,

  * max |partial sum| of every k-chunk (chunk = 16 / 32 / 64: one MMA / the BK=32 k-tile of the prototype / the spec's 64) and max |output|,
  * the abs-sum bound  max_{m,n} sum_{k in chunk} |x_mk||w_nk|  (an upper bound on every running fp16 partial sum of that chunk),
  * operand extremes (|X|, |W|), split into prefix rows (cls + registers, 105 rows in joint layers) and the other rows,
  * the power-of-two pre-scale p (applied to W, undone by `oscale` in the epilogue) that brings the bound below 2^14 (= 16384, a 4x margin to 65504),
    for "all rows" and for "prefix rows excluded" (which then run on the bf16 path).

  python scan_overflow.py --model vitl16 [--frames 4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
import emu_eval as ee  # noqa: E402
from emu import KINDS  # noqa: E402

FP16_MAX = 65504.0


def pneed(bound: float) -> int:
    """smallest p >= 0 with bound * 2^-p <= 2^14."""
    return max(0, math.ceil(math.log2(max(bound, 1e-30) / 16384.0)))


@torch.inference_mode()
def scan_one(Xf: torch.Tensor, Wf: torch.Tensor, pm: torch.Tensor, chunks=(16, 32, 64)) -> dict:
    """Xf [M,K] fp32, Wf [N,K] fp32, pm [M] bool prefix rows."""
    M, K = Xf.shape
    N = Wf.shape[0]
    out = {}
    full = Xf @ Wf.t()
    out["out_max_nonprefix"] = full[~pm].abs().max().item()
    out["out_max_prefix"] = full[pm].abs().max().item()
    out["x_max_nonprefix"] = Xf[~pm].abs().max().item()
    out["x_max_prefix"] = Xf[pm].abs().max().item()
    out["w_max"] = Wf.abs().max().item()
    del full
    Xa, Wa = Xf.abs(), Wf.abs()
    for c in chunks:
        nc = K // c
        pmax = {"np": 0.0, "pfx": 0.0}
        amax = {"np": 0.0, "pfx": 0.0}
        step = max(1, int(2.0e8 // (M * N)))     # chunks per bmm slice (~0.8 GB fp32)
        for s0 in range(0, nc, step):
            s1 = min(nc, s0 + step)
            Xc = Xf[:, s0 * c:s1 * c].reshape(M, s1 - s0, c).permute(1, 0, 2)
            Wc = Wf[:, s0 * c:s1 * c].reshape(N, s1 - s0, c).permute(1, 2, 0)
            P = torch.bmm(Xc, Wc).abs()
            pmax["np"] = max(pmax["np"], P[:, ~pm].max().item())
            pmax["pfx"] = max(pmax["pfx"], P[:, pm].max().item())
            Xc = Xa[:, s0 * c:s1 * c].reshape(M, s1 - s0, c).permute(1, 0, 2)
            Wc = Wa[:, s0 * c:s1 * c].reshape(N, s1 - s0, c).permute(1, 2, 0)
            P = torch.bmm(Xc, Wc)
            amax["np"] = max(amax["np"], P[:, ~pm].max().item())
            amax["pfx"] = max(amax["pfx"], P[:, pm].max().item())
            del P
        out[f"c{c}"] = {"partial_max_nonprefix": pmax["np"], "partial_max_prefix": pmax["pfx"],
                        "absbound_nonprefix": amax["np"], "absbound_prefix": amax["pfx"],
                        "p_all": pneed(max(amax["np"], amax["pfx"])), "p_excl_prefix": pneed(amax["np"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    h = ee.Harness(a.model, need_refs=False)
    net = h.net
    net.set_plan({})
    res = {}

    def hook(i, kind, X):
        B, N, K = X.shape
        pm = net.prefix_mask(B, N, X.device)
        Xf = X.reshape(B * N, K).float()
        Wf = net._orig_w(i, kind).float()
        r = scan_one(Xf, Wf, pm)
        key = f"{i}|{kind}"
        if key in res:      # keep the worst over frames
            for k, v in r.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        res[key][k][kk] = max(res[key][k][kk], vv)
                else:
                    res[key][k] = max(res[key][k], v)
        else:
            res[key] = r

    net.collect = hook
    frames = ee.CAL_IDX[: a.frames]
    t0 = time.time()
    with torch.inference_mode():
        for idx in frames:
            b, meta = h.prep(h.frames[idx])
            net.forward(b)
            print("scanned frame", idx, f"{time.time() - t0:.0f}s", flush=True)
    net.collect = None
    # summary
    nb = net.nb
    n_all = len(res)
    summ = {"chunk32": {}, "chunk64": {}}
    for ck, tag in (("c32", "chunk32"), ("c64", "chunk64")):
        over_all = [k for k, v in res.items() if max(v[ck]["absbound_nonprefix"], v[ck]["absbound_prefix"]) > FP16_MAX]
        over_np = [k for k, v in res.items() if v[ck]["absbound_nonprefix"] > FP16_MAX]
        pmax_over_all = [k for k, v in res.items() if max(v[ck]["partial_max_nonprefix"], v[ck]["partial_max_prefix"]) > FP16_MAX]
        pmax_over_np = [k for k, v in res.items() if v[ck]["partial_max_nonprefix"] > FP16_MAX]
        summ[tag] = {"n_gemms": n_all, "absbound_overflow_all_rows": len(over_all), "absbound_overflow_nonprefix_rows": len(over_np),
                     "partial_sum_overflow_all_rows": len(pmax_over_all), "partial_sum_overflow_nonprefix_rows": len(pmax_over_np),
                     "p_all_hist": _hist([v[ck]["p_all"] for v in res.values()]), "p_excl_prefix_hist": _hist([v[ck]["p_excl_prefix"] for v in res.values()]),
                     "only_prefix_offenders": [k for k in over_all if k not in over_np],
                     "max_p_all": max(v[ck]["p_all"] for v in res.values()), "max_p_excl_prefix": max(v[ck]["p_excl_prefix"] for v in res.values())}
    out = {"model": a.model, "frames": frames, "summary": summ, "per_gemm": res}
    Path(HERE / "profile").mkdir(exist_ok=True)
    (HERE / "profile" / f"overflow_scan_{a.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))
    worst = sorted(res.items(), key=lambda kv: -max(kv[1]["c32"]["absbound_nonprefix"], kv[1]["c32"]["absbound_prefix"]))[:8]
    for k, v in worst:
        c = v["c32"]
        print(f"  {k:10s} absbound np={c['absbound_nonprefix']:.3g} pfx={c['absbound_prefix']:.3g}  partial np={c['partial_max_nonprefix']:.3g} pfx={c['partial_max_prefix']:.3g}"
              f"  |x| np={v['x_max_nonprefix']:.3g} pfx={v['x_max_prefix']:.3g} |out| np={v['out_max_nonprefix']:.3g} pfx={v['out_max_prefix']:.3g}  p_all={c['p_all']} p_excl={c['p_excl_prefix']}")


def _hist(v):
    d = {}
    for x in v:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


if __name__ == "__main__":
    main()
