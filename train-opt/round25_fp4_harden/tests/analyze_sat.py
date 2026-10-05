"""Analysis of ``run.py --action sat_study`` reports (plain python + numpy, no GPU): per (stack, repeat) of each report (= one process = one pair of batches)

  * monitor counts (the device-side ``FP4State`` monitor) and the same counts recomputed from the per-step amax trace (consistency check), kind x headroom what-if (the trace of a run with
    headroom 8 is used to count how many comparisons WOULD have exceeded 2 / 4 / 8 / 16 / 32 / 64 times the previous amax: first-order, ignores the second-order effect of the clipping itself)
  * worst dY sites (block, Linear), event steps and parity (2 batches alternate: even / odd steps use batch 0 / 1), amax levels
  * P3 vs P3B at step 0 (identical weights / batch: the only difference is the head's FP4 gradient): per-site ratio of the dY amax
  * run-to-run: events of the two repeats of the same stack (same batches, same seed)

usage: python tests/analyze_sat.py profile/sat_study_*.json [--json out.json]
"""
import json
import statistics
import sys
from collections import defaultdict

import numpy as np

HS = (2, 4, 8, 16, 32, 64)


def load_run(r):
    tr = r["cand"]["trace"]
    init = np.array(tr["init"], dtype=np.float64)           # [S, 2]
    amax = np.array(tr["amax"], dtype=np.float64)           # [T, S, 2]
    raw = r["cand"]["raw"]
    return init, amax, raw


def events(init, amax, H):
    """bool [T, S, 2]: current amax > H * previous amax (previous of step 0 = ``init``), only where the previous amax is positive."""
    prev = np.concatenate([init[None], amax[:-1]], axis=0)
    return (amax > H * prev) & (prev > 0)


def kind_of(name):
    return name.rsplit(".", 1)[1]


def block_of(name):
    return int(name.split(".")[0][1:])


def lin_of(name):
    return name.split(".")[1]


def analyse_run(key, r):
    init, amax, raw = load_run(r)
    names = raw["site_names"]
    kinds = np.array([kind_of(n) for n in names])
    act = np.zeros(len(names), bool)
    act[raw["active_sites"]] = True
    mon = {c: np.array(raw[f"exceed_{c}"]) for c in ("row", "col")}
    hr = np.array(raw["headroom"])
    out = {"key": key, "stack": r["stack"], "idx": r["idx"], "scen": r["scen"]}
    # consistency: monitor (headroom from the state) vs trace-based
    tb = np.zeros((len(names), 2), int)
    for s in range(len(names)):
        ev = events(init[s:s + 1], amax[:, s:s + 1], hr[s])[:, 0, :]
        tb[s] = ev.sum(0)
    out["monitor_total"] = int(mon["row"].sum() + mon["col"].sum())
    out["trace_total"] = int(tb.sum())
    out["monitor_matches_trace"] = bool(np.array_equal(tb[:, 0], mon["row"]) and np.array_equal(tb[:, 1], mon["col"]))
    # kind x H what-if
    wi = {}
    for k in ("x", "dy", "w"):
        sel = (kinds == k) & act
        wi[k] = {str(H): int(events(init[sel], amax[:, sel], H).sum()) for H in HS}
        wi[k]["calls"] = int(sel.sum() * 2 * amax.shape[0])
    out["what_if"] = wi
    # dy sites: worst list (headroom 8 actual)
    sel = np.where((kinds == "dy") & act)[0]
    ev8 = {s: events(init[s:s + 1], amax[:, s:s + 1], 8.0)[:, 0, :] for s in sel}   # [T, 2]
    rows = []
    for s in sel:
        e = ev8[s]
        n = int(e.sum())
        if n:
            steps = sorted(set(np.where(e.any(1))[0].tolist()))
            prev = np.concatenate([init[None, s:s + 1], amax[:-1, s:s + 1]], 0)[:, 0, :]
            ratio = np.where(prev > 0, amax[:, s, :] / np.maximum(prev, 1e-30), 0.0)
            rows.append({"site": names[s], "events": n, "steps": steps, "max_ratio": float(ratio[e].max()), "amax_med": float(np.median(amax[:, s, 0])),
                         "amax_max": float(amax[:, s, 0].max()), "amax_min": float(amax[:, s, 0].min()),
                         "odd_over_even_med": float(np.median(amax[1::2, s, 0]) / max(np.median(amax[0::2, s, 0]), 1e-30))})
    rows.sort(key=lambda d: -d["events"])
    out["worst_dy"] = rows[:10]
    out["dy_event_steps_hist"] = np.bincount([t for r_ in rows for t in r_["steps"]], minlength=amax.shape[0]).tolist()
    by_block = defaultdict(int)
    for r_ in rows:
        by_block[block_of(r_["site"])] += r_["events"]
    out["dy_events_by_block"] = dict(sorted(by_block.items()))
    by_lin = defaultdict(int)
    for r_ in rows:
        by_lin[lin_of(r_["site"])] += r_["events"]
    out["dy_events_by_linear"] = dict(by_lin)
    return out, (init, amax, raw, ev8)


def step0_ratio(a, b, names, act):
    """per dY site ratio of the step-0 amax (row copy) of run ``a`` over run ``b``."""
    (_, am_a, _, _), (_, am_b, _, _) = a, b
    sel = np.where(np.array([kind_of(n) == "dy" for n in names]) & act)[0]
    ra = am_a[0, sel, 0] / np.maximum(am_b[0, sel, 0], 1e-30)
    late = np.array([block_of(names[s]) >= 20 for s in sel])
    return {"n": int(len(sel)), "median": float(np.median(ra)), "p5": float(np.percentile(ra, 5)), "p95": float(np.percentile(ra, 95)), "min": float(ra.min()), "max": float(ra.max()),
            "median_blocks_ge20": float(np.median(ra[late])) if late.any() else None}


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json") and not a.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    summary = {}
    for f in files:
        d = json.load(open(f))
        print(f"=== {f}  batch_sum {[round(x) for x in d['batch_sum']]}  seq {d['seq']}")
        res, data = {}, {}
        for key, r in d["runs"].items():
            res[key], data[key] = analyse_run(key, r)
        names = next(iter(data.values()))[2]["site_names"]
        act = np.zeros(len(names), bool)
        act[next(iter(data.values()))[2]["active_sites"]] = True
        print("run                      dy exceed/calls  x   w  monitor==trace | what-if dy events at headroom " + " ".join(f"{H:>4}" for H in HS))
        for key, o in res.items():
            wi = o["what_if"]
            print(f"{key:24s} {wi['dy']['8']:4d}/{wi['dy']['calls']}  {wi['x']['2']:3d} {wi['w']['2']:3d}  {str(o['monitor_matches_trace']):5s}          |   " + " ".join(f"{wi['dy'][str(H)]:>4}" for H in HS))
        # P3 vs P3B step 0
        keys = list(res)
        p3b = [k for k in keys if res[k]["stack"] == "P3B"]
        p3 = [k for k in keys if res[k]["stack"] == "P3"]
        cmp = {}
        if p3b and p3:
            cmp["p3_vs_p3b_step0_dy_amax"] = step0_ratio(data[p3[0]], data[p3b[0]], names, act)
            if len(p3) > 1:
                cmp["p3_rep_vs_p3_step0"] = step0_ratio(data[p3[1]], data[p3[0]], names, act)
            if len(p3b) > 1:
                cmp["p3b_rep_vs_p3b_step0"] = step0_ratio(data[p3b[1]], data[p3b[0]], names, act)
            print("step-0 dY amax ratio P3 / P3B   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in cmp["p3_vs_p3b_step0_dy_amax"].items()})
            if "p3_rep_vs_p3_step0" in cmp:
                print("step-0 dY amax ratio P3(rep)/P3 ", {k: round(v, 3) if isinstance(v, float) else v for k, v in cmp["p3_rep_vs_p3_step0"].items()})
        # run-to-run overlap of (site, step) events for the same stack
        ovl = {}
        for st, ks in (("P3", p3), ("P3B", p3b)):
            if len(ks) >= 2:
                e0 = data[ks[0]][3]
                e1 = data[ks[1]][3]
                s0 = {(s, t) for s, e in e0.items() for t in np.where(e.any(1))[0]}
                s1 = {(s, t) for s, e in e1.items() for t in np.where(e.any(1))[0]}
                ovl[st] = {"n_rep0": len(s0), "n_rep1": len(s1), "common": len(s0 & s1)}
        if p3 and p3b:
            e0, e1 = data[p3[0]][3], data[p3b[0]][3]
            s0 = {(s, t) for s, e in e0.items() for t in np.where(e.any(1))[0]}
            s1 = {(s, t) for s, e in e1.items() for t in np.where(e.any(1))[0]}
            ovl["P3_vs_P3B"] = {"n_P3": len(s0), "n_P3B": len(s1), "common": len(s0 & s1)}
            steps0 = sorted({t for _, t in s0})
            steps1 = sorted({t for _, t in s1})
            ovl["event_steps_P3"] = [int(x) for x in steps0]
            ovl["event_steps_P3B"] = [int(x) for x in steps1]
        if ovl:
            print("(site,step) event overlap", ovl)
        for key in (p3[:1] + p3b[:1]):
            o = res[key]
            print(f"-- {key}: by block {o['dy_events_by_block']}  by linear {o['dy_events_by_linear']}")
            print("   event-step histogram (steps 0..49):", "".join(str(min(9, v)) for v in o["dy_event_steps_hist"]))
            for w in o["worst_dy"][:6]:
                print(f"   {w['site']:14s} ev {w['events']:2d} maxratio {w['max_ratio']:7.2f} steps {w['steps'][:8]}  amax med {w['amax_med']:.3g} [{w['amax_min']:.3g}, {w['amax_max']:.3g}] odd/even {w['odd_over_even_med']:.2f}")
        summary[f] = {"batch_sum": d["batch_sum"], "runs": res, "cmp": cmp, "overlap": ovl}
    if out_json:
        json.dump(summary, open(out_json, "w"), indent=1)
        print("wrote", out_json)


if __name__ == "__main__":
    main()
