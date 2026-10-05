"""Greedy plan search for one (model, preset) from the measured sweeps.  GPU job (dual lock), adaptive: every trial is a full 126-frame emulation (gate M on the 8 eval frames, gate H on the 110 others).

  python finalize.py --model vitl16 --preset safe|fast [--min-saving-ms 0.15] [--out sweeps/FIN_vitl16_safe.jsonl]

Options = sweep rows (sweeps/*.jsonl, any grid) that individually passed the preset's gate (safe: M and H, fast: M).  Each option is a bundle of (kinds, layers, Spec).
Speed value of an option = measured per-frame saving (speed_model.Speed: GEMM tier time incl. separately-needed epilogue + quantisation, central estimate) -- options without a
measured kernel for their format (e.g. block-scaled int8) are skipped.  Greedy: options in decreasing saving; an option is applied only to the (layer, kind) units not yet claimed by a
better option; it is kept if the COMBINED plan still passes the gate (and keeps >= min-saving).  Finally head kinds are tried the same way.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
import emu_eval as ee  # noqa: E402
from emu import KINDS, HEAD_KINDS, make_plan, layers_of  # noqa: E402
from speed_model import Speed  # noqa: E402

TIER_OF = {"fp16": "fp16", "int8": "int8", "mx8": "mx8", "nv4": "nv4", "mx4": "mx4"}


def load_rows(model):
    rows = []
    for f in sorted(glob.glob(str(HERE / "sweeps" / f"*_{model}.jsonl"))):
        if "/FIN_" in f:
            continue
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "error" not in r and r.get("mode") == "full":
                    rows.append(r)
    return rows


def units_of(entries, nb):
    """{(layer, kind): spec dict} for backbone entries; {('head', kind): spec} for head."""
    u = {}
    for e in entries:
        for k in e["kinds"]:
            if k in HEAD_KINDS:
                u[("head", k)] = e["spec"]
            else:
                for i in layers_of(e.get("layers", "all"), nb):
                    u[(i, k)] = e["spec"]
    return u


def bundle_saving(sp, model, units, mode="central"):
    """ms / frame saved by `units` ({(layer, kind): spec}) vs r5_v1, from the measured speed model; None if some tier has no measurement."""
    tot = 0.0
    for (l, k), spec in units.items():
        if l == "head":
            continue
        fmt = spec["fmt"]
        if fmt not in TIER_OF:
            return None
        s = sp.saving_ms(model, k, TIER_OF[fmt], fmt, [l], mode, bool(spec.get("pfx")))
        if s is None:
            return None
        tot += s
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", required=True, choices=["safe", "fast"])
    ap.add_argument("--min-saving-ms", type=float, default=0.0)
    ap.add_argument("--time-limit", type=float, default=520)
    ap.add_argument("--max-trials", type=int, default=60)
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-iou", type=float, default=0.0, help="extra selection margin: pooled IoU (M and, for safe, H) of every option / accepted plan must be >= this")
    ap.add_argument("--max-conf", type=float, default=1.0, help="extra selection margin: max per-frame conf rel diff (M and, for safe, H) must be <= this")
    a = ap.parse_args()
    out = Path(a.out or HERE / "sweeps" / f"FIN_{a.model}_{a.preset}{a.tag}.jsonl")
    sp = Speed()
    h = ee.Harness(a.model)
    nb = h.net.nb
    need_cal = False
    rows = load_rows(a.model)

    def gate_ok(res):
        sets = [res["M"]] + ([res["H"]] if a.preset == "safe" else [])
        return all(x["pass"] and x["pooled_iou"] >= a.min_iou and x["conf_max"] <= a.max_conf for x in sets)

    def ok(r):
        return gate_ok(r)
    opts = []
    for r in rows:
        if not ok(r) or not r["entries"]:
            continue
        if any(e["spec"].get("gptq") or e["spec"].get("sq") for e in r["entries"]):
            need_cal = True
        u = units_of(r["entries"], nb)
        bb = {k: v for k, v in u.items() if k[0] != "head"}
        if not bb:
            continue
        s = bundle_saving(sp, a.model, bb)
        if s is None or s < a.min_saving_ms:
            continue
        opts.append((s, r["name"], r["entries"], u))
    mae_of = {}
    for r in rows:
        mae_of[r["name"]] = r["H"]["logit_mae"]
    opts.sort(key=lambda x: (-round(x[0] / 0.02), mae_of.get(x[1], 1.0)))      # saving first (0.02 ms buckets), ties -> lowest H logit MAE
    print(f"{len(opts)} passing backbone options with a measured speed value; best: {[(round(o[0], 3), o[1]) for o in opts[:5]]}", flush=True)
    if need_cal:
        h.calibrate(hess_kinds=("qkv", "proj", "fc1", "fc2"))
        import gptq
        gptq.install(h.net)
    claimed = {}
    cur_entries = []
    best = None
    trials = 0
    t0 = time.time()
    log = []

    def evaluate(entries):
        plan = make_plan(entries, nb)
        return h.eval_plan(plan, "full")

    base = evaluate([])
    print("P0:", json.dumps({"M": base["M"], "H": base["H"]})[:300], flush=True)
    for s, name, entries, u in opts:
        if trials >= a.max_trials or time.time() - t0 > a.time_limit:
            break
        new = {k: v for k, v in u.items() if k not in claimed and k[0] != "head"}
        if not new:
            continue
        # entries restricted to the new units
        add = []
        for (l, k), spec in new.items():
            add.append({"kinds": [k], "layers": [l], "spec": spec})
        trial_entries = cur_entries + add
        res = evaluate(trial_entries)
        trials += 1
        passed = gate_ok(res)
        sv = bundle_saving(sp, a.model, {**claimed, **new})
        rec = {"name": f"FIN|{a.preset}|{trials}|{name}", "model": a.model, "mode": "full", "entries": trial_entries if len(trial_entries) < 400 else "(many)", "accepted": bool(passed),
               "saving_ms_if_accepted": sv, "option_saving_ms": s, **{k: v for k, v in res.items() if k != "per_frame"}, "per_frame": res["per_frame"]}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{trials}] {name:50s} +{len(new)} units  sav {s:.3f} ms  M {res['M']['pooled_iou']:.4f}/{res['M']['conf_max'] * 100:.2f}% H {res['H']['pooled_iou']:.4f}/{res['H']['conf_max'] * 100:.2f}% >3%:{res['H']['n_conf_gt3']} -> {'ACCEPT' if passed else 'reject'}", flush=True)
        if passed:
            claimed.update(new)
            cur_entries = trial_entries
            best = (sv, res)
    # head kinds: each head GEMM (ct1, ct2) independently, fastest passing format first (mx8, then fp16); mask einsum is only gate-checked (no speed value: both operands are activations)
    for hk in ("ct2", "ct1"):
        for fmt in ("mx8", "fp16"):
            if time.time() - t0 > a.time_limit + 120:
                break
            trial_entries = cur_entries + [{"kinds": [hk], "layers": "all", "spec": {"fmt": fmt}}]
            res = evaluate(trial_entries)
            passed = gate_ok(res)
            rec = {"name": f"FIN|{a.preset}|head|{hk}|{fmt}", "model": a.model, "mode": "full", "entries": "(see plan)", "accepted": bool(passed), **{k: v for k, v in res.items() if k != "per_frame"}}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[head] {hk} {fmt:5s} M {res['M']['pooled_iou']:.4f}/{res['M']['conf_max'] * 100:.2f}% H {res['H']['pooled_iou']:.4f}/{res['H']['conf_max'] * 100:.2f}% -> {'ACCEPT' if passed else 'reject'}", flush=True)
            if passed:
                cur_entries = trial_entries
                break
    final = evaluate(cur_entries)
    robust = None
    if need_cal and any(e["spec"].get("sq") or e["spec"].get("gptq") for e in cur_entries):
        # robustness: the same plan with scales / Hessians from a second, shifted calibration set
        h.calibrate(hess_kinds=("qkv", "proj", "fc1", "fc2"), idxs=ee.CAL_ALT)
        robust = evaluate(cur_entries)
        robust = {k: v for k, v in robust.items() if k != "per_frame"}
        print("robustness (alt calibration set):", json.dumps({"M": robust["M"], "H": robust["H"]})[:400], flush=True)
    plan_out = {"model": a.model, "preset": a.preset, "tag": a.tag, "min_iou": a.min_iou, "max_conf": a.max_conf, "entries": cur_entries, "M": final["M"], "H": final["H"], "all126": final["all126"], "saving_ms_central": bundle_saving(sp, a.model, claimed),
                "saving_ms_pess": bundle_saving(sp, a.model, claimed, "pess"), "robustness_alt_calibration": robust}
    (HERE / "profile").mkdir(exist_ok=True)
    (HERE / "profile" / f"plan_{a.model}_{a.preset}{a.tag}.json").write_text(json.dumps(plan_out, indent=1))
    print("FINAL:", json.dumps({k: plan_out[k] for k in ("M", "H", "saving_ms_central", "saving_ms_pess")})[:600], flush=True)


if __name__ == "__main__":
    main()
