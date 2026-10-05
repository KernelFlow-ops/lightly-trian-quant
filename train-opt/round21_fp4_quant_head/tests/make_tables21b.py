"""R21b: markdown tables + results.json["R21b"] from the profile/*.json files (read-only on everything except results.json's new "R21b" key).

  python tests/make_tables21b.py [--write-results]      prints the tables; --write-results merges the R21b block into results.json (R21a keys untouched)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
P = ROUND / "profile"
MODELS = ("vitl16", "vitb16")


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def f(v, nd=4):
    return "null" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


# ------------------------------------------------------------------------------------------------ B
def table_b(out):
    lines = []
    for m in MODELS:
        d = load(P / f"gateB_r21b_{m}.json")
        if not d:
            continue
        lines += [f"#### {m}: B gate (relaxed: median >= 0.98, min >= 0.97, loss_rel <= 2e-2; strict recorded), same process, same weights / batch / RngTape", "",
                  "| cand | policy | n | vs R0 median / min | vs R0 loss_rel max | vs P1 median / min | relaxed (R0 & P1) | strict (R0 & P1) | vs FINAL median | head param cos min |", "|---|---|---:|---|---:|---|---|---|---:|---:|"]
        out[m] = {}
        for name, s in d["summary"].items():
            lines.append(f"| {name} | `{s['policy']}` | {s['n_batches']} | {f(s['vs_r0']['gcos_median'])} / {f(s['vs_r0']['gcos_min'])} | {s['vs_r0']['loss_rel_max']:.1e} | "
                         f"{f(s['vs_p1']['gcos_median'])} / {f(s['vs_p1']['gcos_min'])} | {s['relaxed_pass_vs_r0_and_p1']} | {s['strict_pass_vs_r0_and_p1']} | {f(s['vs_final']['gcos_median'])} | {f(s.get('head_pmin_min'), 3)} |")
            out[m][name] = {"policy": s["policy"], "n": s["n_batches"], "vs_r0": {k: s["vs_r0"][k] for k in ("gcos_median", "gcos_min", "loss_rel_max", "relaxed_ok", "strict_ok")},
                            "vs_p1": {k: s["vs_p1"][k] for k in ("gcos_median", "gcos_min", "loss_rel_max", "relaxed_ok", "strict_ok")}, "vs_final_median": s["vs_final"]["gcos_median"],
                            "relaxed_pass": s["relaxed_pass_vs_r0_and_p1"], "strict_pass": s["strict_pass_vs_r0_and_p1"], "head_param_cos_min": s.get("head_pmin_min")}
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------ S
def table_s(out):
    lines = []
    for m in MODELS:
        files = sorted(glob.glob(str(P / f"gateS_{m}_P3H_graph_vs_R0*.json")))
        if not files:
            continue
        lines += [f"#### {m}: S gate (50 steps x 2 batches, candidate P3H under StepGraphRunner vs R0 eager; EMA(0.9) deviation <= 5%, no NaN)", "",
                  "| scenario | max_rel (ckpts) | pass | finite | grad-norm ratio min / max | loss first (R0 / P3H) | loss last (R0 / P3H) | captures | dY exceed (calls) / max ratio | X exceed | W exceed |", "|---|---:|---|---|---|---|---|---:|---|---|---|"]
        out[m] = {}
        for fn in files:
            d = load(fn)
            for scen, r in d["runs"].items():
                g, c, ref = r["gate"], r["cand"], r["ref"]
                gr_d = g.get("grad_norm_ratios") or {}
                gr = list(gr_d.values()) if isinstance(gr_d, dict) else list(gr_d)
                st = c.get("fp4_stats") or {}
                dz = st.get("dZ_col", {})
                dzr = st.get("dZ_row", {})
                lines.append(f"| {scen} | {g['curve300_max_rel']:.4f} | {g['pass']} | {g['finite']} | {f(min(gr) if gr else None, 3)} / {f(max(gr) if gr else None, 3)} | {ref['loss'][0]:.3f} / {c['loss'][0]:.3f} | "
                             f"{ref['loss'][-1]:.3f} / {c['loss'][-1]:.3f} | {c.get('graph', {}).get('captures')} | row {dzr.get('exceed')}/{dzr.get('calls')} col {dz.get('exceed')}/{dz.get('calls')} max ratio {f(dz.get('max_ratio'), 2)} | "
                             f"{st.get('X_col', {}).get('exceed')}/{st.get('X_col', {}).get('calls')} (max {f(st.get('X_col', {}).get('max_ratio'), 2)}) | {st.get('W_row', {}).get('exceed')}/{st.get('W_row', {}).get('calls')} |")
                out[m][scen] = {"max_rel_ckpt": g["curve300_max_rel"], "pass": g["pass"], "finite": g["finite"], "grad_norm_ratios": gr_d, "checkpoints": g.get("checkpoints"),
                                "loss_first": [ref["loss"][0], c["loss"][0]], "loss_last": [ref["loss"][-1], c["loss"][-1]], "fp4_stats": {k: v for k, v in st.items()}, "captures": c.get("graph", {}).get("captures"),
                                "max_rel_all_steps": max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(ref["loss_ema"], c["loss_ema"]))}
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------ graph_equiv
def table_ge(out):
    lines = []
    for m in MODELS:
        files = sorted(glob.glob(str(P / f"graph_equiv_{m}_P3H*.json")))
        if not files:
            continue
        lines += [f"#### {m}: graph_equiv (8 steps; eager vs eager = run-to-run noise, graph vs eager)", "",
                  "| scenario | loss rel max graph-vs-eager | loss rel max eager-vs-eager | gnorm rel max g-v-e | update cos g-v-e | update rel L2 g-v-e | update rel L2 e-v-e | first-step loss bitwise |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        out[m] = {}
        for fn in files:
            d = load(fn)
            for scen, r in d["scenarios"].items():
                lines.append(f"| {scen} | {r['loss_rel_max_graph_vs_eager']:.2e} | {r['loss_rel_max_eager_vs_eager']:.2e} | {r['gnorm_rel_max_graph_vs_eager']:.2e} | {r['graph_vs_eager']['update_cos']:.6f} | "
                             f"{r['graph_vs_eager']['update_rel_l2']:.2e} | {r['eager_vs_eager']['update_rel_l2']:.2e} | {r['loss_bitwise_first_step']} |")
                out[m][scen] = {k: r[k] for k in ("loss_rel_max_graph_vs_eager", "loss_rel_max_eager_vs_eager", "gnorm_rel_max_graph_vs_eager", "loss_bitwise_first_step")}
                out[m][scen]["graph_vs_eager"] = r["graph_vs_eager"]
                out[m][scen]["eager_vs_eager"] = r["eager_vs_eager"]
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------ bench
def table_bench(out):
    lines = []
    for m in MODELS:
        rows = []
        for fn in sorted(glob.glob(str(P / f"bench_{m}_*_all*.json"))):
            d = load(fn)
            if not d or "summary" not in d:
                continue
            s = d["summary"]
            rows.append((fn, s))
        if not rows:
            continue
        lines += [f"#### {m}: whole-step bench under StepGraphRunner (warmup 20, 50 x 3 per phase, weighted 0.2 / 0.6 / 0.2); order = run order", "",
                  "| run | stack | early ms | mid ms | late ms | weighted img/s | mid img/s | repeat spread mid (ms) |", "|---|---|---:|---:|---:|---:|---:|---|"]
        out[m] = {"runs": {}}
        for fn, s in rows:
            sm = s["step_ms_mean"]
            rep = s["repeat_ms_mean"].get("mid", [])
            lines.append(f"| {s['name']} | {s['stack']} | {sm['early']:.2f} | {sm['mid']:.2f} | {sm['late']:.2f} | {s['weighted_img_s']:.2f} | {s['img_s']['mid']:.2f} | {', '.join(f'{x:.2f}' for x in rep)} |")
            out[m]["runs"][s["name"]] = {"stack": s["stack"], "step_ms": sm, "img_s": s["img_s"], "weighted_img_s": s["weighted_img_s"], "repeat_mid_ms": rep, "file": Path(fn).name}
        # per-stack summary: mean of the runs of the same stack
        by = {}
        for fn, s in rows:
            by.setdefault(s["stack"], []).append(s)
        lines += ["", f"{m} per stack (mean over its runs; ratio vs FINAL = mean of the FINAL runs of the same session):", "", "| stack | runs | mid ms | weighted img/s | vs FINAL (weighted) | vs FINAL (mid ms) |", "|---|---:|---:|---:|---:|---:|"]
        base = by.get("FINAL")
        bw = statistics.fmean(s["weighted_img_s"] for s in base) if base else None
        bm = statistics.fmean(s["step_ms_mean"]["mid"] for s in base) if base else None
        out[m]["by_stack"] = {}
        for stack, ss in by.items():
            w = statistics.fmean(s["weighted_img_s"] for s in ss)
            mid = statistics.fmean(s["step_ms_mean"]["mid"] for s in ss)
            ratio_w = w / bw if bw else None
            ratio_m = bm / mid if bm else None
            lines.append(f"| {stack} | {len(ss)} | {mid:.2f} | {w:.2f} | {f(ratio_w, 4)}x | {f(ratio_m, 4)}x |")
            out[m]["by_stack"][stack] = {"runs": len(ss), "mid_ms": mid, "weighted_img_s": w, "weighted_vs_final": ratio_w, "mid_vs_final": ratio_m, "weighted_img_s_runs": [s["weighted_img_s"] for s in ss]}
        lines.append("")
    return lines


def table_hr2(out):
    lines = []
    for m in MODELS:
        files = sorted(glob.glob(str(P / f"gateS_{m}_P3H_HR2_graph_vs_R0*.json")))
        if not files:
            continue
        lines += [f"#### {m}: saturation study, dZ headroom 2 instead of 8 (`P3H_HR2`, StepGraphRunner, 50 steps)", "",
                  "| scenario | max_rel (ckpts) | pass | dZ row exceed / calls | dZ col exceed / calls | dZ max amax ratio | X exceed | W exceed |", "|---|---:|---|---|---|---:|---|---|"]
        out[m] = {}
        for fn in files:
            d = load(fn)
            for scen, r in d["runs"].items():
                g, c = r["gate"], r["cand"]
                st = c.get("fp4_stats") or {}
                dr, dc = st.get("dZ_row", {}), st.get("dZ_col", {})
                lines.append(f"| {scen} | {g['curve300_max_rel']:.4f} | {g['pass']} | {dr.get('exceed')}/{dr.get('calls')} | {dc.get('exceed')}/{dc.get('calls')} | {f(max(dr.get('max_ratio', 0), dc.get('max_ratio', 0)), 2)} | "
                             f"{st.get('X_col', {}).get('exceed')}/{st.get('X_col', {}).get('calls')} | {st.get('W_row', {}).get('exceed')}/{st.get('W_row', {}).get('calls')} |")
                out[m][scen] = {"max_rel_ckpt": g["curve300_max_rel"], "pass": g["pass"], "fp4_stats": st}
        lines.append("")
    return lines


def extras():
    out = {"part": "R21b", "policy": {"H3": "head.f=mx8,head.dw=nv4+sr", "H1": "head.dw=nv4+sr", "H2": "head.dw=mx8"}, "dy_headroom": 8.0, "x_headroom": 2.0, "w_headroom": 2.0}
    a = load(P / "gateA_head_fp4.json")
    if a:
        out["gate_A"] = {"file": "profile/gateA_head_fp4.json", "all_pass": a.get("all_pass"), "gemm_fail": a["gemm"]["fail"], "head_fail": a["head"]["fail"], "graph_fail": a["graph"]["fail"],
                         "rn_vs_r20b_sim": {"n": len(a["head"]["rn"]), "logits_relL2_max": max(r["logits_relL2"] for r in a["head"]["rn"]), "grad_relL2_max": max(r["grad_relL2_max"] for r in a["head"]["rn"]),
                                            "grad_cos_min": min(r["grad_cos_min"] for r in a["head"]["rn"])},
                         "bf16_identity_all_bitwise": all(r["bitwise_equal_to_r14"] for r in a["head"]["bf16_identity"]),
                         "sr_err_ratio_range": [min(v["err_ratio"] for s in a["head"]["sr"] for v in s["per_grad"].values()), max(v["err_ratio"] for s in a["head"]["sr"] for v in s["per_grad"].values())],
                         "sr_sd_ratio_range": [min(v["sd_ratio"] for s in a["head"]["sr"] for v in s["per_grad"].values()), max(v["sd_ratio"] for s in a["head"]["sr"] for v in s["per_grad"].values())],
                         "graph_bitwise": [(c["model"], c["step1_bitwise"], c["step2_bitwise"]) for c in a["graph"]["cases"]],
                         "gemm_wgrad_relL2_max": max(c["wgrad_relL2"] for c in a["gemm"]["cases"]), "gemm_bf16out_relL2_max": max(max(c["fwd_relL2"], c["dgrad_relL2"]) for c in a["gemm"]["cases"])}
    gb = load(P / "bench_head_gemm.json")
    if gb:
        out["head_gemm_bench"] = {"file": "profile/bench_head_gemm.json", "us": {k: v["us"] for k, v in gb["cases"].items()}, "tflops": {k: v["tflops"] for k, v in gb["cases"].items()}}
    exp = {}
    for m in MODELS:
        d = load(P / f"export_{m}_P3H.json")
        if d:
            exp[m] = {"pass": d["pass"], "ckpt_step": d["ckpt_step"], "optimizer_struct_equal": d["optimizer_struct_equal"], "scheduler_keys_equal": d["scheduler_keys_equal"], "resume_step_loss": d["resume_step_loss"],
                      "resume_step_loss_finite": d["resume_step_loss_finite"], "n_params_changed": d["n_params_changed"], "export_all_finite": d["export_all_finite"], "load_model_predict": d["load_model_predict"]["shape"]}
    if exp:
        out["export_resume"] = exp
    ns = {}
    for m in MODELS:
        d = load(P / f"nsys_report21_{m}.json")
        if d:
            ns[m] = {lab: {"kernel_ms_per_step": r["kernel_ms_per_step"], "window_ms_per_step": r["busy"]["span_ms_per_step"], "busy_ratio": r["busy"]["busy_ratio"], "kernels_per_step": r["kernels_per_step"],
                           "categories_ms_per_step": {c: v["ms_per_step"] for c, v in r["categories"].items()}} for lab, r in d.items()}
    if ns:
        out["nsys"] = ns
    out["r20a_estimate_vs_measured_ms_per_step_mid"] = {"R20a_estimate": {"vitl16": {"H1_dw_nv4": 7.1, "H3_total": 10.4}, "vitb16": {"H1_dw_nv4": 7.0, "H3_total": 10.05}},
                                                        "measured_mid_step_saving": {"vitl16": {"H1": 101.04 - 93.96, "H3": 101.04 - 90.31, "H2": 101.04 - 96.49}, "vitb16": {"H1": 87.51 - 79.47, "H3": 87.51 - 76.33, "H2": 87.51 - 82.64}}}
    out["not_measured"] = ["dZ quantization fused into r14 bwd2 (evaluated analytically only)", "combined P3H + R22 backbone stack (R23)", "e2e 200-step run (R23)"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()
    res = {"gate_B": {}, "gate_S": {}, "gate_S_headroom2": {}, "graph_equiv": {}, "bench": {}}
    text = []
    for name, fn in (("gate_B", table_b), ("gate_S", table_s), ("gate_S_headroom2", table_hr2), ("graph_equiv", table_ge), ("bench", table_bench)):
        text += fn(res[name])
    print("\n".join(text))
    if args.write_results:
        rp = ROUND / "results.json"
        d = json.loads(rp.read_text())
        r21b = d.setdefault("R21b", {})
        r21b.update(extras())
        r21b.update({k: v for k, v in res.items() if v})
        rp.write_text(json.dumps(d, indent=1, allow_nan=False))
        print("results.json R21b updated:", list(r21b.keys()))


if __name__ == "__main__":
    main()
