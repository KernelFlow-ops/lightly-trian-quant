"""Collect round22 results from profile/*.json into profile/tables_r22.md and results.json (nothing is invented: missing pieces stay null).

    python tests/summarize_r22.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
OUT: dict = {"round": "round22_fp4_backbone", "block_gate_A": None, "gate_B": {}, "gate_S": {}, "graph_equiv": {}, "export": {}, "bench": {}, "nsys": {}, "notes": {}}
MD: list[str] = []


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ gate A
def gate_a():
    out = {}
    for m in ("vitb16", "vitl16"):
        d = load(PROF / f"test_block22_{m}.json")
        if d:
            out[m] = d
    OUT["block_gate_A"] = out or None
    MD.append("### Gate A (tests/test_block22.py)\n")
    rows = []
    for m, d in out.items():
        a = {k: v for k, v in d.items() if k.startswith("a_") and isinstance(v, dict) and "ok" in v}
        if a:
            worst_dgrad = max(max([v.get("dgrad", 0), v.get("dgrad_dgelu", 0)]) for v in a.values())
            worst_wgrad = max(max(v.get("wgrad", 0), v.get("wgrad_nosplit", 0)) for v in a.values())
            rows.append(f"| {m} | GEMM vs dequantized product | {len(a)} cases, failures {d.get('a_failures')} | dgrad max rel L2 {worst_dgrad:.2e} (bf16 floor 1.66e-3), wgrad max {worst_wgrad:.1e} |")
        b = {k: v for k, v in d.items() if k.startswith("b_") and isinstance(v, dict) and "out_cos" in v}
        if b:
            rows.append(f"| {m} | RN block vs R20b sim block | {len(b)} cases, failures {d.get(f'b_failures_{m}')} | out cos min {min(v['out_cos'] for v in b.values()):.7f}, dx cos min {min(v['dx_cos'] for v in b.values()):.7f}, dW cos min {min(v['dW_cos_min'] for v in b.values()):.7f}, max dW rel {max(v['dW_rel_max'] for v in b.values()):.1e} |")
        c = {k: v for k, v in d.items() if k.startswith("c_") and isinstance(v, dict) and "real_dx_cos_mean" in v}
        if c:
            rows.append(f"| {m} | SR real vs sim (per-seed cos vs FINAL block) | {len(c)} blocks, failures {d.get(f'c_failures_{m}')} | max |d dx cos mean| {max(v['d_dx_cos_mean'] for v in c.values()):.4f}, max |d dW cos mean| {max(v['d_dW_cos_mean'] for v in c.values()):.4f} (<= 0.003) |")
    MD.append("| model | test | cases | result |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")


# ------------------------------------------------------------------ gate B
def gate_b():
    MD.append("### Gate B (tests/gate_b_real.py, n=8 batches per file, one process per model; relaxed rule vs R0 AND vs P1; strict rule recorded)\n")
    for f in sorted(glob.glob(str(PROF / "gateB_r22_*.json"))):
        d = load(f)
        if not d:
            continue
        tag = Path(f).stem.replace("gateB_r22_", "")
        m = d["model"]
        OUT["gate_B"][tag] = {"model": m, "summary": d["summary"], "cands": d["cands"], "seeds": sorted(d["seeds"], key=int)}
        rows = []
        for n, s in d["summary"].items():
            r0, p1, fn = s["vs_r0"], s["vs_p1"], s["vs_final"]
            rows.append(f"| {n} | {s['n_batches']} | {r0['gcos_median']:.4f} / {r0['gcos_min']:.4f} | {p1['gcos_median']:.4f} / {p1['gcos_min']:.4f} | {max(r0['loss_rel_max'], p1['loss_rel_max']):.1e} | "
                        f"{'pass' if s['relaxed_pass_vs_r0_and_p1'] else 'FAIL'} | {'pass' if s['strict_pass_vs_r0_and_p1'] else 'no'} | {fn['gcos_median']:.4f} |")
        hdr = f"#### {tag} ({m}, seeds {sorted(d['seeds'], key=int)})\n\n| cand | n | vs R0 median / min | vs P1 median / min | loss_rel max | relaxed | strict | vs FINAL median |\n|---|---:|---|---|---:|---|---|---:|\n"
        MD.append(hdr + "\n".join(rows) + "\n")


# ------------------------------------------------------------------ gate S
def gate_s():
    MD.append("### Gate S (StepGraphRunner, candidate graph vs R0 eager, 4 scenarios x 50 steps x 2 batches)\n")
    for f in sorted(glob.glob(str(PROF / "gateS_vit*_graph_vs_R0*.json"))):
        d = load(f)
        if not d:
            continue
        name = Path(f).name
        model = name.split("_")[1]
        stack = d.get("stack")
        for scen, r in d["runs"].items():
            g = r["gate"]
            fp = r["cand"].get("fp4")
            OUT["gate_S"].setdefault(model, {}).setdefault(stack, {})[scen] = {
                "max_rel": g["curve300_max_rel"], "pass": g["pass"], "finite": g["finite"], "grad_norm_ratios": g["grad_norm_ratios"], "checkpoints": g["checkpoints"],
                "fp4": ({k: fp.get(k) for k in ("policy", "step", "total_calls", "total_exceed", "exceed_rate", "per_kind", "worst_sites", "headroom")} if fp else None),
                "graph_captures": r["cand"].get("graph", {}).get("captures"), "file": name}
    rows = []
    for model, st in OUT["gate_S"].items():
        for stack, sc in st.items():
            for scen, v in sc.items():
                gn = list(v["grad_norm_ratios"].values()) if isinstance(v["grad_norm_ratios"], dict) else v["grad_norm_ratios"]
                gnr = f"{min(gn):.3f}..{max(gn):.3f}" if gn else "n/a"
                fp = v["fp4"]
                sat = f"{fp['total_exceed']}/{fp['total_calls']} ({100 * fp['exceed_rate']:.2f}%)" if fp and fp.get("total_calls") is not None else "n/a"
                rows.append(f"| {model} | {stack} | {scen} | {100 * v['max_rel']:.2f}% | {'pass' if v['pass'] else 'FAIL'} | {v['finite']} | {gnr} | {sat} |")
    MD.append("| model | stack | scenario | EMA max rel dev vs R0 | gate | finite | grad-norm ratio (cand/R0) range | saturations exceed/calls |\n|---|---|---|---:|---|---|---|---|\n" + "\n".join(rows) + "\n")


# ------------------------------------------------------------------ graph_equiv
def graph_equiv():
    MD.append("### graph_equiv (tests/graph_equiv22.py, 8 steps)\n")
    rows = []
    for f in sorted(glob.glob(str(PROF / "graph_equiv_vit*.json"))):
        d = load(f)
        if not d:
            continue
        model = Path(f).name.split("_")[2]
        OUT["graph_equiv"][f"{model}:{d['stack']}"] = d["scenarios"]
        for scen, r in d["scenarios"].items():
            rows.append(f"| {model} | {d['stack']} | {scen} | {r['graph_vs_eager']['update_cos']:.6f} | {r['eager_vs_eager']['update_cos']:.6f} | {r['loss_rel_max_graph_vs_eager']:.1e} | {r['loss_rel_max_eager_vs_eager']:.1e} | {r['fp4_state_advance']['ok']} |")
    MD.append("| model | stack | scenario | update cos graph-vs-eager | eager-vs-eager | loss rel max g-vs-e | e-vs-e | state advanced K/step |\n|---|---|---|---:|---:|---:|---:|---|\n" + "\n".join(rows) + "\n")


# ------------------------------------------------------------------ bench
def bench():
    MD.append("### Bench (StepGraphRunner, 0.2/0.6/0.2 weighted early/mid/late, warmup 20, 50 x 3; every run is one process; same-session A-B-A-B)\n")
    runs = []
    for f in sorted(glob.glob(str(PROF / "bench_vit*_graph_all*.json"))):
        d = load(f)
        if not d:
            continue
        sm = d["summary"]
        key = Path(f).stem.replace("bench_", "")
        OUT["bench"][key] = {"stack": sm["stack"], "model": sm["model"], "step_ms_mean": sm["step_ms_mean"], "img_s": sm["img_s"], "weighted_img_s": sm["weighted_img_s"],
                             "repeat_ms_mean": sm["repeat_ms_mean"], "mtime": Path(f).stat().st_mtime}
        runs.append((Path(f).stat().st_mtime, key, sm))
    runs.sort()
    sessions, cur = [], []
    for r in runs:
        if cur and r[0] - cur[-1][0] > 3600:
            sessions.append(cur)
            cur = []
        cur.append(r)
    if cur:
        sessions.append(cur)
    rows, ratios = [], {}
    for si, ses in enumerate(sessions):
        for model in ("vitl16", "vitb16"):
            mr = [r for r in ses if r[2]["model"] == model]
            fin = [r[2] for r in mr if r[2]["stack"] == "FINAL"]
            if not fin:
                continue
            f_img = statistics.fmean(x["weighted_img_s"] for x in fin)
            f_mid = statistics.fmean(x["step_ms_mean"]["mid"] for x in fin)
            for stack in sorted({r[2]["stack"] for r in mr}):
                xs = [r[2] for r in mr if r[2]["stack"] == stack]
                img = statistics.fmean(x["weighted_img_s"] for x in xs)
                mid = statistics.fmean(x["step_ms_mean"]["mid"] for x in xs)
                ratios.setdefault(f"session{si}", {}).setdefault(model, {})[stack] = {
                    "n_runs": len(xs), "weighted_img_s_runs": [x["weighted_img_s"] for x in xs], "weighted_img_s_mean": img, "mid_ms_runs": [x["step_ms_mean"]["mid"] for x in xs], "mid_ms_mean": mid,
                    "early_ms_mean": statistics.fmean(x["step_ms_mean"]["early"] for x in xs), "late_ms_mean": statistics.fmean(x["step_ms_mean"]["late"] for x in xs),
                    "vs_FINAL_img_s": img / f_img, "vs_FINAL_mid_ms_delta": mid - f_mid}
                rows.append(f"| {si} | {model} | {stack} | {len(xs)} | {' / '.join(f'{x['weighted_img_s']:.2f}' for x in xs)} | {img:.2f} | {mid:.2f} | {img / f_img:.4f} | {mid - f_mid:+.2f} |")
    OUT["bench_sessions"] = ratios
    MD.append("| session | model | stack | runs | weighted img/s per run | mean img/s | mean mid ms | x FINAL (img/s) | d mid ms |\n|---|---|---|---:|---|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n")


# ------------------------------------------------------------------ export / resume
def export():
    MD.append("### Export / resume (round19 run_final.py 20 graph steps + val + checkpoint + export, then round19 tests/check_export.py in a fresh process)\n")
    rows = []
    for m in ("vitl16", "vitb16"):
        d = load(PROF / f"export_{m}.json")
        r = load(PROF / f"run_final_{m}_export.json")
        if not d and not r:
            continue
        OUT["export"][m] = {"check_export": ({k: d.get(k) for k in ("pass", "ckpt_step", "optimizer_struct_equal", "scheduler_keys_equal", "resume_step_loss", "resume_step_loss_finite", "n_params_changed",
                                                               "max_abs_weight_change_vs_init", "export_all_finite", "ckpt_meta_equal_R0", "export_meta_equal_original", "resume_load_state_dict", "load_model_predict")} if d else None),
                            "run_final": ({k: r.get(k) for k in ("stack", "steps", "batch", "peak_reserved_gb")} | {"graph": {k: r["graph"].get(k) for k in ("captures", "eager_fallbacks", "replays")}} if r else None)}
        rows.append(f"| {m} | {r['stack'] if r else None} | {d.get('pass') if d else None} | {d.get('resume_step_loss') if d else None} | {d.get('n_params_changed') if d else None} | {d.get('optimizer_struct_equal') if d else None} | {(d.get('load_model_predict') or {}).get('shape') if d else None} |")
    MD.append("| model | stack | check_export pass | resume step loss | n params changed | optimizer struct equal | load_model predict shape |\n|---|---|---|---:|---:|---|---|\n" + "\n".join(rows) + "\n")


def main():
    gate_a()
    gate_b()
    gate_s()
    graph_equiv()
    bench()
    export()
    (PROF / "tables_r22.md").write_text("\n".join(MD))
    old = load(ROUND / "results.json") or {}
    old.update({k: v for k, v in OUT.items() if v not in (None, {})})
    (ROUND / "results.json").write_text(json.dumps(old, indent=1, default=str))
    print("\n".join(MD))


if __name__ == "__main__":
    main()
