"""Extra tables appended to summary.md by make_summary.py (e2e, validation/checkpoint, max batch, gates)."""
from __future__ import annotations

import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
MODELS = (("vitl16", "ViT-L/16 B=7"), ("vitb16", "ViT-B/16 B=14"))


def load(name):
    p = PROF / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def e2e_tables():
    out = ["## End to end with the real DataLoader (200 steps, FINAL under the graph runner)", "",
           "`img/s steady` = steps 21-200 (sync at both ends); `incl. startup` = all 200 steps from before the first `next()` (the `time_e2e` pattern, includes DataLoader worker spawn + first batch). GPU-only = preloaded batches, early phase, same process.", ""]
    for model, label in MODELS:
        d = load(f"e2e_{model}_scan.json")
        if not d:
            continue
        out += [f"### {label}: GPU-only early {d['gpu_only_early']['step_ms']:.1f} ms = {d['gpu_only_early']['img_s']:.1f} img/s", "",
                "| workers | prefetch | pin | persistent | img/s steady | steady / GPU-only | img/s incl. startup | first batch s | host wait mean / p95 / max ms |", "|---:|---:|---|---|---:|---:|---:|---:|---|"]
        for r in d["runs"]:
            st = r["setting"]
            out.append(f"| {st['workers']} | {st['prefetch']} | {'on' if st['pin'] else 'off'} | {'yes' if st['persistent'] else 'no'} | {r['img_s_steady']:.2f} | {r['steady_vs_gpu_only']:.3f} | {r['img_s_total']:.2f} | {r['first_batch_s']:.2f} | {r['wait_ms_mean']:.1f} / {r['wait_ms_p95']:.1f} / {r['wait_ms_max']:.1f} |")
        out.append("")
    for model, label in MODELS:
        d = load(f"run_final_{model}_200.json")
        if d:
            t = d["train"][0]
            out.append(f"`run_final.py --model {model} --steps 200 --workers 16 --persistent`: {t['img_s_steady']:.2f} img/s steady ({t['step_ms_steady']:.1f} ms/step), {t['img_s_total']:.2f} img/s over the 199 steps after the first, peak reserved {d['peak_reserved_gb']:.2f} GB, graph captures {d['graph']['captures']}, eager fallbacks {d['graph']['eager_fallbacks']}.")
    out.append("")
    out += ["## Validation pass and checkpoint (one each, after 20 real steps; Lightly helpers, unpatched fp32 validation path)", "", "| model | val pass s (477 images) | checkpoint save s | export save s | checkpoint GB | export GB |", "|---|---:|---:|---:|---:|---:|"]
    for model, label in MODELS:
        d = load(f"run_final_{model}_export.json")
        if d and "final" in d:
            fi = d["final"]
            out.append(f"| {label} | {fi['val_s']:.1f} | {fi['ckpt_save_s']:.2f} | {fi['export_save_s']:.2f} | {fi['ckpt_gb']:.2f} | {fi['export_gb']:.2f} |")
    out.append("")
    return out


def maxbatch_table(b7):
    rows = []
    for p in sorted(PROF.glob("maxbatch_vitl16_FINAL_graph_B*.json"), key=lambda q: int(q.stem.split("_B")[-1])):
        r = json.loads(p.read_text())["result"]
        rows.append(r)
    out = ["## Max batch, ViT-L, FINAL under the graph (mid phase, warmup 20, 30 steps; B=7 row from the 3-phase bench)", "", "| B | OOM | mid step ms | img/s | reserved GB | device used GB (mem_get_info) |", "|---:|---|---:|---:|---:|---:|"]
    if b7:
        out.append(f"| 7 | no | {b7['mid']:.1f} | {b7['img_mid']:.2f} | {b7['reserved']:.2f} | - |")
    for r in rows:
        m = r.get("mem") or {}
        out.append(f"| {r['batch']} | {'yes' if r['oom'] else 'no'} | {f(r.get('step_ms'))} | {f(r.get('img_s'), 2)} | {f(m.get('reserved_gb'), 2)} | {f(m.get('device_used_gb'), 2)} |")
    out.append("")
    return out


def gate_tables():
    out = ["## Gates on the FINAL stack", ""]
    for model, label in MODELS:
        d = load(f"gateB_{model}_FINAL.json")
        if d and "summary" in d:
            s = d["summary"]["FINAL"]
            out += [f"### Gate B-lite, {label}, {s['n_batches']} batches (1 process, seeds 0-{s['n_batches'] - 1}; each seed is a different real batch)", "",
                    "| reference | grad cos median | grad cos min | loss_rel max | grad_norm_rel max | rule (median>=0.995, min>=0.98, loss_rel<=2e-2) |", "|---|---:|---:|---:|---:|---|"]
            for tag, name in (("vs_r0", "R0 (fp32)"), ("vs_ref", "P1/TB1 (bf16)")):
                v = s[tag]
                out.append(f"| {name} | {v['grad_cos_median']:.5f} | {v['grad_cos_min']:.5f} | {v['loss_rel_max']:.2e} | {v['grad_norm_rel_max']:.4f} | {'PASS' if v['pass'] else 'FAIL'} |")
            out.append("")
            out.append("per batch global grad cos vs R0: " + ", ".join(f"{x:.4f}" for x in s["vs_r0"]["grad_cos_each"]) + "; vs P1: " + ", ".join(f"{x:.4f}" for x in s["vs_ref"]["grad_cos_each"]))
            ref = [d["seeds"][k]["ref_vs_r0"]["grad_cos_global"] for k in d["seeds"]]
            out.append("P1 itself vs R0 on the same batches: " + ", ".join(f"{x:.4f}" for x in ref))
            out.append("")
    for model, label in MODELS:
        d = load(f"gateB_{model}_bisect.json")
        if d and "summary" in d:
            out += [f"### Gate B-lite bisect, {label} (same batches for all candidates)", "", "| candidate | batches | vs R0: median / min / loss_rel max | vs P1: median / min | PASS (R0 and P1) |", "|---|---:|---|---|---|"]
            for name, s in d["summary"].items():
                a, b = s["vs_r0"], s["vs_ref"]
                out.append(f"| {name} | {s['n_batches']} | {a['grad_cos_median']:.5f} / {a['grad_cos_min']:.5f} / {a['loss_rel_max']:.1e} | {b['grad_cos_median']:.5f} / {b['grad_cos_min']:.5f} | {'PASS' if s['pass'] else 'FAIL'} |")
            out.append("")
    out += ["### Gate S: FINAL graph vs R0 eager, 50 steps, 2 real batches, EMA(0.9) deviation, worst of steps 10..50 (threshold 5e-2)", "", "| model | schedule | fixed mid | fixed late | compressed (7 captures) |", "|---|---:|---:|---:|---:|"]
    for model, label in MODELS:
        d = load(f"gateS_{model}_FINAL_graph_vs_R0.json")
        if d:
            r = d["runs"]
            out.append(f"| {label} | " + " | ".join(f"{r[k]['gate']['curve300_max_rel']:.2e} {'PASS' if r[k]['gate']['pass'] else 'FAIL'}" for k in ("schedule", "fixed_mid", "fixed_late", "compressed")) + " |")
    out += ["", "### Graph vs eager, same RNG (first-step loss bitwise equal in every scenario)", "", "| model | scenario | steps | graph vs eager: loss rel max / update rel L2 / update cos | eager vs eager: loss rel max / update rel L2 |", "|---|---|---:|---|---|"]
    for model, label in MODELS:
        d = load(f"graph_equiv_{model}_FINAL.json")
        if d:
            for k, v in d["scenarios"].items():
                g, e = v["graph_vs_eager"], v["eager_vs_eager"]
                out.append(f"| {label} | {k} | {v['steps']} | {v['loss_rel_max_graph_vs_eager']:.1e} / {g['update_rel_l2']:.1e} / {g['update_cos']:.6f} | {v['loss_rel_max_eager_vs_eager']:.1e} / {e['update_rel_l2']:.1e} |")
    out.append("")
    return out
