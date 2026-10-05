"""Summary of round23 gate B reports (pure python, no torch): relaxed / strict rule vs R0 and vs P1, per-seed minima, combined-stack consistency.  Shared by gate_b_p3.py and make_results.py."""
import statistics


def summarize(report, names):
    out = {}
    for name in names:
        rows = [s["cands"][name] for s in report["seeds"].values() if name in s["cands"] and "vs_r0" in s["cands"][name]]
        if not rows:
            continue
        seeds = [k for k, s in report["seeds"].items() if name in s["cands"] and "vs_r0" in s["cands"][name]]
        o = {"n_batches": len(rows), "policy": report["cands"].get(name), "seeds": seeds}
        for tag in ("vs_r0", "vs_p1", "vs_final"):
            gc = [r[tag]["grad_cos_global"] for r in rows]
            mn = [r[tag]["grad_cos_min"] for r in rows]
            lr = [r[tag]["loss_rel"] for r in rows]
            o[tag] = {"gcos_median": statistics.median(gc), "gcos_min": min(gc), "gcos_min_seed": seeds[gc.index(min(gc))], "gcos_max": max(gc), "pmin_min": min(mn), "loss_rel_max": max(lr),
                      "grad_norm_rel_max": max(r[tag]["grad_norm_rel"] for r in rows), "gcos_each": gc,
                      "relaxed_ok": bool(statistics.median(gc) >= 0.98 and min(gc) >= 0.97 and max(lr) <= 2e-2),
                      "strict_ok": bool(statistics.median(gc) >= 0.99 and min(gc) >= 0.98 and max(lr) <= 2e-2)}
        o["relaxed_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["relaxed_ok"] and o["vs_p1"]["relaxed_ok"])
        o["strict_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["strict_ok"] and o["vs_p1"]["strict_ok"])
        o["relaxed_verdict_needs_n>=8"] = len(rows) < 8
        o["step_seconds_mean"] = statistics.fmean(r["seconds"] for r in rows)
        o["peak_gb_max"] = max(r["peak_gb"] for r in rows)
        for ref in ("P3H", "P3B"):                       # combined-stack consistency (only recorded for P3)
            key = f"vs_{ref}"
            if all(key in r for r in rows):
                o[key] = {"global_cos_min": min(r[key]["grad_cos_global"] for r in rows),
                          **{f"{g}_cos_min": min(r[key]["group_cos"].get(g, 1.0) for r in rows) for g in ("blocks", "head_convt")},
                          **{f"{g}_cos_median": statistics.median(r[key]["group_cos"].get(g, 1.0) for r in rows) for g in ("blocks", "head_convt")}}
        out[name] = o
    return out
