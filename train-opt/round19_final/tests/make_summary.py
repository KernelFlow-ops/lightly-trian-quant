"""Write round19_final/summary.md (full waterfall + final-stack tables) from profile/*.json and results.json.  python tests/make_summary.py"""
from __future__ import annotations

import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
MODELS = (("vitl16", "ViT-L/16 B=7"), ("vitb16", "ViT-B/16 B=14"))


def load(name):
    p = PROF / name
    return json.loads(p.read_text()) if p.exists() else None


def bench(model, stack, runner):
    d = load(f"bench_{model}_{stack}_{runner}_all.json")
    if not d:
        return None
    m = d["measurement"]
    return {"w": m["weighted_img_s"], "early": m["early"]["step_ms"]["mean"], "mid": m["mid"]["step_ms"]["mean"], "late": m["late"]["step_ms"]["mean"],
            "img_early": m["early"]["img_s"], "img_mid": m["mid"]["img_s"], "img_late": m["late"]["img_s"],
            "alloc": max(m[p]["peak_allocated_gb"] for p in ("early", "mid", "late")), "reserved": max(m[p]["peak_reserved_gb"] for p in ("early", "mid", "late")),
            "std_mid": m["mid"]["step_ms"]["std"], "repeats_mid": [r["step_ms"]["mean"] for r in m["mid"]["repeats"]]}


# published numbers of the earlier rounds (their NOTES / CONTRACTS Amendments 1-4; weighted img/s and mid step ms; same container class, 575 W)
PUBLISHED = {
    "vitl16": [("R0 (original Lightly step, fp32/TF32)", 16.2, 425, "CONTRACTS 9.3 / round0"), ("P0 (r1+r3+r4+r5+r7, fp32)", 18.0, 382, "CONTRACTS 9.3 / PLAN"),
               ("TB1 (+ BF16 autocast, flash/cuDNN attn, triton RoPE, head LN bf16)", 41.4, 169, "CONTRACTS 9.3"), ("+ r13 fused block (P1R13)", 43.7, 158, "CONTRACTS 10.1 / 9.5"),
               ("+ r17 MXFP8 fc+proj+qkv_dw (P2, eager)", 51.7, 135.6, "round16 NOTES 4.1"), ("+ r15 flat optimizer (eager)", 56.0, 125.1, "round16 NOTES 4.1"),
               ("+ r16 whole-step CUDA graph (P2+r15 graph)", 59.5, 117.8, "round16 NOTES 4.1")],
    "vitb16": [("R0 (original Lightly step, fp32/TF32)", 37.2, 371, "CONTRACTS 9.3 / round0"), ("P0 (r1+r3+r4+r5+r7, fp32)", 44.4, None, "CONTRACTS 9.3"),
               ("TB1 (+ BF16 autocast, flash/cuDNN attn, triton RoPE, head LN bf16)", 96.0, 146, "CONTRACTS 9.3"), ("+ r13 fused block (P1R13)", 100.7, 138.3, "CONTRACTS 10.1 / r14 NOTES"),
               ("+ r17 MXFP8 fc+proj+qkv_dw (P2, eager)", 109.9, 127.6, "round16 NOTES 4.1"), ("+ r15 flat optimizer (eager)", 116.6, 120.3, "round16 NOTES 4.1"),
               ("+ r16 whole-step CUDA graph (P2+r15 graph)", 123.1, 113.9, "round16 NOTES 4.1")],
}


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    out = ["# round19_final summary (waterfall and final-stack tables)", "",
           "Weighted = 0.2 early + 0.6 mid + 0.2 late (step-time weighted), B fixed, warmup 20, 50 x 3, GPU exclusive, `common.bench.time_steps`. "
           "Rows marked *(this round)* were measured in round19 on this container with one harness; the earlier-stage rows are the numbers published by each round "
           "(same hardware class, measured at other times: +-2-3% drift) and are listed for the waterfall only. All ratios use the round19 R0 re-measure as denominator.", ""]
    for model, label in MODELS:
        r0 = bench(model, "R0", "eager")
        fin_g, fin_e = bench(model, "FINAL", "graph"), bench(model, "FINAL", "eager")
        p2r15g, tb1 = bench(model, "P2R15", "graph"), bench(model, "TB1", "eager")
        fin_fc, fin_nomx = bench(model, "FINAL_fc", "graph"), bench(model, "FINAL_noMX", "graph")
        base = r0["w"] if r0 else PUBLISHED[model][0][1]
        out += [f"## {label}", "", "| stage | weighted img/s | mid step ms | x vs R0 (this round) | x vs previous row | source |", "|---|---:|---:|---:|---:|---|"]
        rows = []
        for i, (name, w, mid, src) in enumerate(PUBLISHED[model]):
            if i == 0 and r0:
                rows.append((name + " *(this round, re-measured)*", r0["w"], r0["mid"], "round19 bench"))
            else:
                rows.append((name, w, mid, src))
        if p2r15g:
            rows.append(("  P2+r15 graph re-measured *(this round, drift check)*", p2r15g["w"], p2r15g["mid"], "round19 bench"))
        if fin_e:
            rows.append(("+ r14 fused head, eager (P2+r14+r15, no graph) *(this round)*", fin_e["w"], fin_e["mid"], "round19 bench"))
        if fin_g:
            rows.append(("**+ r14 fused head under the graph = FINAL** *(this round)*", fin_g["w"], fin_g["mid"], "round19 bench"))
        prev = None
        for name, w, mid, src in rows:
            out.append(f"| {name} | {f(w, 2)} | {f(mid)} | {f(w / base, 2)}x | {f(w / prev, 3) + 'x' if prev else '-'} | {src} |")
            if not name.startswith("  "):
                prev = w
        out.append("")
        if fin_g:
            out += ["Final stack, this round (all numbers measured here):", "", "| runner | early ms (img/s) | mid ms (img/s) | late ms (img/s) | weighted img/s | peak allocated GB | peak reserved GB |", "|---|---|---|---|---:|---:|---:|"]
            for tag, b in (("R0 eager", r0), ("FINAL eager", fin_e), ("FINAL graph", fin_g), ("FINAL_fc graph (r17 policy fc)", fin_fc), ("FINAL_noMX graph (bf16 only)", fin_nomx), ("P2+r15 graph (drift check)", p2r15g), ("TB1 eager (drift check)", tb1)):
                if b:
                    out.append(f"| {tag} | {b['early']:.1f} ({b['img_early']:.1f}) | {b['mid']:.1f} ({b['img_mid']:.1f}) | {b['late']:.1f} ({b['img_late']:.1f}) | {b['w']:.2f} | {b['alloc']:.2f} | {b['reserved']:.2f} |")
            out += ["", f"FINAL graph vs R0: {fin_g['w'] / r0['w']:.2f}x ({r0['w']:.2f} -> {fin_g['w']:.2f} img/s); vs FINAL eager {fin_g['w'] / fin_e['w']:.3f}x; "
                    f"memory: R0 {r0['alloc']:.1f} GB allocated -> FINAL {fin_e['alloc']:.1f} GB (eager peak allocated), {fin_g['reserved']:.1f} GB reserved under the graph.", ""]
    out += ["Note: `peak allocated` of a graph run does not contain the graph's private pool (round16 NOTES 5.5), so the graph row's memory figure that counts is `peak reserved`.", ""]
    import sys
    sys.path.insert(0, str(ROUND / "tests"))
    import summary_extra as sx
    out += sx.e2e_tables() + sx.maxbatch_table(bench("vitl16", "FINAL", "graph")) + sx.gate_tables()
    (ROUND / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
