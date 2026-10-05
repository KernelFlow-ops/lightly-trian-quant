"""Markdown tables for NOTES.md from profile/*.json (GEMM matrix, LN matrix, block bench, step timing)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"


def gemm_tables():
    out = []
    for model in ("vitl16", "vitb16"):
        for m in ("plain", "joint"):
            f = PROF / f"gemm_matrix_{model}_{m}.json"
            if not f.exists():
                continue
            rows = json.loads(f.read_text())
            cell = defaultdict(dict)
            for r in rows:
                if "us" in r:
                    cell[(r["linear"], r["kind"])][r["cand"]] = r
            cands = ["torch", "quack", "quack_sk", "spark4", "spark6"]
            M = rows[0]["M"]
            out.append(f"\n**{model}, M={M} ({m})** — us / TFLOPS (%230)\n")
            out.append("| GEMM | " + " | ".join(cands) + " | best |")
            out.append("|---|" + "---|" * (len(cands) + 1))
            for (lin, kind), d in sorted(cell.items(), key=lambda kv: (["qkv", "proj", "fc1", "fc2"].index(kv[0][0]), ["fwd", "dgrad", "wgrad"].index(kv[0][1]))):
                best = min(d.values(), key=lambda r: r["us"])
                cols = []
                for c in cands:
                    r = d.get(c)
                    cols.append(f"{r['us']:.1f} / {r['tflops']:.0f} ({r['pct_230']:.0f}%)" if r else "-")
                out.append(f"| {lin} {kind} | " + " | ".join(cols) + f" | **{best['cand']}** |")
    return "\n".join(out)


def ln_table():
    f = PROF / "ln_matrix_gm1.json"
    if not f.exists():
        return ""
    rows = json.loads(f.read_text())
    out = ["| shape | op | candidate | us | GB/s* | cos |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "us" not in r:
            continue
        cos = f"{r['err']['cos']:.7f}" if "err" in r else "-"
        out.append(f"| {r['model']} M={r['M']} | {r['kind']} | {r['cand']} | {r['us']:.1f} | {r['GBps']:.0f} | {cos} |")
    return "\n".join(out)


def block_table():
    f = PROF / "bench_block_final.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    out = ["| block | variant | fwd ms | bwd ms | fwd+bwd ms | vs autocast | vs r5 ln_fuse |", "|---|---|---|---|---|---|---|"]
    for key, v in d.items():
        base = v["autocast_default"]["fwdbwd_ms"]
        r5 = v["r5_ln_fuse"]["fwdbwd_ms"]
        for name, r in v.items():
            out.append(f"| {key} | {name} | {r['fwd_ms']:.3f} | {r['bwd_ms']:.3f} | {r['fwdbwd_ms']:.3f} | {base / r['fwdbwd_ms']:.3f}x | {r5 / r['fwdbwd_ms']:.3f}x |")
    return "\n".join(out)


def step_table():
    f = ROUND / "results.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    out = []
    for model, variants in d.items():
        if not isinstance(variants, dict):
            continue
        base = variants.get("TB1", {}).get("gpu_step", {})
        out.append(f"\n**{model}**\n")
        out.append("| stack | early ms | mid ms | late ms | weighted img/s | vs TB1 | fwd/bwd/opt/loss (mid) |")
        out.append("|---|---|---|---|---|---|---|")
        for name, r in variants.items():
            gs = r.get("gpu_step") or {}
            if not gs or gs.get("weighted_img_s") is None:
                continue
            bd = r.get("breakdown_ms") or {}
            ratio = gs["weighted_img_s"] / base["weighted_img_s"] if base.get("weighted_img_s") else float("nan")
            out.append(f"| {name} | {gs['early']['step_ms']['mean']:.1f} | {gs['mid']['step_ms']['mean']:.1f} | {gs['late']['step_ms']['mean']:.1f} | {gs['weighted_img_s']:.2f} | {ratio:.3f}x | "
                       f"{bd.get('fwd') or 0:.1f}/{bd.get('bwd') or 0:.1f}/{bd.get('opt') or 0:.1f}/{bd.get('loss') or 0:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("gemm", gemm_tables), ("ln", ln_table), ("block", block_table), ("step", step_table)):
        if which in ("all", name):
            print(f"### {name}")
            print(fn())
