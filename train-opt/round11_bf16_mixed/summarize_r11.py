"""Markdown tables from round11 results.json / profile/bench_*.json.

    python summarize_r11.py matrix            # step table from results.json (all-phase runs)
    python summarize_r11.py screen            # mid-phase screening table from profile/bench_*scr_*_mid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
MODELS = ("vitl16", "vitb16")
BATCH = {"vitl16": 7, "vitb16": 14}
WEIGHTS = {"early": 0.2, "mid": 0.6, "late": 0.2}


def matrix(order: list[str] | None = None) -> str:
    data = json.loads((ROUND / "results.json").read_text())
    lines = []
    for model in MODELS:
        rows = data.get(model, {})
        names = order or list(rows)
        ref = {k: rows[k]["gpu_step"]["weighted_img_s"] for k in ("R0", "P0") if k in rows and rows[k]["gpu_step"]["weighted_img_s"]}
        lines += [f"### {model} (B={BATCH[model]})", "", "| composition | early ms | mid ms | late ms | weighted img/s | vs R0 | vs P0 | peak mem GB | fwd/loss/bwd/opt ms (mid) |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name in names:
            row = rows.get(name)
            if not row:
                continue
            gs = row["gpu_step"]
            ms = [gs[p]["step_ms"]["mean"] for p in ("early", "mid", "late")]
            if any(v is None for v in ms):
                continue
            w = gs["weighted_img_s"]
            bd = row["breakdown_ms"]
            lines.append(
                f"| {name} | {ms[0]:.1f} | {ms[1]:.1f} | {ms[2]:.1f} | {w:.2f} | "
                f"{(w / ref['R0']) if 'R0' in ref else float('nan'):.2f}x | {(w / ref['P0']) if 'P0' in ref else float('nan'):.2f}x | {row['peak_mem_gb']:.1f} | "
                f"{bd['fwd']:.1f} / {bd['loss']:.1f} / {bd['bwd']:.1f} / {bd['opt']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def screen() -> str:
    lines = []
    for model in MODELS:
        files = sorted((ROUND / "profile").glob(f"bench_{model}_scr_*_mid*.json"))
        rows = []
        for f in files:
            summary = json.loads(f.read_text())["summary"]
            bd = summary["breakdown_ms"]["mid"]
            rows.append((summary["name"].removeprefix("scr_"), summary["step_ms_mean"]["mid"], summary["img_s"]["mid"], bd, summary["peak_allocated_gb"]["mid"], summary["stack"]))
        base = next((r for r in rows if r[0] == "TB1"), None)
        lines += [f"### {model} mid-phase screening (warmup 20, 50 x 3)", "", "| name | step ms | img/s | vs TB1 ms | fwd | loss | bwd | opt | peak GB | stack |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, ms, imgs, bd, peak, stack in rows:
            delta = f"{ms - base[1]:+.1f}" if base else ""
            lines.append(f"| {name} | {ms:.1f} | {imgs:.2f} | {delta} | {bd['fwd']:.1f} | {bd['loss']:.1f} | {bd['bwd']:.1f} | {bd['opt']:.1f} | {peak:.1f} | `{stack}` |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    print(matrix(sys.argv[2:] or None) if mode == "matrix" else screen())
