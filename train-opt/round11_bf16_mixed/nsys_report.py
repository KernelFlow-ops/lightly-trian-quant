"""Per-category GPU time, CPU-vs-GPU per NVTX range, GPU busy ratio and top kernels from one nsys export.

Usage:  python nsys_report.py <prefix> [--steps 5] [--out <md>]
Expects the files written by common/profile_nsys.sh (<prefix>.sqlite, <prefix>_cuda_gpu_kern_sum.csv, ...)
and optionally <prefix>_analysis.json from common/nsys_analyze.py (CPU-vs-GPU NVTX table, busy ratio).

GPU time of an op is attributed to the NVTX phase that was active on the host when the op was launched
(fwd / loss / bwd / clip_opt_sched), then classified by kernel name:
    GEMM, attention, head conv/dwconv, LN, RoPE, elementwise/reduce, loss/matcher (whole loss phase),
    optimizer (whole clip_opt_sched phase), memcpy/memset, other.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

PHASES = ("fwd", "loss", "bwd", "clip_opt_sched")

NAME_RULES: list[tuple[str, re.Pattern]] = [
    ("attention", re.compile(r"fmha|flash|sdpa|mha_|scaled_dot|attention|pytorchmemeff|cudnn_generated_oai|fused_attn|softmax_fwd_generic|dot_do_o", re.I)),
    ("head conv/dwconv", re.compile(r"conv|implicit_gemm|dwconv|depthwise|cudnn::|nchwtonhwc|nhwctonchw|cudnn_infer|wgrad2d|dgrad2d|fprop", re.I)),
    ("GEMM", re.compile(r"gemm|cutlass|cublas|xmma|nvjet|gemv|ampere_|sm\d+_(?:xmma|gemm)|splitk|s1688|s16816|wmma|tensorop|hgemm", re.I)),
    ("LN", re.compile(r"layer_norm|layernorm|vectorized_layer|gammabeta|^_fwd$|^_bwd$|^_fwd_|^_bwd_|_ln_|rms_norm|batch_norm|group_norm", re.I)),
    ("RoPE", re.compile(r"^_rotate|rope|rotate_half", re.I)),
    ("elementwise/reduce", re.compile(r"elementwise|vectorized|unrolled|reduce_kernel|reduce|cat_array|catarray|copy|fill|index|gelu|mul_|add_|bfloat16_copy|direct_copy|gatherkernel|scatter|softmax|cumsum|sort|where|clamp|sum|mean|pow|rsqrt", re.I)),
]


def classify_kernel(name: str) -> str:
    short = name.split("(")[0]
    for category, pattern in NAME_RULES:
        if pattern.search(short) or (category in ("attention",) and pattern.search(name[:200])):
            return category
    return "other"


def rows(path: Path):
    return list(csv.DictReader(path.open(newline=""))) if path.exists() else []


def build(prefix: str, steps: int) -> dict:
    con = sqlite3.connect(f"{prefix}.sqlite")
    names = {i: v for i, v in con.execute("SELECT id,value FROM StringIds")}
    runtime = {c: s for c, s in con.execute("SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    nvtx = [(s, e, t if t is not None else names.get(tid, "")) for s, e, t, tid in con.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    step_ranges = sorted(((s, e, n) for s, e, n in nvtx if re.fullmatch(r"step_\d+", n)))
    phase_ranges = sorted(((s, e, n) for s, e, n in nvtx if n in PHASES))
    ops: list[tuple[str, int, int, str]] = []  # (kind, start, end, name)
    for s, e, c, n in con.execute("SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        ops.append(("kernel", runtime.get(c, s), e - s, names.get(n, "?")))
    for s, e, c in con.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        ops.append(("memcpy", runtime.get(c, s), e - s, "memcpy"))
    for s, e, c in con.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"):
        ops.append(("memset", runtime.get(c, s), e - s, "memset"))

    def phase_of(t: int) -> str | None:
        for s, e, n in phase_ranges:
            if s <= t <= e:
                return n
        return None

    in_step = lambda t: any(s <= t <= e for s, e, _ in step_ranges)
    cat_ms = defaultdict(float)
    cat_count = defaultdict(float)
    cat_phase_ms: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    phase_ms = defaultdict(float)
    kernel_names_other: dict[str, float] = defaultdict(float)
    for kind, t, dur, name in ops:
        if not in_step(t):
            continue
        phase = phase_of(t) or "outside_phase"
        if kind != "kernel":
            category = "memcpy/memset"
        elif phase == "loss":
            category = "loss/matcher"
        elif phase == "clip_opt_sched":
            category = "optimizer"
        else:
            category = classify_kernel(name)
            if category == "attention":
                category = "attention bwd" if phase == "bwd" else "attention fwd"
            if category == "other":
                kernel_names_other[name.split("(")[0][:120]] += dur / 1e6
        cat_ms[category] += dur / 1e6
        cat_count[category] += 1
        cat_phase_ms[category][phase] += dur / 1e6
        phase_ms[phase] += dur / 1e6
    n_steps = max(len(step_ranges), 1) if steps <= 0 else steps
    n_steps = len(step_ranges) or steps
    total = sum(cat_ms.values())
    categories = {
        c: {"ms_per_step": v / n_steps, "share": v / total if total else 0.0, "kernels_per_step": cat_count[c] / n_steps,
            "by_phase_ms_per_step": {p: x / n_steps for p, x in cat_phase_ms[c].items()}}
        for c, v in sorted(cat_ms.items(), key=lambda kv: -kv[1])
    }
    wall = sum(e - s for s, e, _ in step_ranges) / 1e6 / n_steps if step_ranges else None
    other_top = sorted(kernel_names_other.items(), key=lambda kv: -kv[1])[:8]
    return {
        "steps": n_steps, "step_wall_ms": wall, "gpu_kernel_ms_per_step": total / n_steps,
        "categories": categories, "phase_gpu_ms_per_step": {p: v / n_steps for p, v in phase_ms.items()},
        "other_top": [(n, ms / n_steps) for n, ms in other_top],
    }


def render(prefix: str, data: dict, analysis: dict | None, kern_rows: list[dict], steps: int, top: int = 20) -> str:
    lines = [f"# nsys category report: {Path(prefix).name}", "",
             f"Captured steps: {data['steps']}; step wall (NVTX, host) {data['step_wall_ms']:.1f} ms; "
             f"summed GPU op time {data['gpu_kernel_ms_per_step']:.1f} ms/step", ""]
    if analysis:
        sq = analysis.get("sqlite", {})
        lines += [f"GPU busy ratio {sq.get('gpu_busy_ratio'):.4f}; kernels/step {analysis.get('kernels_per_step'):.0f}; "
                  f"syncs/step {sq.get('syncs_per_step')}; gaps>20us: {sq.get('gap_count_per_step')}/step = {sq.get('gap_ms_per_step'):.2f} ms/step; "
                  f"launch API time / GPU kernel time = {analysis.get('launch_to_gpu_ratio')}", ""]
    lines += ["## GPU time per category (ms per step)", "", "| category | ms/step | share | kernels/step | fwd | loss | bwd | opt |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in data["categories"].items():
        by = item["by_phase_ms_per_step"]
        lines.append(f"| {name} | {item['ms_per_step']:.2f} | {100 * item['share']:.1f}% | {item['kernels_per_step']:.0f} | "
                     f"{by.get('fwd', 0):.2f} | {by.get('loss', 0):.2f} | {by.get('bwd', 0):.2f} | {by.get('clip_opt_sched', 0):.2f} |")
    lines += ["", "Phase GPU time (ms/step, summed op duration by launch phase): " + ", ".join(f"{p}={v:.1f}" for p, v in data["phase_gpu_ms_per_step"].items()), ""]
    if data["other_top"]:
        lines += ["Largest 'other' kernels (ms/step): " + "; ".join(f"{n} {ms:.2f}" for n, ms in data["other_top"]), ""]
    if analysis:
        lines += ["## CPU vs GPU per NVTX range (per step)", "", "| range | CPU ms | GPU busy ms | GPU ops | CPU/GPU |", "|---|---:|---:|---:|---:|"]
        nv = analysis["sqlite"].get("nvtx", {})
        order = [n for n in ("fwd", "loss", "bwd", "clip_opt_sched") if n in nv] + [n for n in nv if n not in ("fwd", "loss", "bwd", "clip_opt_sched") and not re.fullmatch(r"blk_\d+", n)]
        for name in order:
            s = nv[name]
            ratio = s["cpu_ms"] / s["gpu_ms"] if s["gpu_ms"] else float("nan")
            lines.append(f"| {name} | {s['cpu_ms']:.2f} | {s['gpu_ms']:.2f} | {s['gpu_ops']:.0f} | {ratio:.2f} |")
        blocks = [nv[n] for n in nv if re.fullmatch(r"blk_\d+", n)]
        if blocks:
            lines.append(f"| blk_* (sum of {len(blocks)}) | {sum(b['cpu_ms'] for b in blocks):.2f} | {sum(b['gpu_ms'] for b in blocks):.2f} | {sum(b['gpu_ops'] for b in blocks):.0f} | "
                         f"{sum(b['cpu_ms'] for b in blocks) / max(sum(b['gpu_ms'] for b in blocks), 1e-9):.2f} |")
        lines.append("")
    total_ns = sum(float(r["Total Time (ns)"]) for r in kern_rows)
    lines += [f"## Top {top} kernels", "", "| kernel | ms/step | calls/step | share | category |", "|---|---:|---:|---:|---|"]
    for r in sorted(kern_rows, key=lambda r: -float(r["Total Time (ns)"]))[:top]:
        name = r["Name"].replace("|", "/")
        lines.append(f"| {name[:110]} | {float(r['Total Time (ns)']) / steps / 1e6:.3f} | {float(r['Instances']) / steps:.1f} | {100 * float(r['Total Time (ns)']) / total_ns:.1f}% | {classify_kernel(r['Name'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args()
    data = build(args.prefix, args.steps)
    analysis_path = Path(f"{args.prefix}_analysis.json")
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
    kern = rows(Path(f"{args.prefix}_cuda_gpu_kern_sum.csv"))
    text = render(args.prefix, data, analysis, kern, data["steps"], args.top)
    out = Path(args.out or f"{args.prefix}_r11_report.md")
    out.write_text(text)
    Path(str(out).replace(".md", ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
