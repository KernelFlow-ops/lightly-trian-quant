"""Summarize the CSVs written by profile_nsys.sh into a markdown report.

Usage: python common/nsys_analyze.py <out_prefix> [--iters 5] [--top 15]
Reads <prefix>_cuda_gpu_kern_sum.csv, <prefix>_nvtx_gpu_proj_sum.csv,
<prefix>_cuda_api_sum.csv (nsys stats --format csv). Times are per profiled iteration.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def col(row: dict[str, str], *names: str) -> str:
    for n in names:
        for k in row:
            if k.strip().lower() == n.lower():
                return row[k]
    raise KeyError(f"none of {names} in {list(row)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    p, it = args.prefix, args.iters
    out: list[str] = [f"### nsys: `{Path(p).name}` (per iteration, {it} iterations captured)", ""]

    kern = read_csv(Path(f"{p}_cuda_gpu_kern_sum.csv"))
    if kern:
        total_ns = sum(float(col(r, "Total Time (ns)")) for r in kern)
        n_launch = sum(int(col(r, "Instances")) for r in kern)
        out += [f"- GPU kernel time: **{total_ns / it / 1e6:.3f} ms**, kernel launches: "
                f"**{n_launch / it:.0f}**", "",
                "| # | kernel | time/iter (ms) | % | launches/iter | avg (us) |",
                "|---|---|---|---|---|---|"]
        rows = sorted(kern, key=lambda r: -float(col(r, "Total Time (ns)")))[: args.top]
        for i, r in enumerate(rows):
            t = float(col(r, "Total Time (ns)"))
            name = col(r, "Name").replace("|", "/")
            name = name if len(name) < 90 else name[:87] + "..."
            out.append(f"| {i + 1} | `{name}` | {t / it / 1e6:.3f} | {100 * t / total_ns:.1f} | "
                       f"{int(col(r, 'Instances')) / it:.0f} | {float(col(r, 'Avg (ns)')) / 1e3:.1f} |")
        out.append("")

    nvtx = read_csv(Path(f"{p}_nvtx_gpu_proj_sum.csv"))
    if nvtx:
        out += ["| NVTX range | GPU projected (ms/iter) | CPU range (ms/iter) | GPU/CPU |",
                "|---|---|---|---|"]
        for r in nvtx:
            name = col(r, "Range").lstrip(":")
            if name.startswith(("iter_", "block_")):
                continue
            gpu = float(col(r, "Total Proj Time (ns)")) / it / 1e6
            cpu = float(col(r, "Total Range Time (ns)")) / it / 1e6
            out.append(f"| {name} | {gpu:.3f} | {cpu:.3f} | {gpu / cpu if cpu else 0:.2f} |")
        blocks = [r for r in nvtx if col(r, "Range").lstrip(":").startswith("block_")]
        if blocks:
            g = sum(float(col(r, "Total Proj Time (ns)")) for r in blocks) / it / 1e6
            c = sum(float(col(r, "Total Range Time (ns)")) for r in blocks) / it / 1e6
            out.append(f"| all blocks ({len(blocks)}) | {g:.3f} | {c:.3f} | {g / c if c else 0:.2f} |")
        out.append("")

    api = read_csv(Path(f"{p}_cuda_api_sum.csv"))
    if api:
        out += ["| CUDA API | calls/iter | time/iter (ms) | avg (us) |", "|---|---|---|---|"]
        rows = sorted(api, key=lambda r: -float(col(r, "Total Time (ns)")))[:8]
        for r in rows:
            out.append(f"| {col(r, 'Name')} | {int(col(r, 'Num Calls')) / it:.0f} | "
                       f"{float(col(r, 'Total Time (ns)')) / it / 1e6:.3f} | "
                       f"{float(col(r, 'Avg (ns)')) / 1e3:.1f} |")
        out.append("")
    out += range_breakdown(Path(f"{p}.sqlite"), it)
    text = "\n".join(out)
    Path(f"{p}_summary.md").write_text(text)
    print(text)


STAGES = ("prep", "model", "post")


def range_breakdown(db: Path, it: int) -> list[str]:
    """Attribute every kernel / memcpy to the top-level stage whose CPU-side launch
    (runtime API call) falls inside that NVTX range; report GPU busy vs span."""
    import sqlite3

    if not db.exists():
        return []
    con = sqlite3.connect(db)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    tables = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = q("SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
               "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL")
    stage_ranges = [(s, e, t) for s, e, t in ranges if t in STAGES]
    api = dict(q("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    gpu_ops = []
    for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table in tables:
            gpu_ops += [(s, e, c, kind) for s, e, c in
                        q(f"SELECT start, end, correlationId FROM {table}")]
    stats = {st: {"kernel": [0, 0.0], "memcpy": [0, 0.0], "memset": [0, 0.0],
                  "first": [], "last": []} for st in STAGES}
    for s, e, corr, kind in gpu_ops:
        t_api = api.get(corr)
        if t_api is None:
            continue
        for rs, re_, name in stage_ranges:
            if rs <= t_api <= re_:
                d = stats[name]
                d[kind][0] += 1
                d[kind][1] += (e - s)
                break
    # GPU span per stage instance: first op start .. last op end.
    spans = {st: 0.0 for st in STAGES}
    cpu = {st: 0.0 for st in STAGES}
    for rs, re_, name in stage_ranges:
        ops = [(s, e) for s, e, corr, _ in gpu_ops if rs <= api.get(corr, -1) <= re_]
        cpu[name] += re_ - rs
        if ops:
            spans[name] += max(e for _, e in ops) - min(s for s, _ in ops)
    lines = ["| stage | CPU range (ms) | GPU span (ms) | kernels | kernel time (ms) | "
             "memcpy (n / ms) | GPU busy in span |", "|---|---|---|---|---|---|---|"]
    for st in STAGES:
        d = stats[st]
        busy = d["kernel"][1] + d["memcpy"][1] + d["memset"][1]
        lines.append(
            f"| {st} | {cpu[st] / it / 1e6:.3f} | {spans[st] / it / 1e6:.3f} | "
            f"{d['kernel'][0] / it:.0f} | {d['kernel'][1] / it / 1e6:.3f} | "
            f"{d['memcpy'][0] / it:.0f} / {d['memcpy'][1] / it / 1e6:.3f} | "
            f"{100 * busy / spans[st] if spans[st] else 0:.0f}% |")
    return ["", "Per-stage attribution (kernel assigned by its launching API call's CPU time):", "",
            *lines, ""]


if __name__ == "__main__":
    main()
