"""Analyze nsys CSV/SQLite exports for a locked train-opt profile.

This extends infer-opt/common/nsys_analyze.py with synchronizations, idle gaps,
launch/GPU ratios, and NVTX CPU/GPU attribution required by PLAN section 4.2.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    lowered = {str(key).strip().lower(): item for key, item in row.items()}
    for name in names:
        raw = lowered.get(name.lower())
        if raw is not None and raw != "":
            try:
                return float(raw)
            except ValueError:
                break
    return default


def text(row: dict[str, Any], *names: str, default: str = "") -> str:
    lowered = {str(key).strip().lower(): item for key, item in row.items()}
    for name in names:
        if name.lower() in lowered:
            return str(lowered[name.lower()])
    return default


def sqlite_analysis(path: Path, iterations: int, gap_us: float) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    con = sqlite3.connect(path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def all_rows(query: str) -> list[tuple[Any, ...]]:
            return list(con.execute(query))

        names = {row[0]: row[1] for row in all_rows("SELECT id,value FROM StringIds")}
        runtime = {
            row[0]: {"corr": row[0], "start": row[1], "end": row[2], "name": names.get(row[3], str(row[3])), "callchain_id": row[4]}
            for row in all_rows("SELECT correlationId,start,end,nameId,callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        }
        callchains: dict[int, list[str]] = defaultdict(list)
        if "OSRT_CALLCHAINS" in tables:
            for callchain, symbol_id in all_rows("SELECT id,symbol FROM OSRT_CALLCHAINS ORDER BY id,stackDepth"):
                callchains[callchain].append(names.get(symbol_id, str(symbol_id)))

        gpu: list[dict[str, Any]] = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            for start, end, corr, name_id in all_rows("SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
                gpu.append({"start": start, "end": end, "corr": corr, "name": names.get(name_id, "kernel"), "kind": "kernel"})
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            for start, end, corr, copy_kind in all_rows("SELECT start,end,correlationId,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
                gpu.append({"start": start, "end": end, "corr": corr, "name": f"memcpy:{copy_kind}", "kind": "dtoh" if copy_kind in (2, 4, 12) else "memcpy"})
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
            for start, end, corr in all_rows("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"):
                gpu.append({"start": start, "end": end, "corr": corr, "name": "memset", "kind": "memset"})
        gpu.sort(key=lambda item: item["start"])

        nvtx: list[dict[str, Any]] = []
        if "NVTX_EVENTS" in tables:
            nvtx = [{"start": start, "end": end, "name": label or names.get(text_id, "")}
                    for start, end, label, text_id in all_rows("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
        step_ranges = [item for item in nvtx if re.fullmatch(r"(?:step|iter)_\d+", item["name"])]
        if not step_ranges:
            return {"available": True, "error": "No step_XX/iter_XX NVTX ranges; cannot normalize GPU utilization or syncs by step", "nvtx": nvtx}

        def launch_time(op: dict[str, Any]) -> int:
            api = runtime.get(op["corr"])
            return api["start"] if api else op["start"]

        def belonging(ops: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
            return [op for op in ops if event["start"] <= launch_time(op) <= event["end"]]

        def interval_union(ops: list[dict[str, Any]]) -> int:
            intervals = sorted((op["start"], op["end"]) for op in ops)
            if not intervals:
                return 0
            total, low, high = 0, *intervals[0]
            for start, end in intervals[1:]:
                if start <= high:
                    high = max(high, end)
                else:
                    total += high - low
                    low, high = start, end
            return total + high - low

        def deepest_range(time_ns: int) -> str | None:
            matches = [item for item in nvtx if item["start"] <= time_ns <= item["end"] and item not in step_ranges]
            return min(matches, key=lambda item: item["end"] - item["start"])["name"] if matches else None

        per_step = []
        all_syncs: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        for step in step_ranges:
            step_ops = belonging(gpu, step)
            step_apis = [api for api in runtime.values() if step["start"] <= api["start"] <= step["end"]]
            syncs = []
            for api in step_apis:
                if "synchronize" in api["name"].lower() or (api["name"].lower().startswith("cudamemcpy") and any(op["corr"] == api["corr"] and op["kind"] == "dtoh" for op in step_ops)):
                    callchain = callchains.get(api["callchain_id"]) if api["callchain_id"] is not None else None
                    syncs.append({"name": api["name"], "start_ns": api["start"], "duration_us": (api["end"] - api["start"]) / 1000, "stage": deepest_range(api["start"]), "backtrace": callchain})
            all_syncs.extend(syncs)
            ordered = sorted(step_ops, key=lambda op: op["start"])
            if ordered:
                previous_end = ordered[0]["end"]
                for op in ordered[1:]:
                    gap = op["start"] - previous_end
                    if gap > gap_us * 1000:
                        midpoint = previous_end + gap // 2
                        active = next((api["name"] for api in step_apis if api["start"] <= midpoint <= api["end"]), None)
                        api = runtime.get(op["corr"])
                        gaps.append({"start_ns": previous_end, "end_ns": op["start"], "gap_us": gap / 1000.0, "stage": deepest_range(launch_time(op)), "active_cpu_api": active, "next_launch_api": api["name"] if api else None})
                    previous_end = max(previous_end, op["end"])
            wall = step["end"] - step["start"]
            busy = interval_union(step_ops)
            per_step.append({"name": step["name"], "wall_ms": wall / 1e6, "gpu_busy_ms": busy / 1e6, "gpu_busy_ratio": busy / wall if wall else None, "kernels": sum(op["kind"] == "kernel" for op in step_ops), "gpu_ops": len(step_ops), "syncs": len(syncs)})

        aggregated: dict[str, dict[str, float]] = defaultdict(lambda: {"cpu_ms": 0.0, "gpu_ms": 0.0, "gpu_ops": 0.0, "instances": 0.0})
        for event in nvtx:
            if event in step_ranges:
                continue
            ops = belonging(gpu, event)
            stat = aggregated[event["name"]]
            stat["cpu_ms"] += (event["end"] - event["start"]) / 1e6
            stat["gpu_ms"] += interval_union(ops) / 1e6
            stat["gpu_ops"] += len(ops)
            stat["instances"] += 1
        for stat in aggregated.values():
            for key in ("cpu_ms", "gpu_ms", "gpu_ops", "instances"):
                stat[key] /= len(step_ranges)
        gap_histogram = {"20_50us": 0, "50_100us": 0, "100_500us": 0, "over_500us": 0}
        for gap in gaps:
            duration = gap["gap_us"]
            bucket = "20_50us" if duration < 50 else "50_100us" if duration < 100 else "100_500us" if duration < 500 else "over_500us"
            gap_histogram[bucket] += 1
        return {
            "available": True, "captured_steps": len(step_ranges), "per_step": per_step,
            "gpu_busy_ratio": sum(item["gpu_busy_ratio"] or 0 for item in per_step) / len(per_step),
            "kernels_per_step": sum(item["kernels"] for item in per_step) / len(per_step),
            "syncs": all_syncs, "syncs_per_step": len(all_syncs) / len(step_ranges),
            "sync_calls_by_api": dict(Counter(item["name"] for item in all_syncs)),
            "gaps": gaps, "gap_histogram": gap_histogram,
            "gap_count_per_step": len(gaps) / len(step_ranges),
            "gap_ms_per_step": sum(item["gap_us"] for item in gaps) / len(step_ranges) / 1000.0,
            "gap_cause_by_api": dict(Counter(item["active_cpu_api"] or item["next_launch_api"] or "unknown" for item in gaps)),
            "nvtx": dict(aggregated),
        }
    finally:
        con.close()


def analyze(prefix: str, iterations: int = 5, top: int = 20, gap_us: float = 20.0) -> dict[str, Any]:
    if iterations <= 0 or top <= 0 or gap_us <= 0:
        raise ValueError("iterations, top, and gap_us must be positive")
    root = Path(prefix)
    kernel = rows(Path(f"{prefix}_cuda_gpu_kern_sum.csv"))
    trace = rows(Path(f"{prefix}_cuda_gpu_trace.csv"))
    api = rows(Path(f"{prefix}_cuda_api_sum.csv"))
    total_kernel_ns = sum(value(row, "Total Time (ns)") for row in kernel)
    launches = sum(value(row, "Instances", "Num Calls") for row in kernel)
    launch_api_ns = sum(value(row, "Total Time (ns)") for row in api if "launchkernel" in text(row, "Name").lower() or "launchkernel" in text(row, "name").lower())
    top_rows = sorted(kernel, key=lambda row: value(row, "Total Time (ns)"), reverse=True)[:top]
    top_kernels = [{"name": text(row, "Name"), "total_ms": value(row, "Total Time (ns)") / iterations / 1e6, "count_per_step": value(row, "Instances", "Num Calls") / iterations, "pct_kernel_time": (100 * value(row, "Total Time (ns)") / total_kernel_ns) if total_kernel_ns else 0.0} for row in top_rows]
    roofline_path = Path(f"{prefix}_roofline.json")
    roofline = json.loads(roofline_path.read_text()) if roofline_path.exists() else {}
    roofline_by_name = {item["kernel"]: item for item in roofline.get("kernel_profile", [])}
    for item in top_kernels:
        mapped = roofline_by_name.get(item["name"], {})
        item["tflops"] = mapped.get("tflops")
        item["gbps"] = mapped.get("gbps")
        item["pct_peak_compute"] = mapped.get("pct_peak_compute")
        item["pct_peak_bw"] = mapped.get("pct_peak_bw")
    sqlite = sqlite_analysis(Path(f"{prefix}.sqlite"), iterations, gap_us)
    measured_kernels = sqlite.get("kernels_per_step")
    trace_kernels = sum(bool(text(row, "GrdX")) for row in trace) if trace else None
    result = {
        "profile": str(root), "iterations": iterations,
        "kernel_ms_per_step": total_kernel_ns / iterations / 1e6 if kernel else None,
        "kernels_per_step": launches / iterations if kernel else None,
        "sqlite_kernels_per_step": measured_kernels,
        "kernel_count_match": abs(launches / iterations - measured_kernels) < 1e-6 if kernel and measured_kernels is not None else None,
        "trace_kernel_rows": trace_kernels,
        "trace_count_match": trace_kernels == int(launches) if trace_kernels is not None and kernel else None,
        "syncs_per_step": sqlite.get("syncs_per_step"),
        "gpu_busy_ratio": sqlite.get("gpu_busy_ratio"),
        "launch_api_ms_per_step": launch_api_ns / iterations / 1e6 if launch_api_ns else None,
        "launch_to_gpu_ratio": (launch_api_ns / total_kernel_ns) if total_kernel_ns else None,
        "top_kernels": top_kernels, "sqlite": sqlite,
    }
    Path(f"{prefix}_analysis.json").write_text(json.dumps(result, indent=2))
    lines = [f"# nsys analysis: {root.name}", "", f"Captured steps: {sqlite.get('captured_steps', 'unknown')}",
             f"Kernels/step: {result['kernels_per_step']} (SQLite: {measured_kernels}, match: {result['kernel_count_match']})",
             f"Trace kernel rows: {trace_kernels} (matches summary: {result['trace_count_match']})",
             f"Syncs/step: {result['syncs_per_step']}", f"GPU busy ratio: {result['gpu_busy_ratio']}",
             f"Kernel ms/step: {result['kernel_ms_per_step']}",
             f"Launch API / GPU time: {result['launch_to_gpu_ratio']}",
             f"Gaps >{gap_us:g} us: {sqlite.get('gap_count_per_step')} / step, {sqlite.get('gap_ms_per_step')} ms / step", "",
             "## NVTX ranges", "", "| range | CPU ms/step | GPU busy ms/step | GPU ops/step |", "|---|---:|---:|---:|"]
    for name, stat in sqlite.get("nvtx", {}).items():
        lines.append(f"| {name} | {stat['cpu_ms']:.3f} | {stat['gpu_ms']:.3f} | {stat['gpu_ops']:.1f} |")
    lines += ["", "## Synchronization calls", "", "| API | count |", "|---|---:|"]
    for name, count in sqlite.get("sync_calls_by_api", {}).items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "Detailed call sites and gaps: `_analysis.json`.", "", "## Top kernels", "", "| kernel | ms/step | calls/step | GPU share | TFLOPS | GB/s |", "|---|---:|---:|---:|---:|---:|"]
    for item in top_kernels:
        achieved = f"{item['tflops']:.2f}" if item["tflops"] is not None else "null"
        bandwidth = f"{item['gbps']:.2f}" if item["gbps"] is not None else "null"
        lines.append(f"| {item['name'].replace('|', '/')} | {item['total_ms']:.3f} | {item['count_per_step']:.1f} | {item['pct_kernel_time']:.1f}% | {achieved} | {bandwidth} |")
    Path(f"{prefix}_summary.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--gap-us", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze(args.prefix, args.iters, args.top, args.gap_us)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"profile={args.prefix} kernels/step={result['kernels_per_step']} gpu_ms/step={result['kernel_ms_per_step']} syncs/step={result['sqlite'].get('syncs_per_step')}")
        print("Top kernels:")
        for item in result["top_kernels"]:
            print(f"  {item['count_per_step']:.1f} x {item['name']} {item['total_ms']:.3f} ms ({item['pct_kernel_time']:.1f}%)")
        print(f"gaps > {args.gap_us} us: {result['sqlite'].get('gap_count_per_step')} / step, {result['sqlite'].get('gap_ms_per_step')} ms / step")


if __name__ == "__main__":
    main()
