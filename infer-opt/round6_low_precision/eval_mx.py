"""Real MXFP8 model accuracy and locked CUDA-graph timing on semantic video frames.

Examples:
  python eval_mx.py --model vitb16 --units greedy_vitb16_b32.json --gate --heldout
  python eval_mx.py --model vitb16 --units greedy_vitb16_b32.json --bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "round7_head_postprocess"))

import sens  # noqa: E402
import greedy  # noqa: E402
import head as hd  # noqa: E402
import model_factory as mf  # noqa: E402
import pipeline as pl  # noqa: E402
import fp8linear as row_fp8  # noqa: E402
import mxlinear as mx  # noqa: E402
from bench import gpu_lock, time_fn, write_results, env_info  # noqa: E402
from bench_fp8 import capture  # noqa: E402


def heldout(ev, units):
    frames = greedy.frames_all()
    m32 = mf.load_model(ev.name)
    pl.init(m32)
    ref = greedy.masks_conf(m32, "32-true", frames)
    del m32
    torch.cuda.empty_cache()
    pl.init(ev.model)
    mx.enable(ev.model, units)
    try:
        cand = greedy.masks_conf(ev.model, "bf16-mixed", frames)
    finally:
        mx.disable(ev.model)
    inter = union = 0
    conf_diffs, frame_ious = [], []
    for (rm, rc), (cm, cc) in zip(ref, cand):
        a, b = rm == 1, cm == 1
        i, u = int((a & b).sum()), int((a | b).sum())
        inter += i
        union += u
        frame_ious.append(i / u if u else 1.0)
        conf_diffs.append(abs(cc - rc) / max(abs(rc), 1e-6))
    return {
        "frames": len(frames), "pooled_iou": inter / union if union else 1.0,
        "min_frame_iou": min(frame_ious), "conf_rel_max": max(conf_diffs),
        "conf_rel_p90": float(np.percentile(conf_diffs, 90)),
        "frames_conf_over_3pct": sum(v > 0.03 for v in conf_diffs),
        "frames_conf_over_5pct": sum(v > 0.05 for v in conf_diffs),
    }


def bench(ev, units, reverse=False):
    model = ev.model
    batch = ev.batches[0].clone()
    times = {}
    modes = ("bf16_before", "mx_fp8", "row_fp8", "bf16_after") if reverse else (
        "bf16_before", "row_fp8", "mx_fp8", "bf16_after")
    # Lock the entire A/B/A sequence, including graph warm-up and capture.
    with gpu_lock():
        for mode in modes:
            if mode == "row_fp8":
                row_fp8.enable(model, units)
            elif mode == "mx_fp8":
                mx.enable(model, units)
            try:
                replay, _, _ = capture(model, batch)
                t = time_fn(replay, warmup=15, iters=80, repeats=3, lock=False)
                times[mode] = {"gpu_ms_p50": t["gpu_ms_p50"], "wall_ms_p50": t["wall_ms_p50"]}
                print(mode, times[mode], flush=True)
                del replay
            finally:
                if mode == "row_fp8":
                    row_fp8.disable(model)
                elif mode == "mx_fp8":
                    mx.disable(model)
    base = (times["bf16_before"]["gpu_ms_p50"] + times["bf16_after"]["gpu_ms_p50"]) / 2
    times["mx_vs_bf16_gpu_speedup"] = base / times["mx_fp8"]["gpu_ms_p50"]
    times["mx_vs_row_gpu_speedup"] = times["row_fp8"]["gpu_ms_p50"] / times["mx_fp8"]["gpu_ms_p50"]
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("vitb16", "vitl16"), required=True)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="measure MX before row FP8")
    args = ap.parse_args()
    units = [tuple(u) for u in json.loads(args.units.read_text())["chosen_units"]]
    ev = sens.Evaluator(args.model)
    hd.enable(ev.model)
    out = {"units_spec": str(args.units), "n_units": len(units),
           "counts": {t: sum(u[1] == t for u in units) for t in sens.TYPES}}
    if args.gate:
        out["bf16_gate"] = ev.run()
        mx.enable(ev.model, units)
        try:
            out["mx_gate"] = ev.run()
        finally:
            mx.disable(ev.model)
        print("8-frame gate:", json.dumps({k: out[k] for k in ("bf16_gate", "mx_gate")}), flush=True)
    if args.heldout:
        out["heldout_126"] = heldout(ev, units)
        print("126-frame heldout:", json.dumps(out["heldout_126"]), flush=True)
    if args.bench:
        out["graph_bench_reverse" if args.reverse else "graph_bench"] = bench(ev, units, args.reverse)
    write_results(HERE / "results.json", {"mx_fp8": {args.model: {args.units.stem: out}}, "env_mx": env_info()})


if __name__ == "__main__":
    main()
