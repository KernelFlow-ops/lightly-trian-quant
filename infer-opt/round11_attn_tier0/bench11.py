"""Round 11 same-session paired whole-video bench + gates M / H / G  (adapted from round9_integration/bench9.py).

  python bench11.py --model vitl16 --variants r5_v1,r11 --loops 3 --rounds 2            # perf A-B-A-B + gates
  python bench11.py --model vitl16 --variants r5_v1,r11 --gates-only
  python bench11.py --model vitl16 --variants r5_v1,r11 --perf-only --rounds 3
The first variant is the reference of the paired comparison.  Everything runs under the dual GPU lock (<= 10 min: use few variants/job).
Gate definitions (PLAN_v2 sec 2.3):  M = common/model_factory.gate on the 8 eval frames vs cached fp32 eager refs;
H = 126 frames vs fp32 eager (ref126.py): pooled head IoU >= 0.95 and every-frame conf rel diff <= 5% (+ min-frame IoU, #frames conf > 3%,
mask-score MAE);  G = model_fn CUDA-graph replay == eager (bitwise) and pipeline graph vs non-graph masks identical.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs
from bench import env_info, write_results
from bench_video import NFRAMES, VIDEO, EVAL_IDX, fp32_gate, pipe_run
import ref126 as r126

OUT = c.HERE / "results.json"


def graph_vs_eager(fn):
    x = torch.randn(2, 3, 512, 512, device="cuda").bfloat16()
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        ref = [t.clone() for t in fn(x)]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(x)
        g.replay()
        torch.cuda.synchronize()
    return {k: {"max_abs": (a.float() - b.float()).abs().max().item(), "identical": bool(torch.equal(a, b))}
            for k, a, b in (("mask_logits", out[0], ref[0]), ("class_logits", out[1], ref[1]))}


def gate_h(name, pm, pc, ref):
    """pm/pc: per-frame uint8 masks / conf of the pipeline run vs the 126-frame fp32 reference."""
    inter = union = 0
    ious, crel = [], []
    for i in range(NFRAMES):
        rb = np.unpackbits(ref["head_bits"][i])[: pm[i].size].reshape(pm[i].shape).astype(bool)
        pb = pm[i] == 1
        it, un = int((rb & pb).sum()), int((rb | pb).sum())
        inter += it
        union += un
        ious.append(it / un if un else 1.0)
        crel.append(abs(pc[i] - ref["conf"][i]) / abs(ref["conf"][i]))
    crel = np.array(crel)
    pooled = inter / union
    return {"pooled_iou": pooled, "min_frame_iou": float(min(ious)), "conf_rel_max": float(crel.max()),
            "conf_rel_p99": float(np.percentile(crel, 99)), "frames_conf_gt3pct": int((crel > 0.03).sum()),
            "frames_conf_gt5pct": int((crel > 0.05).sum()),
            "pass": bool(pooled >= 0.95 and crel.max() <= 0.05)}


def score_mae(pipe, frames, ref_lr):
    """mean |lr - ref_lr| (fp32 eager reference mask-score maps) over all frames, synchronous graph replays."""
    tot, mx = 0.0, 0.0
    with torch.inference_mode():
        for i, f in enumerate(frames):
            pipe.pinned_np[0][...] = f
            pipe.frame_gpu[0].copy_(pipe.pinned[0])
            pipe.graphs[0].replay()
            torch.cuda.synchronize()
            d = (pipe.lr - ref_lr[i].cuda()).abs()
            tot += d.mean().item()
            mx = max(mx, d.max().item())
    return {"score_mae": tot / len(frames), "score_max_abs": mx}


def build(name, variant):
    model = c.mf.load_model(name)
    fn, pcls = c.make_variant(name, model, variant)
    pipe = c.make_pipeline(model, fn, pcls)
    return model, fn, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(c.mf.WEIGHTS))
    ap.add_argument("--variants", default="r5_v1,r11")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--perf-only", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    variants = a.variants.split(",")
    frames = r126.decode_all()
    ref = torch.load(r126.ref_path(a.model), weights_only=False)
    ref_lr = ref["lr"]
    res = {}
    with gpu_lock2(), torch.inference_mode():
        print("other gpu procs at start:", other_gpu_procs(), flush=True)
        built = {}
        for v in variants:
            t0 = time.time()
            built[v] = build(a.model, v)
            print(f"built {v} in {time.time() - t0:.0f}s", flush=True)
        for v in variants:
            res[v] = {}
        if not a.perf_only:
            for v in variants:
                model, fn, pipe = built[v]
                print(f"=== gates {a.model} / {v}", flush=True)
                res[v]["G_model_fn_graph_vs_eager"] = graph_vs_eager(fn)
                pipe_run(pipe, 1)
                _, pm, pc = pipe_run(pipe, 1, collect=True)
                pipe.use_graph = False
                _, pm_e, pc_e = pipe_run(pipe, 1, collect=True)
                pipe.use_graph = True
                res[v]["G_pipeline_graph_vs_nograph"] = {
                    "identical_masks": sum(int(np.array_equal(pm[i], pm_e[i])) for i in range(NFRAMES)), "frames": NFRAMES,
                    "max_conf_diff": max(abs(pc[i] - pc_e[i]) for i in range(NFRAMES))}
                gm = fp32_gate(a.model, pm, pc)
                res[v]["M_8frame"] = gm["gate"]
                res[v]["H_126frame"] = gate_h(a.model, pm, pc, ref)
                res[v]["H_126frame"].update(score_mae(pipe, frames, ref_lr))
                print(json.dumps({k: res[v][k] for k in ("G_model_fn_graph_vs_eager", "G_pipeline_graph_vs_nograph", "M_8frame", "H_126frame")}, default=float), flush=True)
        if not a.gates_only:
            fps = {v: [] for v in variants}
            lat = {v: [] for v in variants}
            for r in range(a.rounds):
                for v in variants:
                    pipe = built[v][2]
                    pipe.inflight = 2
                    pipe_run(pipe, 1)
                    r2, _, _ = pipe_run(pipe, a.loops)
                    fps[v].append(r2["fps_steady"] or r2["fps_overall"])
                    lat[v].append(r2["latency_ms_p50"])
                    print(f"round {r} {v}: {fps[v][-1]:.2f} FPS  lat p50 {lat[v][-1]:.2f} ms", flush=True)
            # service latency, inflight=1
            for v in variants:
                pipe = built[v][2]
                pipe.inflight = 1
                r1, _, _ = pipe_run(pipe, 1)
                res[v]["service_ms_p50_inflight1"] = r1["service_ms_p50"]
                res[v]["service_ms_p90_inflight1"] = r1["service_ms_p90"]
                pipe.inflight = 2
            base = statistics.median(fps[variants[0]])
            for v in variants:
                res[v]["fps_rounds"] = fps[v]
                res[v]["fps_median"] = statistics.median(fps[v])
                res[v]["fps_vs_first_pct"] = 100 * (statistics.median(fps[v]) / base - 1)
                res[v]["latency_ms_p50_inflight2"] = statistics.median(lat[v])
            print("PAIRED:", {v: (round(res[v]["fps_median"], 2), round(res[v]["fps_vs_first_pct"], 2)) for v in variants}, flush=True)
        print("other gpu procs at end:", other_gpu_procs(), flush=True)
    key = f"{a.model}|{','.join(variants)}|{a.tag or time.strftime('%m%d_%H%M')}"
    write_results(OUT, {"bench11": {key: {"results": res, "loops": a.loops, "rounds": a.rounds, "env": env_info()}}})


if __name__ == "__main__":
    main()
