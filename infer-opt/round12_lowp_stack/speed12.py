"""Round 12 speed harness: whole-model speed of any variant vs the baselines.  No accuracy gates.

  python speed12.py --model vitl16 --variants r5_v1,r12+head16,r13_mx8 --plugin /root/lightly-train/infer-opt/round13_x:plug13 --tag t1

Variants: r5_v1 | r11[+flags] | r12[+nopatch][+head16] | anything a plugin registers.
Plugin = "DIR:MODULE" (repeatable).  DIR is put on sys.path; MODULE must define
    PREFIXES = ("r13",)                                    # variant.split("+")[0].split("_")[0] must be in PREFIXES
    make_variant(name, model, variant) -> (model_fn, pipeline_class)
model_fn contract (round 9): model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits), capture-safe, no host sync.

Per variant, candidates interleaved in every round, median over rounds:
  model_ms    model_fn alone, CUDA-graph replay on a real preprocessed frame
  frame_ms    the pipeline's per-slot graph (prep + model + post), replayed back to back on a GPU-resident frame
  fps_gpu     1000 / frame_ms  (GPU-bound throughput: what the video FPS would be without host / decode limits)
  fps_video   full pipeline incl. cv2 decode (round-8 pipe_run, 126 frames x --loops, inflight 2)
  sanity      outputs finite; mask-logit cosine and mean |diff| vs the FIRST variant on eval frame 0 (reported, never gated)
Everything runs under the dual GPU lock.  Results are merged into --out (default: this round's results_speed.json).
"""
from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R11 = HERE.parent / "round11_attn_tier0"
for p in (R11, R11 / "impl", HERE / "impl", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
import r11_common as c  # noqa: E402
import bench12  # noqa: F401,E402  (registers r12)
from bench import env_info, write_results  # noqa: E402
from bench_video import pipe_run  # noqa: E402
from lock2 import gpu_lock2, other_gpu_procs  # noqa: E402

PLUGINS: list = []
_base_make = c.make_variant


def make_variant(name, model, variant):
    key = variant.split("+")[0].split("_")[0]
    for mod in PLUGINS:
        if key in mod.PREFIXES:
            return mod.make_variant(name, model, variant)
    return _base_make(name, model, variant)


c.make_variant = make_variant


def _timed(replay, reps, stream=None):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        replay()
    torch.cuda.synchronize()
    if stream is None:
        a.record()
        for _ in range(reps):
            replay()
        b.record()
    else:
        with torch.cuda.stream(stream):
            a.record(stream)
            for _ in range(reps):
                replay()
            b.record(stream)
    b.synchronize()
    return a.elapsed_time(b) / reps


def capture(fn, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(x)
    g.replay()
    torch.cuda.synchronize()
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(c.mf.WEIGHTS))
    ap.add_argument("--variants", required=True)
    ap.add_argument("--plugin", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(HERE / "results_speed.json"))
    a = ap.parse_args()
    for spec in a.plugin:
        d, m = spec.split(":")
        if d not in sys.path:
            sys.path.insert(0, d)
        PLUGINS.append(importlib.import_module(m))
    variants = a.variants.split(",")
    res = {v: {} for v in variants}
    with gpu_lock2(), torch.inference_mode():
        print("other gpu procs at start:", other_gpu_procs(), flush=True)
        frame = c.mf.load_frames()[0]
        built = {}
        for v in variants:
            t0 = time.time()
            model = c.mf.load_model(a.model)
            fn, pcls = make_variant(a.model, model, v)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xb, _ = c.mf.base_prep(model, frame)
            x = xb.to(torch.bfloat16).contiguous()
            g, out = capture(fn, x)
            pipe = c.make_pipeline(model, fn, pcls)
            pipe.pinned_np[0][...] = frame
            pipe.frame_gpu[0].copy_(pipe.pinned[0])
            torch.cuda.synchronize()
            built[v] = (g, out, pipe)
            print(f"built {v} in {time.time() - t0:.0f}s", flush=True)
        ref = built[variants[0]][1][0].float()
        for v in variants:
            ml, cl = built[v][1]
            mlf = ml.float()
            res[v]["sanity"] = {
                "finite": bool(torch.isfinite(mlf).all() and torch.isfinite(cl.float()).all()),
                "ml_cos_vs_first": torch.nn.functional.cosine_similarity(mlf.flatten().double(), ref.flatten().double(), dim=0).item(),
                "ml_mean_abs_diff_vs_first": (mlf - ref).abs().mean().item(), "ml_mean_abs": mlf.abs().mean().item()}
        acc = {v: {"model_ms": [], "frame_ms": [], "fps_video": []} for v in variants}
        for r in range(a.rounds):
            for v in (variants if r % 2 == 0 else variants[::-1]):
                g, _, pipe = built[v]
                acc[v]["model_ms"].append(_timed(g.replay, a.reps))
                acc[v]["frame_ms"].append(_timed(pipe.graphs[0].replay, a.reps, pipe.comp))
                if not a.no_video:
                    pipe.inflight = 2
                    pipe_run(pipe, 1)
                    r2, _, _ = pipe_run(pipe, a.loops)
                    acc[v]["fps_video"].append(r2["fps_steady"] or r2["fps_overall"])
                print(f"round {r} {v}: model {acc[v]['model_ms'][-1]:.3f} ms  frame {acc[v]['frame_ms'][-1]:.3f} ms"
                      + ("" if a.no_video else f"  video {acc[v]['fps_video'][-1]:.1f} FPS"), flush=True)
        b = {k: statistics.median(acc[variants[0]][k]) for k in ("model_ms", "frame_ms")}
        for v in variants:
            m, f = statistics.median(acc[v]["model_ms"]), statistics.median(acc[v]["frame_ms"])
            res[v].update(model_ms=m, frame_ms=f, fps_gpu=1000.0 / f, model_speedup_vs_first=b["model_ms"] / m,
                          frame_speedup_vs_first=b["frame_ms"] / f, rounds=acc[v])
            if not a.no_video:
                res[v]["fps_video"] = statistics.median(acc[v]["fps_video"])
        print("other gpu procs at end:", other_gpu_procs(), flush=True)
    print(f"\n| variant | model ms | x | frame ms | x | GPU-bound FPS | video FPS | finite | ml cos vs {variants[0]} |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for v in variants:
        d = res[v]
        print(f"| {v} | {d['model_ms']:.3f} | {d['model_speedup_vs_first']:.2f} | {d['frame_ms']:.3f} | {d['frame_speedup_vs_first']:.2f} | "
              f"{d['fps_gpu']:.1f} | {d.get('fps_video', float('nan')):.1f} | {d['sanity']['finite']} | {d['sanity']['ml_cos_vs_first']:.6f} |")
    key = f"{a.model}|{','.join(variants)}|{a.tag or time.strftime('%m%d_%H%M')}"
    write_results(a.out, {"speed": {key: {"results": res, "rounds": a.rounds, "reps": a.reps, "loops": a.loops, "env": env_info()}}})


if __name__ == "__main__":
    main()
