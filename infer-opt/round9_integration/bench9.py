"""Round 9 measurements: whole-frame-graph pipeline with pluggable model_fn on the full test video.

  python bench9.py --model vitl16 --variants lightly,lightly+fp8,opt,fused
  ../common/profile_nsys.sh <prefix> bench9.py --model vitl16 --profile fused
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.append(str(HERE.parent / "round8_pipeline"))
import model_factory as mf  # noqa: E402
from bench import env_info, gpu_lock, write_results  # noqa: E402
import video_infer as vi  # noqa: E402  (round9 version)
from bench_video import (NFRAMES, VIDEO, baseline_loop, compare, fp32_gate, pipe_run)  # noqa: E402  (round8 helpers)
import model_impls as mi  # noqa: E402

OUT = HERE / "results.json"
NAMES = {
    "lightly": "lightly+graph+fused-prepost+fused-head",
    "lightly+fp8": "lightly+graph+fused-prepost+fused-head+fp8",
    "opt": "opt(graphopt)+graph+fused-prepost",
    "opt_gemmhead": "opt(graphopt,gemm-head)+graph+fused-prepost",
    "opt_tl": "opt(graphopt,tl-attention)+graph+fused-prepost",
    "opt_tl_rope": "opt(graphopt,tl-rope-attention)+graph+fused-prepost",
    "fused": "fused-block+graph+fused-prepost+fused-head",
    "fused_tl": "fused-block+tl-attention+graph+fused-prepost+fused-head",
    "r5_v1": "round5-v1-tuned+graph+fused-prepost+fused-head",
}


def build(name, variant):
    model = mf.load_model(name)
    impl, fp8 = (variant.split("+")[0], variant.endswith("+fp8"))
    fn = mi.make_model_fn(name, model, impl, fp8)
    return model, fn


def graph_vs_eager(model, fn):
    """Element-wise: model_fn under CUDA graph replay vs eager (same static input)."""
    x = torch.randn(2, 3, 512, 512, device="cuda").bfloat16()
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ref = fn(x)
        torch.cuda.current_stream().wait_stream(s)
        ref = [t.clone() for t in fn(x)]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(x)
        g.replay()
        torch.cuda.synchronize()
    r = {}
    for k, a, b in (("mask_logits", out[0], ref[0]), ("class_logits", out[1], ref[1])):
        d = (a.float() - b.float()).abs()
        r[k] = {"max_abs": d.max().item(), "identical": bool(torch.equal(a, b))}
    return r


def run_variant(name, variant, loops, bmasks, bconfs, base_res):
    with gpu_lock():
        _run_variant_locked(name, variant, loops, bmasks, bconfs, base_res)


def _run_variant_locked(name, variant, loops, bmasks, bconfs, base_res):
    print(f"=== {name} / {variant}", flush=True)
    model, fn = build(name, variant)
    res = {"impl": variant}
    try:
        res["graph_vs_eager_model_fn"] = graph_vs_eager(model, fn)
        print("graph vs eager:", res["graph_vs_eager_model_fn"], flush=True)
    except Exception:  # noqa: BLE001
        res["graph_vs_eager_model_fn"] = "ERROR " + traceback.format_exc()[-400:]
        print(res["graph_vs_eager_model_fn"], flush=True)
    with torch.inference_mode():
        pipe = vi.VideoPipeline(model, fn, 1440, 1900, slots=4, inflight=2)
        pipe_run(pipe, 1)
        r2, pm, pc = pipe_run(pipe, loops, collect=True)
        print("inflight=2", r2, flush=True)
        pipe.inflight = 1
        r1, _, _ = pipe_run(pipe, loops)
        print("inflight=1", r1, flush=True)
        pipe.inflight = 3
        r3, _, _ = pipe_run(pipe, loops)
        pipe.inflight = 2
        # graph replay vs non-graph pipeline: identical masks?
        pipe.use_graph = False
        _, pm_e, pc_e = pipe_run(pipe, 1, collect=True)
        pipe.use_graph = True
    same = sum(int(np.array_equal(pm[i], pm_e[i])) for i in range(NFRAMES))
    res["pipeline_graph_vs_nograph"] = {"identical_masks": same, "frames": NFRAMES,
                                        "max_conf_diff": max(abs(pc[i] - pc_e[i]) for i in range(NFRAMES))}
    acc = compare(bmasks, bconfs, pm, pc)
    gate = fp32_gate(name, pm, pc)
    print("vs baseline:", acc, "\ngate:", gate["gate"], flush=True)
    fps = r2["fps_steady"] or r2["fps_overall"]
    payload = {
        "latency_ms": {"e2e": {"wall_p50": r1["service_ms_p50"], "wall_p90": r1["service_ms_p90"]}},
        "fps_e2e": fps,
        "accuracy_vs_fp32": gate,
        "video": {"baseline": base_res, "pipeline_inflight2": r2, "pipeline_inflight1": r1, "pipeline_inflight3": r3,
                  "accuracy_vs_baseline_126f": acc, "speedup_fps_vs_baseline": fps / base_res["fps"],
                  **{k: v for k, v in res.items()},
                  "note": "latency_ms.e2e = submit->host result, inflight=1; throughput = steady state, inflight=2"},
    }
    write_results(OUT, {name: {NAMES[variant]: payload}, "env": env_info()})
    del pipe, model, fn
    torch.cuda.empty_cache()


def profile(name, variant):
    with gpu_lock(), torch.inference_mode():
        model, fn = build(name, variant)
        pipe = vi.VideoPipeline(model, fn, 1440, 1900, slots=4, inflight=2)
        pipe.run([VIDEO], lambda *a: False, loops=1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        pipe.run([VIDEO], lambda *a: False, loops=1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mf.WEIGHTS))
    ap.add_argument("--variants", default="lightly,lightly+fp8,opt,fused")
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    if args.profile:
        profile(args.model, args.profile)
        return
    orig = vi.load_orig_script()
    with gpu_lock():
        m0 = mf.load_model(args.model)
        baseline_loop(orig, m0, 1, False)
        base_res, bmasks, bconfs = baseline_loop(orig, m0, 2, True)
        del m0
        torch.cuda.empty_cache()
    print("baseline", base_res, flush=True)
    for v in args.variants.split(","):
        try:
            run_variant(args.model, v, args.loops, bmasks, bconfs, base_res)
        except Exception:  # noqa: BLE001
            print(f"VARIANT {v} FAILED:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
