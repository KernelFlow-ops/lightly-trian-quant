"""Round 2: paired eager-vs-CUDA-graph timing inside one process (A/B interleaved, several rounds).

Other workers run un-locked correctness jobs on the same GPU, which inflates any single timing loop;
alternating eager/graph rounds and reporting per-round numbers + min/median makes the comparison fair.
Stages: model (forward_backend on the baseline-prep batch), model_lowres, e2e (baseline prep + model + baseline post).
"""
import argparse, json, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "round1_torch_cleanup"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R1))
import torch
import harness, model_factory as mf
from bench import gpu_lock, time_fn, write_results, env_info
from graphed import GraphedModel
from opt_model import OptEoMT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="all"); ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--sdpa", default="cudnn"); ap.add_argument("--attn", default="sdpa")
ap.add_argument("--tag", default="")
a = ap.parse_args()
models = list(mf.WEIGHTS) if a.model == "all" else [a.model]
out = {}
for name in models:
    ref = mf.load_model(name, torch.float32)
    opt = OptEoMT(ref, sdpa_backend=a.sdpa, attn_impl=a.attn)
    g = GraphedModel(opt.forward_backend, (2, 3, 512, 512))
    gl = GraphedModel(opt.forward_lowres, (2, 3, 512, 512))
    frame = mf.load_frames()[0]
    with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
        batch, meta = mf.base_prep(ref, frame)
    b0 = batch
    def e2e(fn):
        def f():
            with torch.inference_mode():
                with mf.autocast_ctx("bf16-mixed"):
                    bb, mt = mf.base_prep(ref, frame)
                r = fn(bb)
                with mf.autocast_ctx("bf16-mixed"):
                    return mf.base_post(ref, r, mt)
        return f
    stages = {
        "model": (lambda: opt.forward_backend(b0), lambda: g(b0)),
        "model_lowres": (lambda: opt.forward_lowres(b0), lambda: gl(b0)),
        "e2e": (e2e(opt.forward_backend), e2e(g)),
    }
    res = {}
    for st, (fe, fg) in stages.items():
        rows = {"eager": [], "graph": []}
        for r in range(a.rounds):
            for k, fn in (("eager", fe), ("graph", fg)):
                t = time_fn(fn, warmup=10, iters=40, repeats=1)
                rows[k].append({"wall_p50": t["wall_ms_p50"], "gpu_p50": t["gpu_ms_p50"]})
        res[st] = {k: {"per_round_wall_p50": [x["wall_p50"] for x in v],
                       "per_round_gpu_p50": [x["gpu_p50"] for x in v],
                       "wall_p50_min": min(x["wall_p50"] for x in v), "wall_p50_median": statistics.median(x["wall_p50"] for x in v),
                       "gpu_p50_min": min(x["gpu_p50"] for x in v)} for k, v in rows.items()}
        e, gg = res[st]["eager"], res[st]["graph"]
        print(f"[{name}] {st:13s} eager wall min/med {e['wall_p50_min']:.3f}/{e['wall_p50_median']:.3f} ms | "
              f"graph wall min/med {gg['wall_p50_min']:.3f}/{gg['wall_p50_median']:.3f} ms | graph gpu min {gg['gpu_p50_min']:.3f}", flush=True)
    write_results(HERE / "results.json", {name: {f"paired-eager-vs-graph{a.tag}": {"sdpa": a.sdpa, "attn": a.attn, "stages": res}}, "env": env_info()})
    del ref, opt, g, gl
    torch.cuda.empty_cache()
