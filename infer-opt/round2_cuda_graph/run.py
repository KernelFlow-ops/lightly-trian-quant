"""Round 2: CUDA Graph + torch.compile on OptEoMT.

  python round2_cuda_graph/run.py --model vitl16 --variants opt-cudagraph,compile-max-autotune-no-cudagraphs,...
  python round2_cuda_graph/run.py --profile model --mode graph|eager --model vitl16   # nsys via common/profile_nsys.sh
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "round1_torch_cleanup"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R1))
import harness  # noqa: E402
import model_factory as mf  # noqa: E402
from bench import nvtx, profile_fn  # noqa: E402
from graphed import GraphedModel  # noqa: E402
from opt_model import OptEoMT  # noqa: E402

OUT = HERE / "results.json"
SHAPE = (2, 3, 512, 512)


def make_fn(opt: OptEoMT, variant: str):
    """Returns (model_fn(batch)->(mask512, cls), extra_fns, info)."""
    info = {}
    if variant == "opt-cudagraph":
        t0 = time.perf_counter()
        g = GraphedModel(opt.forward_backend, SHAPE)
        info["capture_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        gl = GraphedModel(opt.forward_lowres, SHAPE)
        info["capture_lowres_s"] = time.perf_counter() - t0
        b0 = torch.zeros(SHAPE, device="cuda")
        return g, {"model_lowres": lambda: gl(b0)}, info
    if variant == "opt-eager":
        b0 = torch.zeros(SHAPE, device="cuda")
        return opt.forward_backend, {"model_lowres": lambda: opt.forward_lowres(b0)}, info
    if variant.startswith("compile-"):
        mode = variant[len("compile-"):]
        torch._dynamo.reset()
        fn = torch.compile(opt.forward_backend, mode=mode, fullgraph=True)
        b0 = torch.zeros(SHAPE, device="cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            fn(b0)
            torch.cuda.synchronize()
        info["compile_s_first_call"] = time.perf_counter() - t0
        return fn, {}, info
    raise KeyError(variant)


def profile(name: str, mode: str, sdpa: str) -> None:
    ref = mf.load_model(name, torch.float32)
    opt = OptEoMT(ref, sdpa_backend=sdpa)
    frame = mf.load_frames()[0]
    with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
        batch, meta = mf.base_prep(ref, frame)
    fn, _, _ = make_fn(opt, "opt-cudagraph" if mode == "graph" else "opt-eager")

    def e2e_staged():
        with nvtx("prep"), mf.autocast_ctx("bf16-mixed"):
            b, mt = mf.base_prep(ref, frame)
        with nvtx("model"):
            raw = fn(b)
        with nvtx("post"), mf.autocast_ctx("bf16-mixed"):
            return mf.base_post(ref, raw, mt)

    profile_fn(e2e_staged, warmup=10, iters=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *mf.WEIGHTS])
    ap.add_argument("--variants", default="opt-cudagraph")
    ap.add_argument("--sdpa", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--profile", choices=["e2e"])
    ap.add_argument("--mode", default="graph", choices=["graph", "eager"])
    a = ap.parse_args()
    models = list(mf.WEIGHTS) if a.model == "all" else [a.model]
    if a.profile:
        profile(models[0], a.mode, a.sdpa)
        return
    for name in models:
        ref = mf.load_model(name, torch.float32)
        opt = OptEoMT(ref, sdpa_backend=a.sdpa)
        for v in a.variants.split(","):
            fn, extra_fns, info = make_fn(opt, v)
            print(f"[{name}/{v}] setup info {info}", flush=True)
            harness.bench_variant(name, v + a.tag, ref, fn, OUT, iters=a.iters, warmup=a.warmup,
                                  extra_fns=extra_fns,
                                  extra={"config": {"sdpa_backend": a.sdpa}, "setup": info})
        del ref, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
