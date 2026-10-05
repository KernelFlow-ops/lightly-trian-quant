"""Round 1: OptEoMT (eager, bf16-resident) benchmark, sdpa backend + head sweeps.

  python round1_torch_cleanup/run.py --model vitl16 --variants opt-eager-bf16,...
  python round1_torch_cleanup/run.py --profile model --model vitl16   # for nsys (profile_nsys.sh)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import model_factory as mf  # noqa: E402
from bench import nvtx, profile_fn  # noqa: E402
from opt_model import OptEoMT  # noqa: E402

OUT = HERE / "results.json"
CFG = {
    "opt-eager-bf16": dict(sdpa_backend="cudnn"),  # primary: cudnn SDPA (passes gate on ViT-L), fp32 residual stream
    "opt-eager-bf16-sdpa-auto": dict(),  # torch picks flash (fails ViT-L gate)
    "opt-eager-bf16-res_bf16-sdpa-auto": dict(residual_fp32=False),
    "opt-eager-bf16-sdpa-flash": dict(sdpa_backend="flash"),
    "opt-eager-bf16-sdpa-efficient": dict(sdpa_backend="efficient"),
    "opt-eager-bf16-sdpa-cudnn": dict(sdpa_backend="cudnn"),
    "opt-eager-bf16-sdpa-math": dict(sdpa_backend="math"),
    "opt-eager-bf16-head-channels_last": dict(head_channels_last=True),
    "opt-eager-bf16-head-convt_gemm": dict(head_convt_as_gemm=True),
}


def profile(name: str, variant: str, scope: str) -> None:
    ref = mf.load_model(name, torch.float32)
    opt = OptEoMT(ref, **CFG[variant])
    frame = mf.load_frames()[0]
    with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
        batch, meta = mf.base_prep(ref, frame)

    def model_only():
        return opt.forward_backend(batch)

    def e2e_staged():
        with nvtx("prep"), mf.autocast_ctx("bf16-mixed"):
            b, mt = mf.base_prep(ref, frame)
        with nvtx("model"):
            raw = opt.forward_backend(b)
        with nvtx("post"), mf.autocast_ctx("bf16-mixed"):
            return mf.base_post(ref, raw, mt)

    profile_fn(e2e_staged if scope == "e2e" else model_only, warmup=10, iters=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *mf.WEIGHTS])
    ap.add_argument("--variants", default=",".join(CFG))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--profile", choices=["model", "e2e"])
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    models = list(mf.WEIGHTS) if a.model == "all" else [a.model]
    if a.profile:
        profile(models[0], a.variants.split(",")[0], a.profile)
        return
    for name in models:
        ref = mf.load_model(name, torch.float32)
        for v in a.variants.split(","):
            cfg = dict(CFG[v])
            opt = OptEoMT(ref, **cfg)
            suffix = "-cudnnbench" if a.cudnn_benchmark else ""
            harness.bench_variant(
                name, v + suffix, ref, opt.forward_backend, OUT, iters=a.iters, warmup=a.warmup,
                extra={"config": {**{k: str(x) for k, x in cfg.items()},
                                  "cudnn_benchmark": a.cudnn_benchmark}})
            del opt
        del ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
