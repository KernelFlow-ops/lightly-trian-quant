"""Round 7 end-to-end measurement: fused prep/post + eager backbone.

  python run_r7.py --bench --model vitl16 --variants bf16-autocast
  python run_r7.py --acc-only --model vitl16 --variants bf16-autocast
  nsys ... python run_r7.py --profile e2e --model vitl16
Results merge into results.json; variant names get the suffix "+fused-prepost".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

import model_factory as mf  # noqa: E402
import pipeline as pl  # noqa: E402
sys.path.insert(0, str(HERE / 'impl'))
import head as hd  # noqa: E402
from bench import env_info, nvtx, peak_mem_mb, profile_fn, time_fn, write_results  # noqa: E402

OUT = HERE / "results.json"
SUFFIX = "+fused-prepost"
USE_HEAD = False
VARIANTS = {
    "bf16-autocast": (torch.float32, "bf16-mixed"),
    "bf16-pure": (torch.bfloat16, "32-true"),
}


def count_cuda_ops(fn) -> dict[str, int]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    memcpy = sum(1 for n in names if "Memcpy" in n or "memcpy" in n)
    memset = sum(1 for n in names if "Memset" in n or "memset" in n)
    return {"kernels": len(names) - memcpy - memset, "memcpy": memcpy, "memset": memset}


def stage_fns(model, frame, precision):
    with torch.inference_mode():
        batch, meta = pl.fast_prep(frame)
        ml, cl = pl.model_lowres(model, batch, precision)
        batch = batch.clone()
        pl.fast_post(ml, cl, meta)

    def prep():
        return pl.fast_prep(frame)

    def model_only():
        return pl.model_lowres(model, batch, precision)

    def post():
        return pl.fast_post(ml, cl, meta)

    def e2e():
        return pl.fast_e2e(model, frame, precision)

    return {"prep": prep, "model": model_only, "post": post, "e2e": e2e}, batch


def accuracy(model_name, model, precision) -> dict:
    out = {}
    for i, frame in enumerate(mf.load_frames()):
        with torch.inference_mode():
            batch, meta = pl.fast_prep(frame)
            ml, cl = pl.model_lowres(model, batch, precision)
            mask, conf, logits = pl.fast_post(ml, cl, meta, return_logits=True)
        out[f"frame{i}"] = mf.compare_to_ref(model_name, i, logits, mask, conf)
    out["gate"] = mf.gate(out)
    return out


def bench(model_name: str, variant: str, iters: int, warmup: int) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    pl.init(model)
    if USE_HEAD:
        hd.enable(model)
    frame = mf.load_frames()[0]
    fns, batch = stage_fns(model, frame, precision)
    torch.cuda.reset_peak_memory_stats()
    res = {"batch_shape": list(batch.shape), "latency_ms": {}, "cuda_ops": {}}
    for name, fn in fns.items():
        t = time_fn(fn, warmup=warmup, iters=iters)
        res["latency_ms"][name] = {"wall_p50": t["wall_ms_p50"], "gpu_p50": t["gpu_ms_p50"],
                                   "runs": t["runs"]}
        print(f"[{model_name}/{variant}{SUFFIX}] {name:6s} wall p50 {t['wall_ms_p50']:8.3f} ms "
              f"| gpu p50 {t['gpu_ms_p50']:8.3f} ms", flush=True)
    with torch.inference_mode():
        for name in ("prep", "post", "model", "e2e"):
            res["cuda_ops"][name] = count_cuda_ops(fns[name])
    res["peak_mem_mb"] = peak_mem_mb()
    res["fps_e2e"] = 1e3 / res["latency_ms"]["e2e"]["wall_p50"]
    res["accuracy_vs_fp32"] = accuracy(model_name, model, precision)
    print(f"[{model_name}/{variant}{SUFFIX}] fps {res['fps_e2e']:.2f} cuda_ops {res['cuda_ops']} "
          f"gate {res['accuracy_vs_fp32']['gate']}", flush=True)
    write_results(OUT, {model_name: {variant + SUFFIX: res}, "env": env_info()})
    del model
    torch.cuda.empty_cache()


def accuracy_only(model_name: str, variant: str) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    pl.init(model)
    if USE_HEAD:
        hd.enable(model)
    mf.load_frames()
    acc = accuracy(model_name, model, precision)
    print(f"[{model_name}/{variant}{SUFFIX}] gate {acc['gate']}", flush=True)
    write_results(OUT, {model_name: {variant + SUFFIX: {"accuracy_vs_fp32": acc}}})


def profile(model_name: str, variant: str, scope: str) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    pl.init(model)
    if USE_HEAD:
        hd.enable(model)
    frame = mf.load_frames()[0]
    fns, _ = stage_fns(model, frame, precision)

    def e2e_staged():
        with nvtx("prep"):
            batch, meta = pl.fast_prep(frame)
        with nvtx("model"):
            ml, cl = pl.model_lowres(model, batch, precision)
        with nvtx("post"):
            return pl.fast_post(ml, cl, meta)

    profile_fn(e2e_staged if scope == "e2e" else fns[scope], warmup=10, iters=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *mf.WEIGHTS])
    ap.add_argument("--variants", default="bf16-autocast")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--acc-only", action="store_true")
    ap.add_argument("--profile", choices=["model", "post", "prep", "e2e"])
    ap.add_argument("--head", action="store_true", help="P3: fused mask head")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    global SUFFIX, USE_HEAD
    if args.head:
        SUFFIX, USE_HEAD = "+fused-prepost+fused-head", True
    models = list(mf.WEIGHTS) if args.model == "all" else [args.model]
    for m in models:
        for v in args.variants.split(","):
            if args.bench:
                bench(m, v, args.iters, args.warmup)
            if args.acc_only:
                accuracy_only(m, v)
            if args.profile:
                profile(m, v, args.profile)


if __name__ == "__main__":
    main()
