"""Round 0: PyTorch eager baseline for DINOv3-EoMT semantic head segmentation.

Examples:
  python round0_baseline/run.py --make-ref --model all
  python round0_baseline/run.py --bench --model all --variants fp32,bf16-autocast,bf16-pure
  nsys profile ... python round0_baseline/run.py --profile model --model vitl16 --variants bf16-autocast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import model_factory as mf  # noqa: E402
from bench import env_info, nvtx, peak_mem_mb, profile_fn, time_fn, write_results  # noqa: E402

OUT = Path(__file__).resolve().parent / "results.json"
VARIANTS = {
    # name: (weight dtype, autocast precision)
    "fp32": (torch.float32, "32-true"),
    "bf16-autocast": (torch.float32, "bf16-mixed"),  # == scripts/infer_*_video.py default
    "bf16-pure": (torch.bfloat16, "32-true"),
}


def add_block_nvtx(model) -> None:
    """forward_train calls block.norm1/attn/ls2 directly, so hook the submodules.

    Hooks must return None: a non-None return value replaces the module input/output.
    """

    def push(name):
        def hook(*_):
            torch.cuda.nvtx.range_push(name)

        return hook

    def pop(*_):
        torch.cuda.nvtx.range_pop()

    for i, blk in enumerate(model.backbone.blocks):
        blk.norm1.register_forward_pre_hook(push(f"block_{i:02d}"))
        blk.ls2.register_forward_hook(pop)
    model.upscale.register_forward_pre_hook(push("head.upscale"))
    model.upscale.register_forward_hook(pop)


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
    ctx = lambda: mf.autocast_ctx(precision)  # noqa: E731
    with torch.inference_mode(), ctx():
        batch, meta = mf.base_prep(model, frame)
        raw = mf.base_model(model, batch)

    def prep():
        with ctx():
            return mf.base_prep(model, frame)

    def model_only():
        with ctx():
            return mf.base_model(model, batch)

    def post():
        with ctx():
            return mf.base_post(model, raw, meta)

    def e2e():
        return mf.base_e2e(model, frame, precision)

    return {"prep": prep, "model": model_only, "post": post, "e2e": e2e}, batch


def accuracy(model_name, model, precision) -> dict:
    out = {}
    for i, frame in enumerate(mf.load_frames()):
        with torch.inference_mode(), mf.autocast_ctx(precision):
            batch, meta = mf.base_prep(model, frame)
            raw = mf.base_model(model, batch)
            logits = mf.base_post_logits(model, raw, meta)
        mask, _, conf = mf.host_finalize(model, logits.float().cpu().numpy())
        out[f"frame{i}"] = mf.compare_to_ref(model_name, i, logits, mask, conf)
    out["gate"] = mf.gate(out)
    return out


def bench(model_name: str, variant: str, iters: int, warmup: int) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    frame = mf.load_frames()[0]
    fns, batch = stage_fns(model, frame, precision)
    torch.cuda.reset_peak_memory_stats()
    res = {"batch_shape": list(batch.shape), "latency_ms": {}, "cuda_ops": {}}
    for name, fn in fns.items():
        t = time_fn(fn, warmup=warmup, iters=iters)
        res["latency_ms"][name] = {"wall_p50": t["wall_ms_p50"], "gpu_p50": t["gpu_ms_p50"],
                                   "runs": t["runs"]}
        print(f"[{model_name}/{variant}] {name:6s} wall p50 {t['wall_ms_p50']:8.3f} ms "
              f"| gpu p50 {t['gpu_ms_p50']:8.3f} ms", flush=True)
    with torch.inference_mode():
        for name in ("model", "e2e"):
            res["cuda_ops"][name] = count_cuda_ops(fns[name])
    res["peak_mem_mb"] = peak_mem_mb()
    res["fps_e2e"] = 1e3 / res["latency_ms"]["e2e"]["wall_p50"]
    res["accuracy_vs_fp32"] = accuracy(model_name, model, precision)
    print(f"[{model_name}/{variant}] cuda_ops {res['cuda_ops']}  gate "
          f"{res['accuracy_vs_fp32']['gate']}", flush=True)
    write_results(OUT, {model_name: {variant: res}, "env": env_info()})
    del model
    torch.cuda.empty_cache()


def accuracy_only(model_name: str, variant: str) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    acc = accuracy(model_name, model, precision)
    print(f"[{model_name}/{variant}] gate {acc['gate']}", flush=True)
    write_results(OUT, {model_name: {variant: {"accuracy_vs_fp32": acc}}})
    del model
    torch.cuda.empty_cache()


def profile(model_name: str, variant: str, scope: str) -> None:
    dtype, precision = VARIANTS[variant]
    model = mf.load_model(model_name, dtype)
    add_block_nvtx(model)
    frame = mf.load_frames()[0]
    fns, _ = stage_fns(model, frame, precision)

    def e2e_staged():
        with mf.autocast_ctx(precision):
            with nvtx("prep"):
                batch, meta = mf.base_prep(model, frame)
            with nvtx("model"):
                raw = mf.base_model(model, batch)
            with nvtx("post"):
                return mf.base_post(model, raw, meta)

    profile_fn(e2e_staged if scope == "e2e" else fns[scope], warmup=10, iters=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *mf.WEIGHTS])
    ap.add_argument("--variants", default="fp32,bf16-autocast,bf16-pure")
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--acc-only", action="store_true")
    ap.add_argument("--profile", choices=["model", "post", "prep", "e2e"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    models = list(mf.WEIGHTS) if args.model == "all" else [args.model]
    variants = args.variants.split(",")
    for m in models:
        if args.make_ref:
            mf.make_reference(m)
            print(f"reference written for {m}", flush=True)
        for v in variants:
            if args.bench:
                bench(m, v, args.iters, args.warmup)
            if args.acc_only:
                accuracy_only(m, v)
            if args.profile:
                profile(m, v, args.profile)


if __name__ == "__main__":
    main()
