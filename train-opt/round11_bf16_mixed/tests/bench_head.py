"""Head (``_predict``) candidate matrix under bf16 autocast at the real shapes.

Candidates: source torch path, channels_last (r1), bf16 LayerNorm2D, both, and the round6 variants.
Timing is fwd+bwd of one ``predict`` call (the step has four per step); accuracy is cosine/rel-L2 of
the outputs and of d/dx plus all parameter gradients against the fp32 (no autocast) source path.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.impl import head as head_impl  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare, cosine, fmt  # noqa: E402

MODELS = {"vitl16": (7, 1024), "vitb16": (14, 768)}
NQ, STORAGE, GRID = 100, 4, (32, 32)
TOKENS = NQ + 1 + STORAGE + GRID[0] * GRID[1]


def build_head(dim: int):
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scale_block import ScaleBlock

    torch.manual_seed(0)
    return SimpleNamespace(
        num_queries=NQ,
        backbone=SimpleNamespace(norm=nn.LayerNorm(dim, eps=1e-5).cuda(), n_storage_tokens=STORAGE),
        class_head=nn.Linear(dim, 3).cuda(),
        mask_head=nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).cuda(),
        upscale=nn.Sequential(ScaleBlock(dim), ScaleBlock(dim)).cuda(),
    )


def modules_of(ns):
    return [ns.backbone.norm, ns.class_head, ns.mask_head, ns.upscale]


def params_of(ns):
    return [p for m in modules_of(ns) for p in m.parameters()]


def source_predict(ns, x):
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import DINOv3EoMTSemanticSegmentation as M

    return M._predict(ns, ns.backbone.norm(x), GRID)


def make_candidates(base) -> dict:
    from round6_head_fusion.impl.fused_head import predict as r6_predict

    cands: dict = {}

    def add(name, installs=(), r6=None):
        ns = copy.deepcopy(base)
        ctx = SimpleNamespace(model=ns)
        for install in installs:
            install(ctx)
        if r6 is None:
            cands[name] = (ns, lambda x, ns=ns: source_predict(ns, x))
        else:
            cands[name] = (ns, lambda x, ns=ns, r6=r6: r6_predict(ns, x, GRID, r6))

    add("default(torch)")
    add("head_cl", [head_impl.install_head_cl])
    add("ln_bf16", [head_impl.install_ln2d_bf16])
    add("head_cl+ln_bf16", [head_impl.install_head_cl, head_impl.install_ln2d_bf16])
    for variant in ("channels_last_bmm", "convt_gemm", "dwln", "all", "fused_dwln"):
        add(f"r6:{variant}", r6=variant)
    return cands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    report: dict = {"torch": torch.__version__, "models": {}}
    with gpu_lock():
        for model, (batch, dim) in MODELS.items():
            if model not in args.models:
                continue
            base = build_head(dim)
            cands = make_candidates(base)
            gen = torch.Generator(device="cuda").manual_seed(1)
            x = (torch.randn(batch, TOKENS, dim, device="cuda", generator=gen) * 2.0).requires_grad_(True)
            go_m = torch.randn(batch, NQ, 128, 128, device="cuda", generator=gen).to(torch.bfloat16)
            go_c = torch.randn(batch, NQ, 3, device="cuda", generator=gen)

            def fwdbwd(ns, fn):
                def run():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        mask_logits, class_logits = fn(x)
                    torch.autograd.grad([mask_logits, class_logits], [x, *params_of(ns)], [go_m.to(mask_logits.dtype), go_c.to(class_logits.dtype)], allow_unused=True)
                return run

            def fwd_only(ns, fn):
                def run():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        fn(x)
                return run

            timed = compare({k: fwdbwd(*v) for k, v in cands.items()}, iters=20)
            timed_fwd = compare({k: fwd_only(*v) for k, v in cands.items()}, iters=20)
            print(f"=== {model} B={batch} d={dim}: head fwd+bwd (one predict)", flush=True)
            print(fmt(timed), flush=True)
            print(f"=== {model}: head fwd only", flush=True)
            print(fmt(timed_fwd), flush=True)

            # accuracy against fp32 source path (same weights, no autocast)
            ref_ns = copy.deepcopy(base)
            xr = x.detach().clone().requires_grad_(True)
            mr, cr = source_predict(ref_ns, xr)
            gr = torch.autograd.grad([mr, cr], [xr, *params_of(ref_ns)], [go_m.float(), go_c], allow_unused=True)
            accuracy: dict = {}
            for name, (ns, fn) in cands.items():
                if "error" in timed[name]:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m, c = fn(x)
                grads = torch.autograd.grad([m, c], [x, *params_of(ns)], [go_m.to(m.dtype), go_c.to(c.dtype)], allow_unused=True)
                params_cos = [cosine(a, b) for a, b in zip(grads[1:], gr[1:]) if a is not None and b is not None]
                accuracy[name] = {
                    "mask_logits_cos": cosine(m, mr), "class_logits_cos": cosine(c, cr),
                    "dx_cos": cosine(grads[0], gr[0]), "param_grad_cos_min": min(params_cos), "mask_dtype": str(m.dtype),
                }
                print(f"  {name:24s} {accuracy[name]}", flush=True)
            report["models"][model] = {
                "batch": batch, "dim": dim,
                "fwd_bwd_ms": {k: v.get("ms") for k, v in timed.items()},
                "fwd_ms": {k: v.get("ms") for k, v in timed_fwd.items()},
                "errors": {k: v["error"] for k, v in {**timed, **timed_fwd}.items() if "error" in v},
                "accuracy_vs_fp32": accuracy,
            }
            del cands, x, base, ref_ns
            torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
