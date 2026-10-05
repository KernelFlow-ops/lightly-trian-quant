"""Backend selection microbench for ``ops.attn_core`` at the real training shapes (bf16).

Run under the dual GPU lock (this script takes it itself):
    python round11_bf16_mixed/tests/bench_attn.py --out round11_bf16_mixed/profile/attn_backends.json
Measures fwd and fwd+bwd (grad wrt the packed qkv tensor, so the unbind/stack layout cost is
included) and checks every candidate against an fp32 math-backend reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.impl.attn import AttnCfg, attention, to_bias  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare, errors, fmt  # noqa: E402

MODELS = {"vitl16": (7, 16), "vitb16": (14, 12)}
D = 64
NQ = 100
PREFIX = NQ + 1 + 4


def make_inputs(batch: int, heads: int, tokens: int, *, spiky: bool, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(batch, tokens, 3 * heads * D, device="cuda", dtype=torch.float32, generator=gen)
    if spiky:
        # DINOv3 register tokens produce very peaked attention: a few huge-norm keys, large q.
        view = qkv.view(batch, tokens, 3, heads, D)
        view[:, :, 0] *= 3.0
        view[:, :, 1] *= 3.0
        view[:, NQ + 1 : NQ + 5, 1] *= 8.0 if tokens > NQ + 5 else 1.0
    qkv = qkv.to(torch.bfloat16).requires_grad_(True)
    go = torch.randn(batch, tokens, heads * D, device="cuda", dtype=torch.float32, generator=gen).to(torch.bfloat16)
    return qkv, go


def make_mask(batch: int, tokens: int, density: float, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mask = torch.ones(batch, 1, tokens, tokens, dtype=torch.bool, device="cuda")
    if tokens > PREFIX:
        window = torch.rand(batch, NQ, tokens - PREFIX, device="cuda", generator=gen) < density
        mask[:, 0, :NQ, PREFIX:] = window
    return mask


def split_qkv(qkv: torch.Tensor, heads: int):
    batch, tokens, _ = qkv.shape
    q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv.reshape(batch, tokens, 3, heads, D), 2))
    return q, k, v


def run_attention(qkv: torch.Tensor, heads: int, mask, cfg: AttnCfg) -> torch.Tensor:
    q, k, v = split_qkv(qkv, heads)
    out = attention(q, k, v, mask, cfg)
    return out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], heads * D)


def reference(qkv: torch.Tensor, go: torch.Tensor, heads: int, mask):
    qkv32 = qkv.detach().float().requires_grad_(True)
    q, k, v = split_qkv(qkv32, heads)
    expanded = None if mask is None else mask.expand(-1, heads, -1, -1)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=expanded)
    out = out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], heads * D)
    (grad,) = torch.autograd.grad(out, qkv32, go.float())
    return out.detach(), grad.detach()


def build_candidates(heads: int, mask, mask3, name_prefix: str) -> dict:
    """name -> callable(qkv) -> out"""
    cands: dict = {}
    if mask is None:
        for backend in ("auto", "flash", "cudnn", "eff"):
            cfg = AttnCfg(plain=backend)
            cands[f"{name_prefix}/{backend}"] = (lambda qkv, cfg=cfg: run_attention(qkv, heads, None, cfg))
        return cands

    def default_expand(qkv):
        q, k, v = split_qkv(qkv, heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask3[:, None, ...].expand(-1, heads, -1, -1))
        return out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], heads * D)

    cands[f"{name_prefix}/coord_default(bool3 expand,auto)"] = default_expand
    for backend in ("auto", "eff", "cudnn"):
        cands[f"{name_prefix}/bool4/{backend}"] = (lambda qkv, b=backend: run_attention(qkv, heads, mask, AttnCfg(joint=b)))
    for backend in ("auto", "eff", "cudnn"):
        cands[f"{name_prefix}/bias/{backend}"] = (lambda qkv, b=backend: run_attention(qkv, heads, mask, AttnCfg(joint=b, bias=True)))
    for plain in ("flash", "cudnn", "eff"):
        for joint in ("eff", "cudnn"):
            for bias in (False, True):
                label = f"{name_prefix}/qsplit(free={plain},masked={joint},{'bias' if bias else 'bool'})"
                cands[label] = (lambda qkv, p=plain, j=joint, b=bias: run_attention(qkv, heads, mask, AttnCfg(plain=p, joint=j, bias=b, qsplit=True, nq=NQ)))
    return cands


def measure_case(label: str, batch: int, heads: int, tokens: int, mask, mask3, *, spiky: bool, iters: int, check: bool) -> dict:
    qkv, go = make_inputs(batch, heads, tokens, spiky=spiky)
    cands = build_candidates(heads, mask, mask3, label)
    result: dict = {"label": label, "batch": batch, "heads": heads, "tokens": tokens, "masked": mask is not None, "spiky": spiky}

    def fwd_closure(fn):
        def run():
            with torch.no_grad():
                fn(qkv)
        return run

    def fwdbwd_closure(fn):
        def run():
            out = fn(qkv)
            torch.autograd.grad(out, qkv, go)
        return run

    fwd = compare({k: fwd_closure(v) for k, v in cands.items()}, iters=iters)
    both = compare({k: fwdbwd_closure(v) for k, v in cands.items()}, iters=iters)
    print(f"--- {label} (B={batch} H={heads} N={tokens} masked={mask is not None}) fwd+bwd", flush=True)
    print(fmt(both), flush=True)
    print(f"--- {label} fwd only", flush=True)
    print(fmt(fwd), flush=True)
    result["fwd_ms"] = {k: v.get("ms", None) for k, v in fwd.items()}
    result["fwd_bwd_ms"] = {k: v.get("ms", None) for k, v in both.items()}
    result["errors"] = {k: v["error"] for k, v in {**fwd, **both}.items() if "error" in v}
    if check:
        out_ref, grad_ref = reference(qkv, go, heads, mask)
        acc = {}
        for name, fn in cands.items():
            if name in result["errors"]:
                continue
            out = fn(qkv)
            (grad,) = torch.autograd.grad(out, qkv, go)
            acc[name] = {"out": errors(out, out_ref), "dqkv": errors(grad, grad_ref)}
        result["accuracy_vs_fp32_math"] = acc
        worst = min((v["dqkv"]["cos"] for v in acc.values()), default=None)
        print(f"    min dqkv cosine vs fp32 reference over candidates: {worst}", flush=True)
    del qkv, go
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.5)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    report: dict = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "density": args.density, "cases": []}
    with gpu_lock():
        for model in args.models:
            batch, heads = MODELS[model]
            for tokens, masked in ((1029, False), (1129, True), (1129, False)):
                mask = make_mask(batch, tokens, args.density) if masked else None
                mask3 = mask[:, 0] if masked else None
                label = f"{model}/N{tokens}/{'masked' if masked else 'nomask'}"
                case = measure_case(label, batch, heads, tokens, mask, mask3, spiky=True, iters=args.iters, check=True)
                case["model"] = model
                report["cases"].append(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
