"""Gate A for the round14 head: ops.predict (fused) vs the fp32 source implementation, outputs + all gradients.

Cases: plain gaussian tokens, spiky scale-8 tokens (massive-activation registers/patch outliers), both models' shapes.
Also reports the stock torch bf16 head (head_ln_bf16, the TB1 reference head) against the same fp32 reference for context.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.impl import head as head_impl  # noqa: E402
from round11_bf16_mixed.tests.bench_head import GRID, MODELS, NQ, TOKENS, build_head, params_of, source_predict  # noqa: E402
from round14_head_sm120.impl import head as r14head  # noqa: E402


def stats(a, b):
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    d = (a - b)
    return {
        "cos": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)),
        "rel_l2": float(d.norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float(d.abs().max()),
    }


def randomize(ns, gen):
    """non-trivial affine params / depthwise weights so that every gradient path is exercised."""
    with torch.no_grad():
        for st in ns.upscale:
            st.norm.weight.copy_(1 + 0.3 * torch.randn_like(st.norm.weight))
            st.norm.bias.copy_(0.2 * torch.randn_like(st.norm.bias))
            st.conv2.weight.mul_(3.0)
            st.conv1.bias.copy_(0.1 * torch.randn_like(st.conv1.bias))
        ns.backbone.norm.weight.copy_(1 + 0.3 * torch.randn_like(ns.backbone.norm.weight))
        ns.backbone.norm.bias.copy_(0.1 * torch.randn_like(ns.backbone.norm.bias))


def make_x(B, C, spiky, gen):
    x = torch.randn(B, TOKENS, C, device="cuda", generator=gen) * 2.0
    if spiky:
        x = x * 8.0
        # a few massive-activation channels / tokens as in DINOv3 registers
        idx = torch.randint(0, C, (4,), device="cuda", generator=gen)
        x[:, :, idx] += 300.0
        x[:, 105:110, :] *= 6.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = {}
    ok_all = True
    with gpu_lock():
        for model in args.models:
            B, C = MODELS[model]
            for spiky in (False, True):
                gen = torch.Generator(device="cuda").manual_seed(3)
                torch.manual_seed(0)
                base = build_head(C)
                randomize(base, gen)
                x0 = make_x(B, C, spiky, gen)
                go_m = torch.randn(B, NQ, 128, 128, device="cuda", generator=gen)
                go_c = torch.randn(B, NQ, 3, device="cuda", generator=gen)
                # fp32 reference (no autocast, highest matmul precision)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                ref_ns = copy.deepcopy(base)
                xr = x0.clone().requires_grad_(True)
                mr, cr = source_predict(ref_ns, xr)
                gr = torch.autograd.grad([mr, cr], [xr, *params_of(ref_ns)], [go_m, go_c], allow_unused=True)
                names = ["x"] + [n for n, _ in [(f"p{i}", None) for i in range(len(gr) - 1)]]
                pnames = ["x"] + [n for n, p in [("bbnorm.w", 0), ("bbnorm.b", 0)]]
                named = [("x", None)]
                for modname, mod in (("bb_norm", ref_ns.backbone.norm), ("class_head", ref_ns.class_head), ("mask_head", ref_ns.mask_head), ("upscale", ref_ns.upscale)):
                    named += [(f"{modname}.{n}", p) for n, p in mod.named_parameters()]
                results = {}
                for cand in ("r14", "r14_mx8", "r14_mx8bwd", "r14_mx8wgrad", "torch_bf16"):
                    ns = copy.deepcopy(base)
                    ctx = SimpleNamespace(train_model=SimpleNamespace(model=ns), model=ns)
                    r14head.POLICY["mx8"] = {"r14_mx8": True, "r14_mx8bwd": "bwd", "r14_mx8wgrad": "wgrad"}.get(cand, False)
                    if cand == "torch_bf16":
                        head_impl.install_ln2d_bf16(ctx)
                        fn = lambda x, ns=ns: source_predict(ns, x)
                    else:
                        fn = lambda x, ctx=ctx: r14head.predict(ctx, x, GRID)
                    xc = x0.clone().requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        mc, cc = fn(xc)
                    gc = torch.autograd.grad([mc, cc], [xc, *params_of(ns)], [go_m.to(mc.dtype), go_c.to(cc.dtype)], allow_unused=True)
                    res = {"mask_logits": stats(mc, mr), "class_logits": stats(cc, cr), "mask_dtype": str(mc.dtype), "class_dtype": str(cc.dtype)}
                    per = {}
                    for (n, _), a, b in zip(named, gc, gr):
                        per[n] = stats(a, b)
                    res["grad_cos_min"] = min(v["cos"] for v in per.values())
                    res["grad_rel_max"] = max(v["rel_l2"] for v in per.values())
                    res["worst"] = sorted(((v["cos"], n) for n, v in per.items()))[:3]
                    res["per_param"] = per
                    results[cand] = res
                key = f"{model}/{'spiky8' if spiky else 'gauss'}"
                report[key] = results
                print(f"== {key} B={B} C={C}")
                for cand, res in results.items():
                    print(f"  {cand:11s} mask cos={res['mask_logits']['cos']:.7f} rel={res['mask_logits']['rel_l2']:.2e} | class cos={res['class_logits']['cos']:.7f} | grad cos_min={res['grad_cos_min']:.6f} rel_max={res['grad_rel_max']:.2e} worst={res['worst']}")
                r = results["r14"]
                passed = r["mask_logits"]["cos"] > 0.9999 and r["grad_cos_min"] > 0.999 and r["class_logits"]["cos"] > 0.9999
                # must be no worse than torch bf16 head by a factor 2 on rel error
                passed = passed and r["grad_rel_max"] <= 2.0 * max(results["torch_bf16"]["grad_rel_max"], 1e-3)
                print("  GATE A:", "PASS" if passed else "FAIL")
                ok_all &= passed
                del base, ref_ns
                torch.cuda.empty_cache()
    print("GATE A overall:", "PASS" if ok_all else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
