"""attn_core with torch RoPE vs round4 fused Triton RoPE (bf16 qkv), fwd+bwd at the real shapes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare, errors, fmt  # noqa: E402

MODELS = {"vitl16": (7, 16), "vitb16": (14, 12)}


def rope_tables(scale: float = 1.0):
    pos = torch.arange(1024, device="cuda", dtype=torch.float32)
    freq = torch.exp(torch.linspace(0, -3, 32, device="cuda")) * scale
    angles = (pos[:, None] * freq[None, :]) * 0.05
    angles = torch.cat((angles, angles), dim=-1)
    return torch.sin(angles), torch.cos(angles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    ctx = SimpleNamespace()
    report: dict = {"cases": []}
    with gpu_lock():
        for model, (batch, heads) in MODELS.items():
            for tokens in (1029, 1129):
                rope = rope_tables(8.0)
                gen = torch.Generator(device="cuda").manual_seed(0)
                qkv = torch.randn(batch, tokens, 3 * heads * 64, device="cuda", generator=gen).to(torch.bfloat16).requires_grad_(True)
                go = torch.randn(batch, tokens, heads * 64, device="cuda", generator=gen).to(torch.bfloat16)
                cores = {
                    "rope=torch": make_attn_core(AttnCfg(rope="torch", plain="flash"), None),
                    "rope=triton": make_attn_core(AttnCfg(rope="triton", plain="flash"), None),
                }

                def fwdbwd(core):
                    def run():
                        o = core(ctx, qkv, rope, None, heads)
                        torch.autograd.grad(o, qkv, go)
                    return run

                def fwd(core):
                    def run():
                        with torch.no_grad():
                            core(ctx, qkv, rope, None, heads)
                    return run

                both = compare({k: fwdbwd(v) for k, v in cores.items()})
                fwd_only = compare({k: fwd(v) for k, v in cores.items()})
                print(f"--- {model} N={tokens} attn_core(flash, rope) fwd+bwd", flush=True)
                print(fmt(both), flush=True)
                print(f"    fwd only", flush=True)
                print(fmt(fwd_only), flush=True)
                o_t = cores["rope=torch"](ctx, qkv, rope, None, heads)
                o_x = cores["rope=triton"](ctx, qkv, rope, None, heads)
                (g_t,) = torch.autograd.grad(o_t, qkv, go)
                (g_x,) = torch.autograd.grad(o_x, qkv, go)
                acc = {"out": errors(o_x, o_t), "dqkv": errors(g_x, g_t), "out_bitwise_equal": bool(torch.equal(o_x, o_t)), "dqkv_bitwise_equal": bool(torch.equal(g_x, g_t))}
                print(f"    triton vs torch rope: {acc}", flush=True)
                report["cases"].append({"model": model, "tokens": tokens, "fwd_bwd_ms": {k: v.get("ms") for k, v in both.items()}, "fwd_ms": {k: v.get("ms") for k, v in fwd_only.items()}, "triton_vs_torch": acc})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
