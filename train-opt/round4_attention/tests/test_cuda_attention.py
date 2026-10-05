"""A-level checks for the real CUDA C++ attention extension."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from round4_attention.impl.cuda_attention import attention
from round4_attention.impl.attention import reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_cpp_attention_small() -> None:
    result = one(1, 9, 1, 3, True, True, torch.float32)
    assert result["out_cos"] >= 0.999
    assert result["grad_cos"] >= 0.999
    assert result["lse_finite"]


def one(B, N, H, prefix, rope_on, masked, dtype):
    torch.manual_seed(20261002 + B + N + H + int(rope_on) + int(masked))
    qkv = torch.randn(B, N, 3, H, 64, device="cuda", dtype=dtype, requires_grad=True)
    rope = None
    if rope_on:
        z = torch.randn(N - prefix, 32, device="cuda", dtype=torch.float32)
        z = torch.cat((z, z), -1)
        rope = (z.sin().contiguous(), z.cos().contiguous())
    mask = torch.rand(B, 2, N - prefix, device="cuda") > 0.43 if masked else None
    out, lse = attention(qkv, rope, mask, prefix, return_lse=True)
    dy = torch.randn_like(out)
    dq = torch.autograd.grad(out, qkv, dy)[0]
    ref_qkv = qkv.detach().clone().requires_grad_()
    ref = reference(ref_qkv, rope, mask, prefix)
    ref_dq = torch.autograd.grad(ref, ref_qkv, dy)[0]
    return {"shape": [B, N, H], "dtype": str(dtype), "rope": rope_on, "masked": masked,
            "out_max_abs": float((out - ref).abs().max()),
            "out_cos": float(F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)),
            "grad_max_abs": float((dq - ref_dq).abs().max()),
            "grad_cos": float(F.cosine_similarity(dq.float().flatten(), ref_dq.float().flatten(), dim=0)),
            "lse_finite": bool(torch.isfinite(lse).all())}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--full", action="store_true"); p.add_argument("--output", type=Path)
    a = p.parse_args()
    cases = ([one(14, 1029, 12, 5, True, False, torch.bfloat16), one(7, 1129, 16, 105, True, True, torch.bfloat16)]
             if a.full else [one(1, 9, 1, 3, r, m, torch.float32) for r, m in ((False, False), (True, False), (True, True))])
    for c in cases:
        c["pass"] = c["out_cos"] >= .999 and c["grad_cos"] >= .999 and c["lse_finite"]
        print(json.dumps(c), flush=True)
    result = {"cases": cases, "pass": all(c["pass"] for c in cases)}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
