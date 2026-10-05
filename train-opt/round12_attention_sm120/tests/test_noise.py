"""Where does the fused backward differ from an fp64 reference compared with the TB1 path (triton rope + flash)?  Per-region relative errors."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import reference as ref  # noqa: E402
from round12_attention_sm120.impl.fused import make_attn_core as make_fused  # noqa: E402
from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core as make_tb1  # noqa: E402
from common.ops import default_ops  # noqa: E402


class C:
    pass


def rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def main():
    B, H, N = 2, 16, 1029
    ctx = C()
    fused = make_fused()
    tb1 = make_tb1(AttnCfg(rope="triton", plain="flash", joint="cudnn", qsplit=True, nq=100), fallback=default_ops().attn_core)
    torch.manual_seed(0)
    for scale, spiky in ((1.0, False), (4.0, True)):
        x = torch.randn(B, N, 3, H, 64, device="cuda") * scale ** 0.5
        if spiky:
            x[:, 1:5, 1] *= 4.0
            x[:, 1:5, 0] += 2.0
        qkv0 = x.to(torch.bfloat16)
        rope = bl.make_rope()
        dy = torch.randn(B, N, H * 64, device="cuda").to(torch.bfloat16)
        xd = qkv0.double().requires_grad_()
        o_ref, _ = ref.ref_attention(xd, rope, None, 5, torch.float64, round_bf16=True)
        (g_ref,) = torch.autograd.grad(o_ref, xd, dy.double())
        print(f"--- scale={scale} spiky={spiky}")
        for name, fn in (("fused", fused), ("TB1(flash+triton rope)", tb1), ("torch default", default_ops().attn_core)):
            q = qkv0.reshape(B, N, -1).clone().requires_grad_()
            o = fn(ctx, q, rope, None, H)
            (g,) = torch.autograd.grad(o, q, dy)
            g = g.view(B, N, 3, H, 64)
            out = {"O": rel(o, o_ref)}
            for i, nm in enumerate(("dq", "dk", "dv")):
                out[nm] = rel(g[:, :, i], g_ref[:, :, i])
            out["dq_prefix"] = rel(g[:, :5, 0], g_ref[:, :5, 0])
            out["dk_prefix"] = rel(g[:, :5, 1], g_ref[:, :5, 1])
            out["dqk_tail1024+"] = rel(g[:, 1024:, :2], g_ref[:, 1024:, :2])
            print(f"{name:24s} " + " ".join(f"{k} {v:.2e}" for k, v in out.items()))


if __name__ == "__main__":
    main()
