"""Large-batch consistency: same inputs twice (fwd bitwise, bwd noise), and accuracy per batch index vs an fp32 SDPA reference at the real B."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import reference as ref  # noqa: E402
from round12_attention_sm120.impl.fused import make_attn_core as make_fused  # noqa: E402
from round12_attention_sm120.impl.bitmask import QueryBitmask  # noqa: E402
from round12_attention_sm120.impl import attn_ext  # noqa: E402


class C:
    pass


def main():
    ctx = C()
    fused = make_fused()
    ext = attn_ext.load_ext()
    torch.manual_seed(0)
    for (B, H, N, joint) in ((7, 16, 1029, False), (7, 16, 1129, True), (14, 12, 1029, False)):
        x = (torch.randn(B, N, 3, H, 64, device="cuda") * 1.5).to(torch.bfloat16)
        rope = bl.make_rope()
        dy = torch.randn(B, N, H * 64, device="cuda").to(torch.bfloat16)
        mask = None
        dense = None
        if joint:
            bits = torch.randint(-2**31, 2**31 - 1, (B, 100, 32), device="cuda", dtype=torch.int64).to(torch.int32)
            mask = QueryBitmask(bits, N - 1024, 100)
            dense = mask.to_dense(N)
        runs = []
        for _ in range(2):
            q = x.reshape(B, N, -1).clone().requires_grad_()
            o = fused(ctx, q, rope, mask, H)
            (g,) = torch.autograd.grad(o, q, dy)
            runs.append((o.detach(), g.detach()))
        print(f"B{B} H{H} N{N} joint={joint}: fwd bitwise repeat {bool(torch.equal(runs[0][0], runs[1][0]))}, bwd repeat rel diff {float((runs[0][1].float()-runs[1][1].float()).norm()/runs[0][1].float().norm()):.2e}")
        # fp32 reference (torch math on rope-rounded inputs), per batch index
        xr = x.float().requires_grad_()
        o_ref, _ = ref.ref_attention(xr, rope, dense, N - 1024, torch.float32, round_bf16=True)
        (g_ref,) = torch.autograd.grad(o_ref, xr, dy.float())
        errs_o = [float((runs[0][0][b].float() - o_ref[b].detach()).norm() / o_ref[b].detach().norm()) for b in range(B)]
        g0 = runs[0][1].view(B, N, 3, H, 64).float()
        errs_g = [float((g0[b] - g_ref[b]).norm() / g_ref[b].norm()) for b in range(B)]
        print("   O rel err per batch:", " ".join(f"{e:.1e}" for e in errs_o))
        print("   dqkv rel err per batch:", " ".join(f"{e:.1e}" for e in errs_g))
        qkd = [float((g0[:, :, i] - g_ref[:, :, i]).norm() / g_ref[:, :, i].norm()) for i in range(3)]
        print("   dq/dk/dv rel err:", " ".join(f"{e:.1e}" for e in qkd), flush=True)
        del xr, o_ref, g_ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
