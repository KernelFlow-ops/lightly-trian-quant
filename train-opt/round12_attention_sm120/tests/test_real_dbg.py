"""Per-token error profile of the fused backward on a captured real block (see test_real.py)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
sys.path.insert(0, str(R12.parent / "src"))
from round12_attention_sm120.impl import reference as ref  # noqa: E402
from round12_attention_sm120.impl.fused import make_attn_core as make_fused  # noqa: E402
from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core as make_tb1  # noqa: E402
from common.ops import default_ops  # noqa: E402


class C:
    pass


d = torch.load(sys.argv[1])
qkv, do, h = d["qkv"].cuda(), d["do"].cuda(), d["h"]
rope = tuple(t.cuda() for t in d["rope"])
B, N, _ = qkv.shape
prefix = N - 1024
xd = qkv.reshape(B, N, 3, h, 64).double().requires_grad_()
o_ref, lse_ref = ref.ref_attention(xd, rope, None, prefix, torch.float64, round_bf16=True)
(g_ref,) = torch.autograd.grad(o_ref, xd, do.double().view(B, N, h * 64))
fused = make_fused()
tb1 = make_tb1(AttnCfg(rope="triton", plain="flash", joint="cudnn", qsplit=True, nq=100), fallback=default_ops().attn_core)
q5 = qkv.reshape(B, N, 3, h, 64)
print("qkv absmax per slot:", [float(q5[:, :, i].abs().max()) for i in range(3)], " per-token |qk| max (first 8 tokens):", [round(float(q5[:, t, :2].abs().max()), 1) for t in range(8)])
print("lse_ref range:", float(lse_ref.min()), float(lse_ref.max()))
for name, fn in (("fused", fused), ("TB1", tb1)):
    q = qkv.clone().requires_grad_()
    o = fn(C(), q, rope, None, h)
    (g,) = torch.autograd.grad(o, q, do)
    g = g.view(B, N, 3, h, 64).double()
    for i, nm in enumerate(("dq", "dk")):
        err = (g[:, :, i] - g_ref[:, :, i]).norm(dim=-1)  # [B,N,H]
        tot = g_ref[:, :, i].norm()
        per_tok = (err ** 2).sum(dim=(0, 2)).sqrt() / tot  # contribution of each token to the relative L2 error
        top = torch.topk(per_tok, 6)
        print(f"{name} {nm}: rel {float((err**2).sum().sqrt()/tot):.3e}; worst tokens {[(int(t), round(float(v),4)) for v, t in zip(top.values, top.indices)]}; "
              f"prefix share {float((per_tok[:5]**2).sum()):.3e} of total sq err {float((per_tok**2).sum()):.3e}")
