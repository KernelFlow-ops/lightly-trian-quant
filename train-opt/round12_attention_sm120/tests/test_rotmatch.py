"""Is the kernel's rotated q|k (qk_rot) bit-identical to the torch/Triton RoPE on real data?"""
import sys
from pathlib import Path

import torch

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
sys.path.insert(0, str(R12.parent / "src"))
from round12_attention_sm120.impl import attn_ext  # noqa: E402
from common.ops import _apply_rope  # noqa: E402
from round4_attention.impl.rope import rope_qkv  # noqa: E402

d = torch.load(sys.argv[1])
qkv, h = d["qkv"].cuda(), d["h"]
rope = tuple(t.cuda() for t in d["rope"])
B, N, _ = qkv.shape
ext = attn_ext.load_ext()
o, lse2, qk_rot = ext.attn_fwd(qkv, h, rope[0], rope[1], None, N - 1024, 0, 0, True)
q5 = qkv.view(B, N, 3, h, 64)
q, k, v = (t.transpose(1, 2) for t in q5.unbind(2))
qr, kr = _apply_rope(q, k, rope)  # torch (bf16 in -> fp32 rope -> bf16)
tr = rope_qkv(q5.contiguous(), rope)  # triton
mine = qk_rot.view(B, N, 2, h, 64)
for nm, ref_q, ref_k in (("torch", qr.transpose(1, 2), kr.transpose(1, 2)), ("triton", tr[:, :, 0], tr[:, :, 1])):
    mq = int((mine[:, :, 0] != ref_q).sum()); mk = int((mine[:, :, 1] != ref_k).sum())
    print(f"{nm}: mismatching bf16 elements q {mq} k {mk} of {ref_q.numel()}")
print("torch vs triton rope: q", int((qr.transpose(1, 2) != tr[:, :, 0]).sum()), "k", int((kr.transpose(1, 2) != tr[:, :, 1]).sum()))
# separability of the table
sin, cos = rope
y = torch.arange(1024, device="cuda") // 32
x = torch.arange(1024, device="cuda") % 32
ok = torch.equal(cos[:, :16], cos[y * 32, :16]) and torch.equal(cos[:, 16:32], cos[x, 16:32]) and torch.equal(sin[:, :16], sin[y * 32, :16]) and torch.equal(sin[:, 16:32], sin[x, 16:32])
dup = torch.equal(cos[:, :32], cos[:, 32:]) and torch.equal(sin[:, :32], sin[:, 32:])
print("table separable:", ok, " halves duplicated:", dup)
