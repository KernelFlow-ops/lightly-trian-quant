"""Correctness of the TileLang attention kernels vs the PyTorch reference (SDPA / RoPE in fp32)."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, str(HERE.parent / "common"))
import torch, torch.nn.functional as F
from check import tensor_diff
import ops

def rope_ref(x, sin, cos, prefix):
    # x [B,H,N,D] bf16; identical to SelfAttention.apply_rope
    xf = x.float()
    p, r = xf[:, :, :prefix], xf[:, :, prefix:]
    x1, x2 = r.chunk(2, -1)
    r = r * cos + torch.cat([-x2, x1], -1) * sin
    return torch.cat([p, r], 2).to(x.dtype)

torch.manual_seed(0)
def rand_qkv(B, H, N, scale=2.0):
    return (torch.randn(B, N, 3, H, 64, device="cuda", dtype=torch.bfloat16) * scale)

sys.path.insert(0, "/root/lightly-train/infer-opt/round1_torch_cleanup")
import model_factory as mf
m = mf.load_model("vitl16")
sin, cos = m.backbone.rope_embed(H=32, W=32)
assert torch.equal(sin[:, :32], sin[:, 32:]) and torch.equal(cos[:, :32], cos[:, 32:])

cfgs = [None] if len(sys.argv) < 2 else [eval(sys.argv[1])]
ok = True
cases = [(2, 16, 1029, 2.0), (2, 16, 1129, 2.0), (2, 12, 1029, 2.0), (2, 12, 1129, 2.0),
         (2, 16, 1029, 8.0), (2, 16, 1129, 8.0), (2, 12, 1029, 8.0)]  # 8.0: very peaked softmax (real DINOv3 registers)
for (B, H, N, sc) in cases:
    qkv = rand_qkv(B, H, N, sc)
    q, k, v = [qkv[:, :, i].transpose(1, 2).contiguous() for i in range(3)]
    ref = F.scaled_dot_product_attention(q, k, v)
    ref32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    prefix = N - 1024
    qr, kr = rope_ref(q, sin, cos, prefix), rope_ref(k, sin, cos, prefix)
    ref_rope = F.scaled_dot_product_attention(qr, kr, v)
    ref_rope32 = F.scaled_dot_product_attention(qr.float(), kr.float(), v.float())
    for cfg in cfgs:
        outs = {
            "attention": ops.attention(q, k, v, cfg).transpose(1, 2).reshape(B, N, H * 64),
            "attention_qkv": ops.attention_qkv(qkv, cfg),
            "attention_rope": ops.attention_rope(q, k, v, sin, cos, prefix, cfg).transpose(1, 2).reshape(B, N, H * 64),
            "attention_rope_qkv": ops.attention_rope_qkv(qkv, sin, cos, prefix, cfg),
        }
        refs = {"attention": (ref, ref32), "attention_qkv": (ref, ref32),
                "attention_rope": (ref_rope, ref_rope32), "attention_rope_qkv": (ref_rope, ref_rope32)}
        for name, o in outs.items():
            r, r32 = refs[name]
            r = r.transpose(1, 2).reshape(B, N, H * 64); r32 = r32.transpose(1, 2).reshape(B, N, H * 64)
            d_sdpa = tensor_diff(o, r); d32 = tensor_diff(o, r32)
            # yardstick: SDPA-bf16 itself vs fp32
            d_ref32 = tensor_diff(r, r32)
            good = d32["max_abs"] <= 3 * d_ref32["max_abs"] + 1e-3 and not d32["nan"]
            ok &= good
            print(f"B{B} H{H} N{N} sc{sc} {name:19s} vs sdpa-bf16: max_abs {d_sdpa['max_abs']:.4f} mean_abs {d_sdpa['mean_abs']:.5f} cos {d_sdpa['cos']:.7f} | "
                  f"vs fp32 ref: max_abs {d32['max_abs']:.4f} (sdpa-bf16 vs fp32: {d_ref32['max_abs']:.4f}) {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAIL")
