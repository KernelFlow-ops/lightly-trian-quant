"""Gate A (backward): fused attn_bwd (dqkv packed, inverse RoPE) vs fp64 autograd reference.

Cases: N=1029 plain / N=1129 joint, H=16/12, mask none/all/none-open/random/row-masked, normal and spiky scale-8 inputs.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import attn_ext, reference as ref  # noqa: E402
import test_fwd as tf  # noqa: E402


def run_case(ext, B, H, N, mask_kind, scale, spiky, rope_on=True, seed=0):
    qkv = tf.make_qkv(B, H, N, scale, spiky, seed)
    prefix = N - 1024
    rope = bl.make_rope() if rope_on else None
    bits = tf.make_bits(B, mask_kind) if mask_kind else None
    dense = ref.dense_mask_from_bits(bits, N, prefix) if bits is not None else None
    sin, cos = (rope if rope else (None, None))
    o, lse2, qk_rot = ext.attn_fwd(qkv, H, sin, cos, bits, prefix if rope_on else -1, 100 if bits is not None else 0, prefix)
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    do = torch.randn(B, N, H * 64, device="cuda", generator=g).to(torch.bfloat16)
    dqkv = ext.attn_bwd(qkv, qk_rot if rope_on else None, o, lse2, do, H, sin, cos, bits, prefix if rope_on else -1, 100 if bits is not None else 0, prefix)
    out = {"B": B, "H": H, "N": N, "mask": mask_kind, "scale": scale, "spiky": spiky, "rope": rope_on}
    for name, rb in (("fp64", False), ("fp64_roped_bf16", True)):
        x = qkv.detach().double().requires_grad_()
        o_ref, _ = ref.ref_attention(x, rope, dense, prefix, torch.float64, round_bf16=rb)
        (gx,) = torch.autograd.grad(o_ref, x, do.double())
        out[f"dqkv_vs_{name}"] = bl.err_stats(dqkv.float(), gx)
        for i, nm in enumerate(("dq", "dk", "dv")):
            out[f"{nm}_vs_{name}"] = bl.err_stats(dqkv[:, :, i].float(), gx[:, :, i])
    out["finite"] = bool(torch.isfinite(dqkv.float()).all())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    ext = attn_ext.load_ext()
    results = []
    cases = []
    for N, masks in ((1029, [None]), (1129, [None, "all", "none", "random", "rowmasked"])):
        for H in ((16,) if a.quick else (16, 12)):
            for mk in masks:
                for scale, spiky in ((1.0, False), (8.0, True)):
                    cases.append((2, H, N, mk, scale, spiky))
    cases.append((2, 16, 1029, None, 1.0, False, False))  # rope off
    for c in cases:
        r = run_case(ext, *c) if len(c) == 6 else run_case(ext, *c[:6], rope_on=c[6])
        results.append(r)
        e = r["dqkv_vs_fp64_roped_bf16"]
        f = r["dqkv_vs_fp64"]
        print(f"B{r['B']} H{r['H']} N{r['N']} mask={r['mask']} scale={r['scale']} spiky={r['spiky']} rope={r['rope']}: "
              f"dqkv(vs roped-bf16) cos {e['cos']:.7f} rel {e['rel_l2']:.2e} maxabs {e['max_abs']:.2e} | vs fp64 cos {f['cos']:.7f} | "
              f"dq {r['dq_vs_fp64_roped_bf16']['cos']:.6f} dk {r['dk_vs_fp64_roped_bf16']['cos']:.6f} dv {r['dv_vs_fp64_roped_bf16']['cos']:.6f} | finite {r['finite']}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
