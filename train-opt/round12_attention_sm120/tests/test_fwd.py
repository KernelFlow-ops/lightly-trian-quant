"""Gate A (forward): fused attn_fwd vs fp64 math reference; N=1029/1129, rope on/off, mask all/none/random/row-masked, spiky inputs."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402  (also puts train-opt on sys.path)

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import attn_ext, reference as ref  # noqa: E402


def make_qkv(B, H, N, scale=1.0, spiky=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, N, 3, H, 64, device="cuda", generator=g) * scale ** 0.5
    if spiky:  # DINOv3-like register-token spikes: a few keys with a large norm aligned with many queries
        x[:, 1:5, 1] *= 4.0
        x[:, 1:5, 0] += 2.0
    return x.to(torch.bfloat16)


def make_bits(B, kind, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "all":
        return torch.full((B, 100, 32), -1, dtype=torch.int32, device="cuda")
    if kind == "none":
        return torch.zeros(B, 100, 32, dtype=torch.int32, device="cuda")
    bits = torch.randint(-2**31, 2**31 - 1, (B, 100, 32), generator=g, device="cuda", dtype=torch.int64).to(torch.int32)
    if kind == "rowmasked":  # a few rows fully masked over patches (they still see the prefix), a few fully open
        bits[:, ::7] = 0
        bits[:, 3::11] = -1
    return bits


def run_case(ext, B, H, N, mask_kind, scale, spiky, rope_on=True):
    qkv = make_qkv(B, H, N, scale, spiky)
    prefix = N - 1024
    rope = bl.make_rope() if rope_on else None
    bits = make_bits(B, mask_kind) if mask_kind else None
    dense = ref.dense_mask_from_bits(bits, N, prefix) if bits is not None else None
    o, lse2, qk_rot = ext.attn_fwd(qkv, H, rope[0] if rope else None, rope[1] if rope else None, bits, prefix if rope_on else -1, 100 if bits is not None else 0, prefix)
    o_ref, lse_ref = ref.ref_attention(qkv, rope, dense, prefix, torch.float64, round_bf16=False)
    o_bf, _ = ref.ref_attention(qkv, rope, dense, prefix, torch.float64, round_bf16=True)  # same rope rounding as the kernel
    lse_k = lse2 * 0.6931471805599453  # log2 -> natural
    res = {
        "B": B, "H": H, "N": N, "mask": mask_kind, "scale": scale, "spiky": spiky, "rope": rope_on,
        "O_vs_fp64": bl.err_stats(o.float(), o_ref), "O_vs_fp64_roped_bf16": bl.err_stats(o.float(), o_bf),
        "lse": bl.err_stats(lse_k, lse_ref),
        "finite": bool(torch.isfinite(o.float()).all()),
    }
    # bf16 output rounding bound: |err| <= ~2^-8 * max|o| for a bf16 result; report ratio
    res["O_maxabs_over_bf16_ulp"] = res["O_vs_fp64_roped_bf16"]["max_abs"] / max(float(o_ref.abs().max()) * 2 ** -8, 1e-30)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    ext = attn_ext.load_ext(verbose=False)
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
        e = r["O_vs_fp64_roped_bf16"]
        print(f"B{r['B']} H{r['H']} N{r['N']} mask={r['mask']} scale={r['scale']} spiky={r['spiky']} rope={r['rope']}: "
              f"O(vs roped-bf16 ref) maxabs {e['max_abs']:.3e} rel {e['rel_l2']:.3e} cos {e['cos']:.8f} | vs fp64 maxabs {r['O_vs_fp64']['max_abs']:.3e} cos {r['O_vs_fp64']['cos']:.8f} "
              f"| lse maxabs {r['lse']['max_abs']:.3e} | finite {r['finite']}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
