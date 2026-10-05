"""Level-A correctness: each fused kernel vs the torch op sequence (bf16) and vs an fp32 reference."""
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import json
import math

import torch
import torch.nn.functional as F
from check import tensor_diff

from impl import ops

torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
only = sys.argv[1:] or ["k1", "k4", "k3", "k2"]
results = {}


def rand(*s, scale=1.0):
    return (torch.randn(*s, device=dev) * scale).to(bf)


def rope_ref(q, sin, cos, prefix):  # q [B,H,N,64] bf16
    qf = q.float()
    x = qf[:, :, prefix:]
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], -1)
    out = qf.clone()
    out[:, :, prefix:] = x * cos + rot * sin
    return out.to(q.dtype)


for name, d, ff, H in (("vitl16", 1024, 4096, 16), ("vitb16", 768, 3072, 12)):
    for (B, N, prefix) in ((2, 1029, 5), (2, 1129, 105)):
        M = B * N
        tag = f"{name}_N{N}"
        if "k1" in only:
            x, r = rand(B, N, d), rand(B, N, d)
            w = (1 + 0.1 * torch.randn(d, device=dev)).float()
            b = (0.1 * torch.randn(d, device=dev)).float()
            s, y = ops.add_layernorm(x, r, w, b)
            s_ref = x + r
            y_ref = F.layer_norm(s_ref, (d,), w.to(bf), b.to(bf), 1e-5)
            y32 = F.layer_norm((x.float() + r.float()), (d,), w, b, 1e-5)
            _, y0 = ops.add_layernorm(x, None, w, b)
            # fp32 residual stream variants (autocast-like): x fp32, r bf16 or fp32
            xf, rf = x.float(), r.float()
            s32, y32b = ops.add_layernorm(xf, r, w, b)          # fp32 x + bf16 delta
            s32b, y32c = ops.add_layernorm(xf, rf, w, b)        # fp32 + fp32
            _, y32d = ops.add_layernorm(xf, None, w, b)
            tr = F.layer_norm(xf + rf, (d,), w, b, 1e-5)
            results[f"k1_fp32res_{tag}"] = {"s_x32_rbf16": tensor_diff(s32, xf + rf), "y_x32_rbf16": tensor_diff(y32b, tr),
                                            "s_x32_r32": tensor_diff(s32b, xf + rf), "y_x32_r32": tensor_diff(y32c, tr),
                                            "y_ln_only_x32": tensor_diff(y32d, F.layer_norm(xf, (d,), w, b, 1e-5))}
            results[f"k1_add_ln_{tag}"] = {"s": tensor_diff(s, s_ref), "y_vs_torch_bf16": tensor_diff(y, y_ref),
                                           "y_vs_fp32": tensor_diff(y, y32), "torch_vs_fp32": tensor_diff(y_ref, y32),
                                           "ln_only": tensor_diff(y0, F.layer_norm(x, (d,), w.to(bf), b.to(bf), 1e-5))}
        if "k4" in only:
            h = rand(M, d)
            W1 = rand(ff, d, scale=d ** -0.5)
            b1 = rand(ff, scale=0.1)
            out = ops.gemm_bias_gelu(h, W1, b1)
            ref = F.gelu(F.linear(h, W1, b1))
            ref32 = F.gelu(F.linear(h.float(), W1.float(), b1.float()))
            results[f"k4_gemm_gelu_{tag}"] = {"vs_torch_bf16": tensor_diff(out, ref), "vs_fp32": tensor_diff(out, ref32),
                                              "torch_vs_fp32": tensor_diff(ref, ref32)}
        if "k3" in only:
            for kname, K_, Nn in (("fc2", ff, d), ("proj", d, d)):
                a = rand(M, K_)
                W = rand(Nn, K_, scale=K_ ** -0.5)
                bb = rand(Nn, scale=0.1)
                res = rand(M, Nn)
                gam = (torch.rand(Nn, device=dev) * 2).float()
                out = ops.gemm_bias_residual(a, W, bb, res, gam)
                out32 = ops.gemm_bias_residual(a, W, bb, res.float(), gam)
                ref = res + gam.to(bf) * F.linear(a, W, bb)   # eager LayerScale form
                ref32 = res.float() + gam * F.linear(a.float(), W.float(), bb.float())
                results[f"k3_{kname}_{tag}"] = {"vs_torch_bf16": tensor_diff(out, ref), "vs_fp32": tensor_diff(out, ref32),
                                                "torch_vs_fp32": tensor_diff(ref, ref32),
                                                "fp32res_vs_fp32": tensor_diff(out32, ref32)}
        if "k2" in only:
            h = rand(B, N, d)
            Wq = rand(3 * d, d, scale=d ** -0.5)
            bq = rand(3 * d, scale=0.1)
            mask = torch.ones(3 * d, device=dev, dtype=bf)
            mask[d:2 * d] = 0  # K-bias masked
            bm_ = bq * mask
            ang = torch.randn(1024, 32, device=dev) * 2
            sin32 = torch.sin(ang).repeat(1, 2).float().contiguous()
            cos32 = torch.cos(ang).repeat(1, 2).float().contiguous()
            out_f = ops.qkv_rope(h, Wq, bm_, sin32, cos32, prefix, H)                   # rope on fp32 accumulator
            out_r = ops.qkv_rope(h, Wq, bm_, sin32, cos32, prefix, H, round_bf16=True)  # torch-like rounding first
            qkv = F.linear(h, Wq, bm_).reshape(B, N, 3, H, 64)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
            qr, kr = rope_ref(q, sin32, cos32, prefix), rope_ref(k, sin32, cos32, prefix)
            qkv32 = F.linear(h.float(), Wq.float(), bm_.float()).reshape(B, N, 3, H, 64)
            q3, k3, v3 = [t.transpose(1, 2) for t in qkv32.unbind(2)]
            qt, kt = rope_ref(q3, sin32, cos32, prefix), rope_ref(k3, sin32, cos32, prefix)
            results[f"k2_qkv_rope_{tag}"] = {
                "roundbf16_q_vs_torch": tensor_diff(out_r[0], qr), "roundbf16_k_vs_torch": tensor_diff(out_r[1], kr),
                "v_vs_torch": tensor_diff(out_r[2], v),
                "fp32rope_q_vs_fp32": tensor_diff(out_f[0], qt), "fp32rope_k_vs_fp32": tensor_diff(out_f[1], kt),
                "fp32rope_v_vs_fp32": tensor_diff(out_f[2], v3),
                "torch_q_vs_fp32": tensor_diff(qr, qt), "torch_k_vs_fp32": tensor_diff(kr, kt),
            }


def fmt(dct):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dct.items() if k in ("max_abs", "cos", "nan")}


for k, v in results.items():
    print(k)
    for kk, vv in v.items():
        print("   ", kk, fmt(vv))
json.dump(results, open("/root/autodl-tmp/infer-opt-tmp/blockfuse/test_kernels_last.json", "w"), indent=1)
