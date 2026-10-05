"""Round 13 step-1 op-level sanity (NOT a gate, no timing, no lock needed):
  quantisers vs round10 quant_ref.mx8_quantize (bit-exact), cuBLASLt MXFP8 GEMM vs fp32 product of the dequantised operands, epilogue passes vs torch.
  python tests/test13a.py [--d 1024 --H 16]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
for p in (HERE / "impl", ROOT / "round10_lowp_probe" / "impl"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import mx13  # noqa: E402
import quant_ref as qr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=1024)
ap.add_argument("--H", type=int, default=16)
a = ap.parse_args()
d, H, ff = a.d, a.H, 4 * a.d
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
E = mx13.ext()
res = {}


def deq(buf_q, buf_sf, R, K):
    """dequantised fp32 [R, K] of a (q [R,K] u8, blocked sf) pair"""
    rm, rk = mx13.pad128(R) // 128, K // 128
    byte = qr.from_blocked(buf_sf.view(rm, rk, 32, 4, 4), R, K // 32)
    return qr.mx8_dequant(buf_q.view(torch.float8_e4m3fn), byte)


def exact(name, x_q, x_sf, ref_in, R, K):
    q, byte = qr.mx8_quantize(ref_in)
    ok_q = (q.view(torch.uint8) == x_q[:R]).float().mean().item()
    ok_s = (qr.from_blocked(x_sf.view(mx13.pad128(R) // 128, K // 128, 32, 4, 4), R, K // 32) == byte).float().mean().item()
    res[name] = dict(q_match=ok_q, sf_match=ok_s)
    print(f"{name:34s} q bytes equal {ok_q:.7f}  scale bytes equal {ok_s:.7f}")


def mk(shape, scale=1.0, outl=True):
    x = torch.randn(*shape, device=dev) * scale
    if outl:
        x[:, :: 97] *= 8.0
    return x


for M in (2058, 2258):
    print(f"--- M={M}  d={d}")
    for K, tag in ((d, "d"), (ff, "ff")):
        x = mk((M, K))
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            xd = x.to(dt)
            b = mx13.ActBuf(M, K)
            E.quant_rows(xd, b.q, b.sf)
            exact(f"quant_rows {dt}".replace("torch.", "") + f" K={tag} M={M}", b.q, b.sf, xd.float(), M, K)
    # gelu_quant
    y = mk((M, ff), 2.0).to(torch.bfloat16)
    bias = torch.randn(ff, device=dev) * 0.1
    b = mx13.ActBuf(M, ff)
    E.gelu_quant(y, bias, b.q, b.sf)
    exact(f"gelu_quant M={M}", b.q, b.sf, F.gelu(y.float() + bias), M, ff)
    # ln_mx8
    xr = mk((M, d), 3.0)
    w, bb = torch.rand(d, device=dev) + 0.5, torch.randn(d, device=dev) * 0.1
    b = mx13.ActBuf(M, d)
    E.ln_mx8(xr, w, bb, b.q, b.sf, 1e-6)
    exact(f"ln_mx8 M={M}", b.q, b.sf, F.layer_norm(xr, (d,), w, bb, 1e-6), M, d)
    # GEMMs: fp32 product of the dequantised operands
    for kind, N, K in (("qkv", 3 * d, d), ("proj", d, d), ("fc1", ff, d), ("fc2", d, ff)):
        X = mk((M, K), 0.5)
        Wf = torch.randn(N, K, device=dev) * 0.03
        W = mx13.MXW(Wf)
        xb = mx13.ActBuf(M, K)
        E.quant_rows(X, xb.q, xb.sf)
        ref = deq(xb.q, xb.sf, M, K) @ deq(W.q, W.sf, N, K).t()
        out = mx13.mm(xb, W).float()
        rel = ((out - ref).norm() / ref.norm()).item()
        cos = F.cosine_similarity(out.flatten().double(), ref.flatten().double(), dim=0).item()
        mx = (out - ref).abs().max().item()
        # fp32 operands for context
        rel32 = ((out - X @ Wf.t()).norm() / (X @ Wf.t()).norm()).item()
        res[f"gemm {kind} M={M}"] = dict(rel_l2=rel, cos=cos, max_abs=mx, finite=bool(torch.isfinite(out).all()), rel_l2_vs_fp32_operands=rel32)
        print(f"gemm {kind:4s} M={M} N={N:5d} K={K:5d}: rel L2 vs dequant-fp32 product {rel:.2e}  cos {cos:.8f}  max abs {mx:.3e}  | vs raw fp32 operands {rel32:.2e}")

# epi_rope / epi_res
M, Ntok = 2058, 1029
B = M // Ntok
prefix = Ntok - 1024
y = (torch.randn(M, 3 * d, device=dev) * 2).to(torch.bfloat16)
bias = torch.randn(3 * d, device=dev) * 0.1
sin, cos = torch.randn(1024, 64, device=dev), torch.randn(1024, 64, device=dev)
o = E.epi_rope(y, bias, sin, cos, Ntok, prefix, d, H)
v = (y.float() + bias).view(B, Ntok, 3, H, 64)
ref = torch.empty(3, B, Ntok, H, 64, device=dev)
for wh in range(3):
    t = v[:, :, wh].clone()
    if wh < 2:
        lo, hi = t[..., :32], t[..., 32:]
        c_, s_ = cos[None, :, None, :], sin[None, :, None, :]
        tt = t[:, prefix:]
        lo, hi = tt[..., :32], tt[..., 32:]
        cc, ss = cos[None, :, None, :], sin[None, :, None, :]
        nl = lo * cc[..., :32] - hi * ss[..., :32]
        nh = hi * cc[..., 32:] + lo * ss[..., 32:]
        t[:, prefix:] = torch.cat([nl, nh], -1)
    ref[wh] = t
err = (o.float() - ref).abs().max().item()
rel = ((o.float() - ref).norm() / ref.norm()).item()
res["epi_rope"] = dict(max_abs=err, rel_l2=rel)
print(f"epi_rope: max abs {err:.3e}  rel L2 {rel:.2e} (fp16 output rounding ~5e-4)")
for N in (d, ff // 4):
    yy = torch.randn(M, N, device=dev).to(torch.bfloat16)
    r = torch.randn(M, N, device=dev)
    g, bs = torch.rand(N, device=dev), torch.randn(N, device=dev) * 0.1
    out = E.epi_res(yy, r, g, bs)
    ref = r + g * (yy.float() + bs)
    err = (out - ref).abs().max().item()
    res[f"epi_res N={N}"] = dict(max_abs=err)
    print(f"epi_res N={N}: max abs {err:.3e}")
Path(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"test13a_d{d}.json").write_text(json.dumps(res, indent=1))
