"""Round 14 step 2: fused epilogues of the own FP4 GEMM vs the fp64 product of the dequantised operands (sanity, not a gate).
  EPI 1 residual (fp32), EPI 2 bias+RoPE+[3,B,N,H,64] fp16 layout, EPI 3 bias+GELU+FP4 quantize-out (codes vs ref quantisation of the exact GELU), EPI 4 GELU bf16.
  python t_ep14.py [model]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl"))
import torch
import fp4gb
import fp4util as U
from fp4util import R

dev = "cuda"
torch.manual_seed(2)
model = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
D = dict(vitl16=dict(d=1024, ff=4096, H=16), vitb16=dict(d=768, ff=3072, H=12))[model]
d, ff, H = D["d"], D["ff"], D["H"]


def heavy(M, K):
    x = torch.randn(M, K, device=dev)
    x[:, 5] *= 30
    x[:, K // 3] *= 12
    return x


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def ref_quant(x, fmt, ts):
    xf = x.float()
    r, c = xf.shape
    if fmt == "nv4":
        sf8 = R._nv4_sf(xf.reshape(r, c // 16, 16).abs().amax(-1), ts)
        return R.pack_nibbles(R._codes_nv4(xf, sf8, ts, False, None)), R.to_blocked(sf8.view(torch.uint8))
    byte = R._mx4_byte(xf.reshape(r, c // 32, 32).abs().amax(-1))
    return R.pack_nibbles(R._codes_mx4(xf, byte, False, None)), R.to_blocked(byte.to(torch.uint8))


CFG = dict(bm=128, bn=128, bkb=64, st=4, wm=2, wn=4, minb=1)
for fmt, fi in (("nv4", 0), ("mx4", 1)):
    for M, Ntok in ((2058, 1029), (2258, 1129)):
        prefix = Ntok - 1024
        B = M // Ntok
        sin = torch.randn(1024, 64, device=dev)
        cos = torch.randn(1024, 64, device=dev)
        # ---- EPI 2 qkv
        N, K = 3 * d, d
        xq = R.quantize_matrix(heavy(M, K), fmt)
        wq = R.quantize_matrix(torch.randn(N, K, device=dev) * 0.03, fmt)
        ts = (xq[2] * wq[2]).reshape(1).contiguous() if fmt == "nv4" else None
        P = R.dequant_copy(*xq, fmt, M, K).double() @ R.dequant_copy(*wq, fmt, N, K).double().t()
        bias = torch.randn(N, device=dev) * 0.1
        k2 = fp4gb.get(dict(CFG, fmt=fi, epi=2))
        out = torch.empty(3, B, Ntok, H, 64, device=dev, dtype=torch.float16)
        k2(xq[0], wq[0], xq[1], wq[1], M=M, N=N, K=K, ts=ts, bias=bias, C=out, sin=sin, cos=cos, Ntok=Ntok, prefix=prefix, d=d, H=H)
        y = (P + bias.double()).view(M, 3, H, 64)
        lo, hi = y[..., :32].clone(), y[..., 32:].clone()
        tok = torch.arange(M, device=dev) % Ntok
        pp = (tok - prefix).clamp(min=0)
        c_, s_ = cos[pp].double()[:, None, None, :], sin[pp].double()[:, None, None, :]       # [M,1,1,64]
        rl = lo * c_[..., :32] - hi * s_[..., :32]
        rh = hi * c_[..., 32:] + lo * s_[..., 32:]
        use = (tok >= prefix)[:, None, None, None]
        y2 = y.clone()
        y2[:, :2, :, :32] = torch.where(use, rl[:, :2], lo[:, :2])
        y2[:, :2, :, 32:] = torch.where(use, rh[:, :2], hi[:, :2])
        ref = y2.view(B, Ntok, 3, H, 64).permute(2, 0, 1, 3, 4)
        print(f"{fmt} M={M} EPI2 qkv+RoPE+layout fp16: relL2={rel(out.float(), ref):.2e}", flush=True)
        # ---- EPI 1 residual (proj K=d N=d ; fc2 K=ff N=d)
        for kind, N, K in (("proj", d, d), ("fc2", d, ff)):
            xq = R.quantize_matrix(heavy(M, K), fmt)
            wq = R.quantize_matrix(torch.randn(N, K, device=dev) * 0.03, fmt)
            ts = (xq[2] * wq[2]).reshape(1).contiguous() if fmt == "nv4" else None
            P = R.dequant_copy(*xq, fmt, M, K).double() @ R.dequant_copy(*wq, fmt, N, K).double().t()
            bias, gamma, Rs = torch.randn(N, device=dev) * 0.1, torch.rand(N, device=dev), torch.randn(M, N, device=dev)
            k1 = fp4gb.get(dict(CFG, fmt=fi, epi=1))
            o = torch.empty(M, N, device=dev, dtype=torch.float32)
            k1(xq[0], wq[0], xq[1], wq[1], M=M, N=N, K=K, ts=ts, bias=bias, gamma=gamma, R=Rs, C=o)
            print(f"{fmt} M={M} EPI1 {kind} residual fp32: relL2={rel(o, Rs.double() + gamma.double() * (P + bias.double())):.2e}", flush=True)
        # ---- EPI 3 fc1 gelu + quantize-out
        N, K = ff, d
        xq = R.quantize_matrix(heavy(M, K), fmt)
        wq = R.quantize_matrix(torch.randn(N, K, device=dev) * 0.03, fmt)
        ts = (xq[2] * wq[2]).reshape(1).contiguous() if fmt == "nv4" else None
        P = R.dequant_copy(*xq, fmt, M, K).double() @ R.dequant_copy(*wq, fmt, N, K).double().t()
        bias = torch.randn(N, device=dev) * 0.1
        z = P + bias.double()
        gz = torch.nn.functional.gelu(z)
        tso = (gz.abs().amax() * (2.0 / 2688.0)).float().reshape(1)
        vec = 16 if fmt == "nv4" else 32
        Qo = torch.zeros(M, N // 2, dtype=torch.uint8, device=dev)
        SFo = torch.zeros((M + 127) // 128, N // (vec * 4), 32, 4, 4, dtype=torch.uint8, device=dev)
        k3 = fp4gb.get(dict(CFG, fmt=fi, epi=3))
        k3(xq[0], wq[0], xq[1], wq[1], M=M, N=N, K=K, ts=ts, bias=bias, Qo=Qo, SFo=SFo, tso=tso if fmt == "nv4" else None)
        qr, sr = ref_quant(gz.float(), fmt, tso[0])
        deq = R.dequant_copy(Qo, SFo, tso if fmt == "nv4" else None, fmt, M, N)
        deq_r = R.dequant_copy(qr, sr, tso if fmt == "nv4" else None, fmt, M, N)
        print(f"{fmt} M={M} EPI3 fc1 GELU+quant-out: identical codes {(Qo == qr).float().mean().item() * 100:.3f}%  identical scales {(SFo == sr).float().mean().item() * 100:.3f}%  "
              f"relL2(deq vs GELU) {rel(deq, gz):.3e} (ref quant of exact GELU {rel(deq_r, gz):.3e})", flush=True)
        # ---- EPI 4 gelu bf16 (used for calibration)
        k4 = fp4gb.get(dict(CFG, fmt=fi, epi=4))
        o4 = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        k4(xq[0], wq[0], xq[1], wq[1], M=M, N=N, K=K, ts=ts, bias=bias, C=o4)
        print(f"{fmt} M={M} EPI4 fc1 GELU bf16: relL2={rel(o4.float(), gz):.2e}", flush=True)
