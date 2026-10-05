"""Round 14 step 2: producer kernels (lnq14.cu) vs the fp4_ref spec.
  quantx : bit-exact q and scale bytes for fp16 / bf16 / fp32 input, nv4 (static ts) and mx4
  lnq    : LN + quant; the LN part has its own fp32 arithmetic so the comparison is (a) fraction of identical codes vs ref-quantised torch LN, (b) dequantised relL2 vs the LN output"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl"))
import torch
import ext14
import fp4util as U
from fp4util import R

dev = "cuda"
torch.manual_seed(1)


def heavy(M, K):
    x = torch.randn(M, K, device=dev)
    x[:, 5] *= 30
    x[:, K // 3] *= 12
    x[7] = 0
    return x


def ref_quant(x, fmt, ts):
    xf = x.float()
    r, c = xf.shape
    if fmt == "nv4":
        sf8 = R._nv4_sf(xf.reshape(r, c // 16, 16).abs().amax(-1), ts)
        return R.pack_nibbles(R._codes_nv4(xf, sf8, ts, False, None)), R.to_blocked(sf8.view(torch.uint8))
    byte = R._mx4_byte(xf.reshape(r, c // 32, 32).abs().amax(-1))
    return R.pack_nibbles(R._codes_mx4(xf, byte, False, None)), R.to_blocked(byte.to(torch.uint8))


def bufs(M, K, fmt):
    vec = 16 if fmt == "nv4" else 32
    q = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
    sf = torch.zeros((M + 127) // 128, K // (vec * 4), 32, 4, 4, dtype=torch.uint8, device=dev)
    return q, sf


e = ext14.lnq_ext()
for fmt, fi in (("nv4", 0), ("mx4", 1)):
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for (M, K) in ((2058, 1024), (2258, 768), (2058, 4096)):
            x = heavy(M, K).to(dt)
            ts = (x.float().abs().amax() * (2.0 / 2688.0)).reshape(1)
            q, sf = bufs(M, K, fmt)
            e.quantx(x, ts if fmt == "nv4" else None, q, sf, fi)
            qr, sr = ref_quant(x, fmt, ts[0])
            print(f"quantx {fmt} {str(dt)[6:]:8s} M={M} K={K}: q_exact={bool((q == qr).all())} sf_exact={bool((sf == sr).all())}", flush=True)
    for (M, D) in ((2058, 1024), (2258, 768)):
        x = heavy(M, D) * 3 + 0.5
        w = torch.rand(D, device=dev) + 0.5
        b = torch.randn(D, device=dev) * 0.1
        y = torch.nn.functional.layer_norm(x, (D,), w, b, 1e-6)
        ts = (y.abs().amax() * (2.0 / 2688.0)).reshape(1)
        q, sf = bufs(M, D, fmt)
        e.lnq(x, w, b, 1e-6, ts if fmt == "nv4" else None, q, sf, fi)
        qr, sr = ref_quant(y, fmt, ts[0])
        deq = R.dequant_copy(q, sf, ts if fmt == "nv4" else None, fmt, M, D)
        deq_r = R.dequant_copy(qr, sr, ts if fmt == "nv4" else None, fmt, M, D)
        print(f"lnq    {fmt} M={M} D={D}: identical codes {(q == qr).float().mean().item() * 100:.3f}%  identical scales {(sf == sr).float().mean().item() * 100:.3f}%  "
              f"relL2(deq vs LN) {((deq - y).norm() / y.norm()).item():.3e}  (ref quant {((deq_r - y).norm() / y.norm()).item():.3e})", flush=True)
