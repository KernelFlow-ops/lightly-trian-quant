"""Round 14 step-1 op-level sanity (not a gate): fp4q2 static-ts quantiser bit-exact vs fp4_ref, QuACK ts-epilogue GEMMs vs the fp64 product of the dequantised operands."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl"))
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import fp4util as U
from fp4util import R

torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item(), torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def heavy(M, K):
    x = torch.randn(M, K, device=dev)
    x[:, 5] *= 30
    x[:, K // 3] *= 12
    return x


def ref_nv4(x, ts):
    xf = x.float()
    r, c = xf.shape
    sf8 = R._nv4_sf(xf.reshape(r, c // 16, 16).abs().amax(-1), ts)
    return R.pack_nibbles(R._codes_nv4(xf, sf8, ts, False, None)), R.to_blocked(sf8.view(torch.uint8))


def ref_mx4(x):
    xf = x.float()
    r, c = xf.shape
    byte = R._mx4_byte(xf.reshape(r, c // 32, 32).abs().amax(-1))
    return R.pack_nibbles(R._codes_mx4(xf, byte, False, None)), R.to_blocked(byte.to(torch.uint8))


for fmt in ("nv4", "mx4"):
    calib = U.Calib()
    for (name, M, N, K) in (("qkv", 2058, 3072, 1024), ("proj", 2258, 1024, 1024), ("fc1", 2058, 4096, 1024), ("fc2", 2058, 1024, 4096)):
        x = heavy(M, K).to(torch.bfloat16)
        site = U.Site(fmt, calib)
        calib.on = True
        xq = site.quant(x)
        if fmt == "nv4":
            q_ref, sf_ref = ref_nv4(x, site.ts2[0])
        else:
            q_ref, sf_ref = ref_mx4(x)
        qe, se = bool((xq.q == q_ref).all()), bool((xq.sf.view(torch.uint8)[: sf_ref.shape[0]] == sf_ref).all())
        W = torch.randn(N, K, device=dev) * 0.03
        bias = torch.randn(N, device=dev) * 0.1
        gamma = torch.rand(N, device=dev)
        lin = U.FP4Linear(W, bias, fmt, calib, gamma=gamma)
        resid = torch.randn(M, N, device=dev)
        deq_x, deq_w = U.dequant_fq(xq), lin.dequant()
        P = deq_x.double() @ deq_w.double().t()
        y = lin.fwd_bias(xq)
        r1 = rel(y.float(), P + bias.double())
        calib.on = False
        o = lin.fwd_res(xq, resid)
        r2 = rel(o, resid.double() + gamma.double() * (P + bias.double()))
        z = lin.fwd_gelu(xq)
        zf = P + bias.double()
        r3 = rel(z.float(), torch.nn.functional.gelu(zf))
        yp = lin.fwd_plain(xq)
        r4 = rel(yp.float(), P)
        print(f"{fmt} {name:4s} M={M}: quant q_exact={qe} sf_exact={se} | bias relL2={r1[0]:.2e} max={r1[1]:.2e} | res relL2={r2[0]:.2e} | gelu relL2={r3[0]:.2e} | plain relL2={r4[0]:.2e} cos={r4[2]:.7f}", flush=True)
