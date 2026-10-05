"""Round 13 step-2 op-level sanity of the own MXFP8 GEMM (mgx): every kernel vs the fp32 product of the DEQUANTISED operands + the epilogue in torch (not a gate).
  python tests/test_mgx.py --d 1024 --H 16 [--bench]        (correctness needs no lock; --bench takes the dual GPU lock)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
for p in (HERE / "impl", ROOT / "round10_lowp_probe" / "impl", ROOT / "common"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import mx13  # noqa: E402
import mgx  # noqa: E402
import quant_ref as qr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=1024)
ap.add_argument("--H", type=int, default=16)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--Ms", default="2058,2258")
ap.add_argument("--kinds", default="qkv,proj,fc1,fc2,plain")
a = ap.parse_args()
d, H, ff = a.d, a.H, 4 * a.d
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
E = mx13.ext()
res = {}


def deq(q, sf, R, K):
    rm, rk = mx13.pad128(R) // 128, K // 128
    byte = qr.from_blocked(sf.view(rm, rk, 32, 4, 4), R, K // 32)
    return qr.mx8_dequant(q.view(torch.float8_e4m3fn), byte)


def mk(shape, scale=1.0):
    x = torch.randn(*shape, device=dev) * scale
    x[:, ::97] *= 8.0
    return x


def rope_ref(Y, bias, sin, cos, B, Ntok, prefix):
    v = (Y + bias).view(B, Ntok, 3, H, 64)
    ref = torch.empty(3, B, Ntok, H, 64, device=dev)
    for wh in range(3):
        t = v[:, :, wh].clone()
        if wh < 2:
            tt = t[:, prefix:]
            lo, hi = tt[..., :32], tt[..., 32:]
            cc, ss = cos[None, :, None, :], sin[None, :, None, :]
            t[:, prefix:] = torch.cat([lo * cc[..., :32] - hi * ss[..., :32], hi * cc[..., 32:] + lo * ss[..., 32:]], -1)
        ref[wh] = t
    return ref


def metrics(out, ref):
    out, ref = out.float(), ref.float()
    return dict(rel_l2=((out - ref).norm() / ref.norm()).item(), cos=F.cosine_similarity(out.flatten().double(), ref.flatten().double(), dim=0).item(),
                max_abs=(out - ref).abs().max().item(), finite=bool(torch.isfinite(out).all()))


for M in [int(x) for x in a.Ms.split(",")]:
    Ntok = 1029 if M == 2058 else 1129
    B, prefix = M // Ntok, Ntok - 1024
    for kind in a.kinds.split(","):
        N, K = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff), "plain": (4 * d, d)}[kind]
        X, Wf = mk((M, K), 0.5), torch.randn(N, K, device=dev) * 0.03
        xb, W = mx13.ActBuf(M, K), mx13.MXW(Wf)
        E.quant_rows(X, xb.q, xb.sf)
        Y = deq(xb.q, xb.sf, M, K) @ deq(W.q, W.sf, N, K).t()
        bias = torch.randn(N, device=dev) * 0.1
        cfg = mgx.pick(kind, N, K, M)
        if kind == "qkv":
            sin, cos = torch.randn(1024, 64, device=dev), torch.randn(1024, 64, device=dev)
            fn = lambda: mgx.qkv_rope(xb, W, bias, sin, cos, Ntok, prefix, H)
            m = metrics(fn(), rope_ref(Y, bias, sin, cos, B, Ntok, prefix))
        elif kind in ("proj", "fc2"):
            R, g = torch.randn(M, N, device=dev), torch.rand(N, device=dev)
            fn = lambda: mgx.gemm_res(xb, W, bias, g, R)
            m = metrics(fn(), R + g * (Y + bias))
        elif kind == "fc1":
            ob = mx13.ActBuf(M, N)
            fn = lambda: mgx.gemm_gelu_q(xb, W, bias, ob)
            fn()
            ref = F.gelu(Y + bias)
            q, byte = qr.mx8_quantize(ref)
            m = metrics(deq(ob.q, ob.sf, M, N), ref)
            m["q_bytes_equal_vs_quant_ref(ref gelu)"] = (q.view(torch.uint8) == ob.q).float().mean().item()
            m["sf_bytes_equal"] = (qr.from_blocked(ob.sf.view(mx13.pad128(M) // 128, N // 128, 32, 4, 4), M, N // 32) == byte).float().mean().item()
            m["rel_l2_of_quantization_itself"] = ((qr.mx8_dequant(q, byte) - ref).norm() / ref.norm()).item()
        else:
            fn = lambda: mgx.gemm_plain(xb, W)
            m = metrics(fn(), Y)
        m["cfg"] = cfg
        if a.bench:
            from lock2 import gpu_lock2
            with gpu_lock2():
                for _ in range(10):
                    fn()
                g_ = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_):
                    for _ in range(20):
                        fn()
                g_.replay(); torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); g_.replay(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) / 20 * 1e3)
                m["us"] = sorted(ts)[2]
                m["tflops"] = 2 * M * N * K / m["us"] / 1e6
        res[f"{kind} M={M}"] = m
        print(f"{kind:5s} M={M} N={N:5d} K={K:5d}: " + "  ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items() if k != "cfg"))
Path(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"test_mgx_d{d}.json").write_text(json.dumps(res, indent=1))
