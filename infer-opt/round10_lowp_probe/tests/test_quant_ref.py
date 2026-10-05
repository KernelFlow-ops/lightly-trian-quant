"""Gate A for quant_ref.py: bit-exact vs the train-opt kernels / references it was adapted from.
   PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src python tests/test_quant_ref.py
 - mx8: vs r17 `quant_dual` (CUDA, row copy): e4m3 bytes and ue8m0 scale bytes
 - nv4 / mx4: vs r20 `fp4_ref.quantize_matrix` (packed codes, blocked scale bytes, ts)
 - qdq == dequant(quantize(.)) round trips; int8 / fp16 sanity."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "impl"))
import torch
import quant_ref as qr
import bench10 as b
from round17_mxfp8_sm120.impl import mx8 as r17
from round20_fp4_probe.impl import fp4_ref as r20

r17._BUILD = Path("/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build/r17_mx8")   # never build into train-opt-tmp

def data(R, C, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(R, C, device="cuda", generator=g)
    if kind == "outlier":
        x[:, :3] *= 60; x[::7, 5] *= 400
    if kind == "wide":
        x = x * torch.exp2(torch.randint(-12, 12, (R, C // 32, 1), device="cuda", generator=g).float()).repeat_interleave(32, 1).reshape(R, C)
    if kind == "zeros":
        x[: R // 4] = 0; x[:, 64:96] = 0
    return x

ok = True
res = {}
for (R, C) in ((2058, 1024), (2258, 4096), (130, 768)):
    for kind in ("normal", "outlier", "wide", "zeros"):
        x = data(R, C, kind, R + C).to(torch.bfloat16)
        r = {}
        # ---- mx8 vs r17
        if C % 128 == 0:
            m = r17.quantize(x, row=True, col=False)
            q_ref = m.qr.view(torch.uint8) if m.qr.dtype != torch.uint8 else m.qr
            sf_ref = qr.from_blocked(m.sfr, R, C // 32)
            q, byte = qr.mx8_quantize(x)
            r["mx8_q_equal"] = bool((q.view(torch.uint8) == q_ref).all())
            r["mx8_scale_equal"] = bool((byte == sf_ref).all())
        # ---- nv4 / mx4 vs r20
        for fmt in ("nv4", "mx4"):
            qref, sfref, tsref = r20.quantize_matrix(x, fmt)
            if fmt == "nv4":
                q, sf8, ts = qr.nv4_quantize(x)
                eq_s = bool((qr.to_blocked(sf8.view(torch.uint8)) == sfref).all()); eq_ts = bool((ts == tsref).all())
            else:
                q, byte = qr.mx4_quantize(x)
                eq_s = bool((qr.to_blocked(byte) == sfref).all()); eq_ts = True
            r[fmt + "_q_equal"] = bool((q == qref).all()); r[fmt + "_scale_equal"] = eq_s; r[fmt + "_ts_equal"] = eq_ts
        # ---- round trips and error levels
        for fmt in ("mx8", "nv4", "mx4", "int8", "fp16"):
            d = qr.qdq(x, fmt)
            r[fmt + "_rel_l2"] = ((d - x.float()).norm() / x.float().norm().clamp(min=1e-30)).item()
        res[f"{R}x{C}|{kind}"] = r
        bad = [k for k, v in r.items() if k.endswith("equal") and not v]
        ok &= not bad
        print(f"{R}x{C} {kind:8s} {'OK' if not bad else 'MISMATCH ' + str(bad)}  " + " ".join(f"{k}={v:.3e}" for k, v in r.items() if k.endswith("l2")), flush=True)
b.merge_results(HERE / "results.json", {"quant_ref_test": {"all_bit_exact": bool(ok), "cases": res}})
print("ALL BIT-EXACT" if ok else "FAILED")
