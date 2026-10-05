"""R24 gate A (copy of round21 tests/test_fp4q.py with the module under test replaced by round24 impl/fp4q2.py, plus the parts "fast" and "philox7"):
fused FP4 quantizer (impl/fp4q2.py, impl/fp4_kernels2.cu) against the frozen bit-exact reference round20_fp4_probe/impl/fp4_ref.py and against round21 fp4q.

  python tests/test_fp4q.py [--part rn|sr|exact|weight|state|gemm|all] [--quick]       -> profile/gateA_fp4q.json (merged per part)

Parts
  rn      RN, bit-exact (q, scale bytes, xp) for nv4 / mx4, row and transposed copy, RHT off / on, prefix off / on, R x C sweep, data kinds incl. spikes / zeros / 1e-30 / 1e4
  sr      explicit uniforms == fp4_ref.e2m1_sr bit for bit; in-kernel Philox == explicit philox_uniform stream; unbiasedness; same seed same bits; next step differs
  exact   device ts == amax/(6*448) (fp32 division, and the reciprocal-multiply that fp4_ref uses on CUDA); delayed = 2 * previous amax
  weight  2D-scale weight quantizer == fp4_ref.quantize_weight(two_d=True)
  state   amax_roll + delayed amax bookkeeping, CUDA-graph capture of roll + quantizers, replay == eager
  div     Markstein reciprocal-based division of the nv4 path vs IEEE division (brute force, ~1e10 samples)
  codes   element rounding: hardware RN vs fp4_ref thresholds on all 2^32 fp32 patterns; SR code vs fp4_ref.e2m1_sr on 4e7 random / special inputs
  headroom  headroom argument (default bit-identical to the pre-change build, headroom 8 / 3 / 0.5 ts bits) + roll() saturation monitor + graph replay
  gemm    quantizer output -> QuACK nv4 GEMM (fwd / dgrad / wgrad layouts) vs the dequantized fp32 product (rel L2 <= 2e-3)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND.parent / "round20_fp4_probe" / "impl"))
import numpy as np
import torch

import fp4_ref as R  # round20 (frozen reference, read only)
import fp4q2 as fp4q
from common.bench import gpu_lock

dev = "cuda"
OUT = ROUND / "profile" / "gateA_fp4q2.json"
RS = (7203, 7903, 14406, 15806, 1029, 33, 28672, 57344)
CS = (768, 1024, 2304, 3072, 4096)
KINDS = ("gauss", "spike8", "spike3", "zeros", "tiny", "big")
PREFIX = {7203: (7, 1029, 5), 7903: (7, 1129, 105), 14406: (14, 1029, 5), 15806: (14, 1129, 105), 1029: (1, 1029, 5)}
FAILS: list = []


def data(kind, rows, cols, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device=dev)
    if kind in ("spike8", "spike3"):  # DINOv3-like: outlier channels x8, outlier rows, one big element
        x[:, ::97] *= 8.0
        x[::53] *= 4.0
        if rows > 5 and cols > 5:
            x[3, 5] = 200.0
    if kind == "spike3":  # + x1e3 outlier channels, x1e4 massive rows (cls / register-like)
        x[:, ::131] *= 1e3
        x[rows // 2] *= 1e4
        x[2] *= 1e4
    elif kind == "zeros":
        x.zero_()
    elif kind == "tiny":
        x *= 1e-30
    elif kind == "big":
        x *= 1e4
    return x.to(dtype)


def nbad(a, b):
    if a.shape != b.shape:
        return -1
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum())


def patch_of(x, prefix):
    B, N, npre = prefix
    return x.view(B, N, -1)[:, npre:, :].reshape(B * (N - npre), -1).contiguous()


def ref_copies(x, fmt, rht_r, rht_c, prefix, sr_r=False, sr_c=False, u_row=None, u_col=None, row=True, col=True):
    """(row FP4Tensor or None, col FP4Tensor or None) from fp4_ref; the col copy is built from the patch tokens only in prefix mode."""
    rr = R.quantize(x, fmt, row=True, col=False, rht_row=rht_r, sr=sr_r, u_row=u_row) if row else None
    if not col:
        return rr, None
    xc = x if prefix is None else patch_of(x, prefix)
    rc = R.quantize(xc, fmt, row=False, col=True, rht_col=rht_c, sr=sr_c, u_col=u_col, rpad=None if prefix is None else xc.shape[0])
    return rr, rc


def check_vs_ref(tag, x, fmt, rht_r, rht_c, prefix, *, amax="ts", sr_r=False, sr_c=False, seed=None, u_row=None, u_col=None, sr_off=0, div_true=False, row=True, col=True, rr=None, rc=None):
    if rr is None and rc is None:
        rr, rc = ref_copies(x, fmt, rht_r, rht_c, prefix, sr_r, sr_c, u_row, u_col, row, col)
    ts = [rr.ts if rr is not None and rr.ts is not None else torch.ones(1, device=dev), rc.tst if rc is not None and rc.tst is not None else torch.ones(1, device=dev)] if fmt == "nv4" else None
    tsin = torch.cat([t.reshape(1) for t in ts]) if ts is not None else None
    k = fp4q.quant_act(x, fmt, row=row, col=col, rht_row=rht_r, rht_col=rht_c, sr_row=sr_r, sr_col=sr_c, prefix=prefix, amax=amax, ts=tsin, seed=seed, sr_offset=sr_off,
                       u_row=u_row, u_col=u_col, div_true=div_true)
    res = {}
    if row:
        res["q"], res["sf"] = nbad(k.q, rr.q), nbad(k.sf, rr.sf)
        if fmt == "nv4":
            res["ts"] = int(k.ts.view(torch.int32) != rr.ts.reshape(1).view(torch.int32))
    if col:
        res["qt"], res["sft"] = nbad(k.qt, rc.qt), nbad(k.sft, rc.sft)
        if fmt == "nv4":
            res["tst"] = int(k.tst.view(torch.int32) != rc.tst.reshape(1).view(torch.int32))
    if prefix is not None:
        B, N, npre = prefix
        ref_xp = x.view(B, N, -1)[:, :npre, :].reshape(B * npre, -1).to(torch.bfloat16)
        res["xp"] = nbad(k.xp, ref_xp)
    bad = {a: b for a, b in res.items() if b != 0}
    if bad:
        FAILS.append({"case": tag, "bad": bad})
    return res, bad


# ------------------------------------------------------------------------------------------------ parts
def part_rn(quick):
    out = {"cases": 0, "fail": 0, "fail_list": []}
    t0 = time.time()
    idx = 0
    # 1) full R x C sweep, both formats, RHT off / on(row and col), kinds rotating
    for ri, Rr in enumerate(RS):
        for ci, C in enumerate(CS):
            if quick and (Rr > 20000 or ci % 2):
                continue
            for fi, fmt in enumerate(("nv4", "mx4")):
                for rht in (False, True):
                    kind = KINDS[(ri + ci + fi + int(rht)) % len(KINDS)]
                    x = data(kind, Rr, C, seed=idx)
                    prefix = PREFIX.get(Rr) if (idx % 2 == 0) else None
                    for exact_mode in ((False,) if kind != "gauss" else (False, True)):
                        if prefix is not None and not exact_mode:
                            # prefix-mode + plain-mode both
                            check_vs_ref(f"rn R={Rr} C={C} {fmt} rht={rht} {kind} prefix", x, fmt, rht, rht, prefix)
                            out["cases"] += 1
                        check_vs_ref(f"rn R={Rr} C={C} {fmt} rht={rht} {kind} plain" + (" amax=exact" if exact_mode else ""), x, fmt, rht, rht, None, amax="exact" if exact_mode and fmt == "nv4" else "ts")
                        out["cases"] += 1
                    del x
                    idx += 1
                    torch.cuda.empty_cache()
    # 2) all kinds on small shapes, all RHT combinations (row / col independent), row-only / col-only, fp32 input, C % 128 == 64
    for kind in KINDS:
        for fmt in ("nv4", "mx4"):
            for rr_, rc_ in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for Rr, C in ((33, 768), (1029, 1024), (7203, 576)):
                    x = data(kind, Rr, C, seed=7)
                    check_vs_ref(f"rn-small R={Rr} C={C} {fmt} {kind} rht=({rr_},{rc_})", x, fmt, bool(rr_), bool(rc_), None)
                    out["cases"] += 1
                    if Rr == 7203:
                        check_vs_ref(f"rn-small prefix R={Rr} C={C} {fmt} {kind} rht=({rr_},{rc_})", x, fmt, bool(rr_), bool(rc_), PREFIX[Rr])
                        check_vs_ref(f"rn-small row-only R={Rr} C={C} {fmt} {kind}", x, fmt, bool(rr_), bool(rc_), None, col=False)
                        check_vs_ref(f"rn-small col-only R={Rr} C={C} {fmt} {kind}", x, fmt, bool(rr_), bool(rc_), PREFIX[Rr], row=False)
                        check_vs_ref(f"rn-small col-only plain R={Rr} C={C} {fmt} {kind}", x, fmt, bool(rr_), bool(rc_), None, row=False)
                        out["cases"] += 4
                    xf = data(kind, Rr, C, seed=9, dtype=torch.float32)
                    if kind in ("gauss", "spike3"):
                        xf = xf * 1.2345678
                    check_vs_ref(f"rn-fp32 R={Rr} C={C} {fmt} {kind} rht=({rr_},{rc_})", xf, fmt, bool(rr_), bool(rc_), PREFIX.get(Rr) if Rr == 7203 else None)
                    out["cases"] += 1
    # 3) negative zero / sign handling and exact ties: values on the e2m1 decision thresholds
    thr = torch.tensor([0.0, -0.0, 0.25, -0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, 0.2499, 0.2501, 5.0001, 4.9999], device=dev)
    for fmt in ("nv4", "mx4"):
        x = (thr.repeat(64 * 64 // 16).view(64, 64) * 0.5).clone()  # per-block amax 3.5 -> exact-looking scales; plus random sign flips
        x2 = (torch.randn(64, 64, device=dev) * 0.0 + thr.repeat(64 * 64 // 16).view(64, 64)).to(torch.bfloat16)
        for xx in (x.to(torch.bfloat16), x2, (-x).to(torch.bfloat16)):
            check_vs_ref(f"rn-thresholds {fmt}", xx.contiguous(), fmt, False, False, None)
            out["cases"] += 1
    out["fail"] = sum(1 for f in FAILS if f["case"].startswith("rn"))
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("rn")][:20]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_sr(quick):
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    seed = torch.tensor([123456789012345], dtype=torch.int64, device=dev)
    # Philox self test: known-answer vector of Philox4x32-10 and kernel stream == numpy implementation
    def philox_np(c0, c1, c2, c3, k0, k1):
        M0, M1, W0, W1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57), np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
        m = np.uint64(0xFFFFFFFF)
        c0, c1, c2, c3, k0, k1 = [np.asarray(v, dtype=np.uint64) for v in (c0, c1, c2, c3, k0, k1)]
        for _ in range(10):
            p0, p1 = M0 * c0, M1 * c2
            c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m, p1 & m, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m
            k0, k1 = (k0 + W0) & m, (k1 + W1) & m
        return c0, c1, c2, c3

    kat = philox_np(0, 0, 0, 0, 0, 0)
    out["philox_known_answer_ok"] = [int(v) for v in kat] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    n = 4096
    u_dev = fp4q.philox_uniform(n, seed, 5).cpu().numpy()
    s = int(seed.item())
    i8 = np.arange(n // 8, dtype=np.uint64)
    ws = philox_np(i8, np.uint64(5), np.uint64(0), np.uint64(0), np.uint64(s & 0xFFFFFFFF), np.uint64((s >> 32) & 0xFFFFFFFF))
    words = np.stack(ws, 1)  # [n/8, 4]
    halves = np.stack([words & np.uint64(0xFFFF), words >> np.uint64(16)], axis=2).reshape(-1)  # element j: word j//2, half j%2
    u_np = (halves * np.uint64(2) + np.uint64(1)).astype(np.float64) * 2.0 ** -17
    out["philox_kernel_equals_numpy"] = bool(np.array_equal(u_dev.astype(np.float64), u_np))
    if not (out["philox_known_answer_ok"] and out["philox_kernel_equals_numpy"]):
        FAILS.append({"case": "sr philox self test", "bad": {}})
    out["u_mean"] = float(u_dev.mean())
    # bit-exact: explicit u == fp4_ref; in-kernel Philox == explicit philox_uniform stream
    sr_off = 3
    cfg = [(7203, 1024, "nv4", True, True), (7203, 1024, "mx4", True, True), (7903, 768, "nv4", False, True), (1029, 4096, "mx4", False, False), (33, 768, "nv4", True, False),
           (14406, 768, "nv4", True, True), (7203, 4096, "nv4", False, False)]
    if quick:
        cfg = cfg[:4]
    for ci, (Rr, C, fmt, rht_r, rht_c) in enumerate(cfg):
        for prefix in (None, PREFIX.get(Rr)):
            if prefix is None and ci % 2:
                continue
            kind = ("gauss", "spike3", "spike8")[ci % 3]
            x = data(kind, Rr, C, seed=ci)
            Kt = (prefix[0] * (prefix[1] - prefix[2])) if prefix else fp4q.pad128(Rr)
            u_row = fp4q.philox_uniform(Rr * C, seed, 2 * sr_off, dev).view(Rr, C)
            u_col = fp4q.philox_uniform(C * Kt, seed, 2 * sr_off + 1, dev).view(C, Kt)
            tag = f"sr R={Rr} C={C} {fmt} {kind} rht=({int(rht_r)},{int(rht_c)}) prefix={prefix is not None}"
            rr, rc = ref_copies(x, fmt, rht_r, rht_c, prefix, True, True, u_row, u_col)
            r1, b1 = check_vs_ref(tag + " explicit-u", x, fmt, rht_r, rht_c, prefix, sr_r=True, sr_c=True, u_row=u_row, u_col=u_col, rr=rr, rc=rc)
            r2, b2 = check_vs_ref(tag + " philox", x, fmt, rht_r, rht_c, prefix, sr_r=True, sr_c=True, seed=seed, sr_off=sr_off, rr=rr, rc=rc)
            # one copy SR, the other RN
            rr2, rc2 = ref_copies(x, fmt, rht_r, rht_c, prefix, False, True, None, u_col)
            check_vs_ref(tag + " row-RN col-SR", x, fmt, rht_r, rht_c, prefix, sr_r=False, sr_c=True, u_col=u_col, rr=rr2, rc=rc2)
            out["cases"] += 3
            del x, u_row, u_col
            torch.cuda.empty_cache()
    # statistics: unbiasedness (rms of the mean error over S seeds vs the expected SR noise), fixed ts so the scales are the same for every draw
    stats = {}
    S = 256 if not quick else 64
    for fmt in ("nv4", "mx4"):
        for copy in ("row", "col"):
            Rr, C = 128, 1024
            x = data("gauss", Rr, C, seed=11)
            rr0, rc0 = ref_copies(x, fmt, False, False, None, False, False, None, None)
            ts = torch.cat([rr0.ts.reshape(1), rc0.tst.reshape(1)]) if fmt == "nv4" else None
            acc = None
            for sidx in range(S):
                sd = torch.tensor([1000 + sidx], dtype=torch.int64, device=dev)
                k = fp4q.quant_act(x, fmt, row=(copy == "row"), col=(copy == "col"), sr_row=True, sr_col=True, amax="ts", ts=ts, seed=sd, sr_offset=1)
                d = R.dequant_copy(k.q, k.sf, k.ts, fmt, Rr, C) if copy == "row" else R.dequant_copy(k.qt, k.sft, k.tst, fmt, C, k.kt)[:, :Rr]
                acc = d.double() if acc is None else acc + d.double()
            mean = (acc / S).float()
            xe = x.float() if copy == "row" else x.float().t()
            # expected noise: per element var of the SR draw, from the RN-free value v = x / (sf * ts)
            base = rr0 if copy == "row" else rc0
            sfm = R.from_blocked(base.sf if copy == "row" else base.sft, Rr if copy == "row" else C, (C if copy == "row" else base.qt.shape[1] * 2) // R.VEC[fmt])
            sv = sfm.view(torch.float8_e4m3fn).float() if fmt == "nv4" else torch.exp2(sfm.float() - 127.0)
            ncol = xe.shape[1]
            scale = sv.repeat_interleave(R.VEC[fmt], 1)[:, :ncol]
            tsv = (rr0.ts if copy == "row" else rc0.tst) if fmt == "nv4" else None
            if tsv is not None:
                scale = scale * tsv
            v = (xe / scale.clamp_min(1e-38)).abs()
            ok = (v <= 6.0) & (scale > 0)
            g = R._grid(dev)
            a = v.clamp(max=6.0)
            ix = (torch.bucketize(a, g, right=True) - 1).clamp(0, 6)
            lo, hi = g[ix], g[ix + 1]
            p = (a - lo) / (hi - lo)
            var = ((hi - lo) ** 2 * p * (1 - p)) * scale ** 2
            exp_rms = float(torch.sqrt((var[ok]).mean() / S))
            bias_rms = float(torch.sqrt(((mean - xe)[ok] ** 2).mean()))
            sat_frac = float((~ok).float().mean())
            stats[f"{fmt}_{copy}"] = {"bias_rms_over_expected_noise": round(bias_rms / exp_rms, 4), "saturated_frac_excluded": round(sat_frac, 4), "seeds": S}
            if not (0.9 < bias_rms / exp_rms < 1.12):
                FAILS.append({"case": f"sr unbiasedness {fmt} {copy}", "bad": stats[f"{fmt}_{copy}"]})
            out["cases"] += 1
    out["unbiasedness"] = stats
    # same seed -> same bits; next step (seed + 1 through FP4State.roll) -> different bits
    st = fp4q.FP4State(2, seed=777)
    x = data("gauss", 1029, 1024, seed=5)
    outs = []
    for _ in range(2):
        k = fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="exact", seed=st.seed, sr_offset=2)
        outs.append(k)
    same = nbad(outs[0].q, outs[1].q) + nbad(outs[0].qt, outs[1].qt)
    st.roll()
    k3 = fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="exact", seed=st.seed, sr_offset=2)
    k4 = fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="exact", seed=st.seed, sr_offset=3)
    out["same_seed_bad_bytes"] = same
    out["next_step_changed_frac_q"] = round(nbad(outs[0].q, k3.q) / outs[0].q.numel(), 4)
    out["next_step_changed_frac_qt"] = round(nbad(outs[0].qt, k3.qt) / outs[0].qt.numel(), 4)
    out["other_offset_changed_frac_q"] = round(nbad(k3.q, k4.q) / k3.q.numel(), 4)
    out["seed_after_roll"] = int(st.seed.item())
    if same != 0 or out["next_step_changed_frac_q"] < 0.01 or out["next_step_changed_frac_qt"] < 0.01 or out["other_offset_changed_frac_q"] < 0.01:
        FAILS.append({"case": "sr seed/step determinism", "bad": {k_: out[k_] for k_ in ("same_seed_bad_bytes", "next_step_changed_frac_q", "next_step_changed_frac_qt", "other_offset_changed_frac_q")}})
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("sr")][:20]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_exact(quick):
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    two = torch.tensor(6.0 * 448.0, device=dev)
    res = {}
    # a) ts of the exact path vs fp32 division, over many random amax values (both div modes) and vs fp4_ref (CUDA = reciprocal multiply; CPU = division)
    n_ok = {"true_div_vs_torch_tensor_div": 0, "default_vs_fp4_ref_cuda": 0, "true_div_vs_fp4_ref_cpu": 0}
    n = 300 if not quick else 60
    g = torch.Generator(device=dev).manual_seed(5)
    for i in range(n):
        x = (torch.randn(64, 64, device=dev, generator=g) * 10 ** (torch.rand(1, device=dev, generator=g).item() * 12 - 6)).to(torch.bfloat16)
        for rht in (False, True):
            xr = R.rht_apply(x.float(), R.rht_signs(device=dev)) if rht else x.float()
            amax = xr.abs().amax()
            ts_true = (amax / two).reshape(1)
            ts_ref_cuda = R._nv4_ts(xr)
            ts_ref_cpu = R._nv4_ts(xr.cpu())
            kt = fp4q.quant_act(x, "nv4", rht_row=rht, rht_col=rht, amax="exact", div_true=True)
            kd = fp4q.quant_act(x, "nv4", rht_row=rht, rht_col=rht, amax="exact", div_true=False)
            n_ok["true_div_vs_torch_tensor_div"] += int(torch.equal(kt.ts.view(torch.int32), ts_true.view(torch.int32)))
            n_ok["default_vs_fp4_ref_cuda"] += int(torch.equal(kd.ts.view(torch.int32), ts_ref_cuda.view(torch.int32)))
            n_ok["true_div_vs_fp4_ref_cpu"] += int(torch.equal(kt.ts.cpu().view(torch.int32), ts_ref_cpu.view(torch.int32)))
            out["cases"] += 3
    res["ts_random_amax"] = {"trials": 2 * n, **n_ok}
    for k_, v_ in n_ok.items():
        if v_ != 2 * n:
            FAILS.append({"case": f"exact ts {k_}", "bad": {"ok": v_, "of": 2 * n}})
    # b) whole quantization, div_true=True vs fp4_ref on CPU tensors (the literal CONTRACTS 12.2) -- q / scale bytes
    for fmt in ("nv4", "mx4"):
        for kind in ("gauss", "spike3"):
            for rht in (False, True):
                x = data(kind, 1029, 1024, seed=21)
                rrc = R.quantize(x.cpu(), fmt, row=True, col=True, rht_row=rht, rht_col=rht)
                k = fp4q.quant_act(x, fmt, rht_row=rht, rht_col=rht, amax="exact", div_true=True)
                bad = {"q": nbad(k.q.cpu(), rrc.q), "sf": nbad(k.sf.cpu(), rrc.sf), "qt": nbad(k.qt.cpu(), rrc.qt), "sft": nbad(k.sft.cpu(), rrc.sft)}
                res[f"cpu_ref_{fmt}_{kind}_rht{int(rht)}"] = bad
                if any(bad.values()):
                    FAILS.append({"case": f"exact cpu-ref {fmt} {kind} rht={rht}", "bad": bad})
                out["cases"] += 1
    # c) the default mode vs fp4_ref on CUDA, full exact path (no ts given): q, sf, ts all equal
    for fmt in ("nv4", "mx4"):
        for kind in KINDS:
            x = data(kind, 7203, 1024, seed=22)
            rr, rc = ref_copies(x, fmt, True, True, None)
            k = fp4q.quant_act(x, fmt, rht_row=True, rht_col=True, amax="exact")
            bad = {"q": nbad(k.q, rr.q), "sf": nbad(k.sf, rr.sf), "qt": nbad(k.qt, rc.qt), "sft": nbad(k.sft, rc.sft)}
            if fmt == "nv4":
                bad["ts"] = int(k.ts.view(torch.int32) != rr.ts.view(torch.int32))
                bad["tst"] = int(k.tst.view(torch.int32) != rc.tst.view(torch.int32))
            if any(bad.values()):
                FAILS.append({"case": f"exact full {fmt} {kind}", "bad": bad})
            out["cases"] += 1
    # d) prefix: the col amax covers patch tokens only
    x = data("spike3", 7203, 1024, seed=23)
    k = fp4q.quant_act(x, "nv4", rht_col=True, prefix=PREFIX[7203], amax="exact")
    xr = R.rht_apply(patch_of(x, PREFIX[7203]).float().t().contiguous(), R.rht_signs(device=dev))
    ts_p = R._nv4_ts(xr)
    full = R._nv4_ts(R.rht_apply(torch.nn.functional.pad(x.float(), (0, 0, 0, 7296 - 7203)).t().contiguous(), R.rht_signs(device=dev)))
    res["prefix_col_ts_equals_patch_only"] = bool(torch.equal(k.tst.view(torch.int32), ts_p.view(torch.int32)))
    res["prefix_col_ts_differs_from_all_tokens"] = bool(float(ts_p) != float(full))
    if not res["prefix_col_ts_equals_patch_only"]:
        FAILS.append({"case": "exact prefix ts", "bad": {}})
    out["detail"] = res
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("exact")][:20]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_weight(quick):
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    shapes = [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (2304, 768), (768, 768), (3072, 768), (768, 3072), (4096, 4096), (128, 128)]
    if quick:
        shapes = shapes[:3] + shapes[4:5]
    for si, (N, K) in enumerate(shapes):
        for fmt in ("nv4", "mx4"):
            for kind in ("gauss", "spike3", "zeros", "tiny", "big") if si < 4 else ("gauss", "spike3"):
                w = data(kind, N, K, seed=30 + si, dtype=torch.float32) * 0.02
                ref = R.quantize_weight(w, fmt, two_d=True)
                ts = ref.ts if fmt == "nv4" else None
                k = fp4q.quant_weight(w, fmt, amax="ts", ts=ts)
                bad = {"q": nbad(k.q, ref.q), "sf": nbad(k.sf, ref.sf), "qt": nbad(k.qt, ref.qt), "sft": nbad(k.sft, ref.sft)}
                if fmt == "nv4":
                    bad["ts"] = int(k.ts.view(torch.int32) != ref.ts.reshape(1).view(torch.int32))
                    ke = fp4q.quant_weight(w, fmt, amax="exact")
                    bad["ts_exact"] = int(ke.ts.view(torch.int32) != ref.ts.reshape(1).view(torch.int32))
                    bad["q_exact"] = nbad(ke.q, ref.q)
                    bad["qt_exact"] = nbad(ke.qt, ref.qt)
                    # transposed copy of the weight == transposed codes of the row copy (2D identity), checked on the reference side too
                bad = {a: b for a, b in bad.items() if b}
                if bad:
                    FAILS.append({"case": f"weight N={N} K={K} {fmt} {kind}", "bad": bad})
                out["cases"] += 1
                if kind == "gauss":  # bf16 weights as well
                    wb = w.bfloat16()
                    refb = R.quantize_weight(wb.float(), fmt, two_d=True)
                    kb = fp4q.quant_weight(wb, fmt, amax="ts", ts=refb.ts if fmt == "nv4" else None)
                    bad = {"q": nbad(kb.q, refb.q), "qt": nbad(kb.qt, refb.qt), "sf": nbad(kb.sf, refb.sf), "sft": nbad(kb.sft, refb.sft)}
                    bad = {a: b for a, b in bad.items() if b}
                    if bad:
                        FAILS.append({"case": f"weight-bf16 N={N} K={K} {fmt}", "bad": bad})
                    out["cases"] += 1
            del w
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("weight")][:20]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_state(quick):
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    x = data("spike3", 1029, 1024, seed=40)
    xw = data("gauss", 1024, 1024, seed=41, dtype=torch.float32) * 0.02
    st = fp4q.FP4State(3, seed=100)
    # step 0: first call -> exact initialisation of prev
    st.roll()
    k0 = fp4q.quant_act(x, "nv4", rht_col=True, amax="delayed", state=st, site=1, sr_offset=1)
    xr = x.float()
    xc = R.rht_apply(torch.nn.functional.pad(x.float(), (0, 0, 0, 128 * 9 - 1029)).t().contiguous(), R.rht_signs(device=dev))
    a_row, a_col = xr.abs().amax(), xc.abs().amax()
    checks = {}
    checks["init_prev_row"] = bool(torch.equal(st.amax[0, 1, 0], a_row))
    checks["init_prev_col"] = bool(torch.equal(st.amax[0, 1, 1], a_col))
    st.fold()  # round24: the fast kernels publish the amax spread over NPART sectors; roll() / fold() move it to amax[1]
    checks["next_row"] = bool(torch.equal(st.amax[1, 1, 0], a_row))
    checks["next_col"] = bool(torch.equal(st.amax[1, 1, 1], a_col))
    checks["ts_is_2x_prev_row"] = bool(torch.equal(k0.ts.view(torch.int32), (a_row * 2.0 * torch.tensor(1.0 / 2688.0, device=dev)).reshape(1).view(torch.int32)))
    # weights
    kw0 = fp4q.quant_weight(xw, "nv4", amax="delayed", state=st, site=2)
    st.fold()
    checks["w_next"] = bool(torch.equal(st.amax[1, 2, 0], xw.abs().amax()))
    # step 1: roll, then delayed with a different tensor
    st.roll()
    checks["roll_moved_next_to_prev"] = bool(torch.equal(st.amax[0, 1, 0], a_row)) and float(st.amax[1].abs().max()) == 0.0
    checks["seed_step_after_roll"] = [int(st.seed.item()), int(st.step.item())]
    x2 = (x.float() * 0.5).to(torch.bfloat16)
    k1 = fp4q.quant_act(x2, "nv4", rht_col=True, amax="delayed", state=st, site=1, sr_offset=1)
    checks["step1_ts_from_prev"] = bool(torch.equal(k1.ts.view(torch.int32), k0.ts.view(torch.int32)))  # prev amax of site 1 is step 0's amax -> same ts as step 0
    st.fold()
    checks["step1_next_row"] = bool(torch.equal(st.amax[1, 1, 0], x2.float().abs().amax()))
    # bit-exact vs fp4_ref with the delayed ts
    rr, rc = ref_copies(x2, "nv4", False, True, None)
    tsin = torch.cat([k1.ts.reshape(1), k1.tst.reshape(1)])
    k1b = fp4q.quant_act(x2, "nv4", rht_col=True, amax="ts", ts=tsin)
    checks["delayed_equals_ts_mode"] = nbad(k1.q, k1b.q) + nbad(k1.qt, k1b.qt) + nbad(k1.sf, k1b.sf) + nbad(k1.sft, k1b.sft) == 0
    # CUDA graph: roll + act + weight quantizers captured once, replayed; replay == eager on a twin state
    stg = fp4q.FP4State(3, seed=100)
    ste = fp4q.FP4State(3, seed=100)
    for s_ in (stg, ste):
        s_.roll()
        fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="delayed", state=s_, site=1, sr_offset=1)
        fp4q.quant_weight(xw, "nv4", amax="delayed", state=s_, site=2)
    torch.cuda.synchronize()

    def step(s_):
        s_.roll()
        a = fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="delayed", state=s_, site=1, sr_offset=1)
        w = fp4q.quant_weight(xw, "nv4", amax="delayed", state=s_, site=2)
        return a, w

    strm = torch.cuda.Stream()
    strm.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    # NOTE: the capture itself does not execute the kernels; replay count == eager step count
    with torch.cuda.graph(g, stream=strm):
        ga, gw = step(stg)
    n_steps = 4
    for _ in range(n_steps):
        g.replay()
    for _ in range(n_steps):
        ea, ew = step(ste)
    torch.cuda.synchronize()
    checks["graph_replay_equals_eager"] = bool(nbad(ga.q, ea.q) + nbad(ga.qt, ea.qt) + nbad(ga.sf, ea.sf) + nbad(ga.sft, ea.sft) + nbad(gw.q, ew.q) + nbad(gw.qt, ew.qt) == 0)
    checks["graph_state_equals_eager"] = bool(torch.equal(stg.amax, ste.amax) and torch.equal(stg.parts, ste.parts) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step))
    checks["graph_step_counter"] = int(stg.step.item())
    for k_, v_ in checks.items():
        if v_ is False:
            FAILS.append({"case": f"state {k_}", "bad": {}})
    out["checks"] = checks
    out["cases"] = len(checks)
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("state")]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def relL2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def part_gemm(quick):
    import fp4_cands as C
    import fp4_epi

    out = {"cases": [], "fail_list": []}
    t0 = time.time()
    E4 = fp4_epi.nv4_epilogues()
    g = torch.Generator(device=dev).manual_seed(0)
    cfgs = [("fc1-like", 7203, 1024, 4096, PREFIX[7203]), ("qkv-like", 7903, 1024, 3072, PREFIX[7903]), ("head-like", 7168, 1024, 4096, None)]
    if quick:
        cfgs = cfgs[:1]
    for name, M, K, N, prefix in cfgs:
        X = (torch.randn(M, K, device=dev, generator=g) * 1.0).to(torch.bfloat16)
        X[::997] *= 8
        dY = (torch.randn(M, N, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
        W = torch.randn(N, K, device=dev, generator=g) * 0.02
        for fmt in ("nv4",):
            st = fp4q.FP4State(4)
            tw = fp4q.quant_weight(W, fmt, amax="exact")
            tx = fp4q.quant_act(X, fmt, rht_col=True, prefix=prefix, amax="exact")
            td = fp4q.quant_act(dY, fmt, rht_col=True, sr_row=True, sr_col=True, prefix=prefix, amax="exact", seed=st.seed, sr_offset=0)
            Mp = M if prefix is None and M % 128 == 0 else None
            # fwd: Y[M, N] = X Wt
            y = torch.empty(M, N, device=dev, dtype=torch.float32)
            C.quack_fp4(C.operand(tx.row_triple(), fmt), C.operand(tw.row_triple(), fmt), y)
            ref = R.dequant_copy(tx.q, tx.sf, tx.ts, fmt, M, K, apply_ts=True).double() @ R.dequant_copy(tw.q, tw.sf, tw.ts, fmt, N, K, apply_ts=True).double().t()
            r_f = relL2(y, ref)
            # plain_ts epilogue with a device scalar
            y2 = torch.empty(M, N, device=dev, dtype=torch.float32)
            qa, qb = tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t()
            kw = C.epi_kw(tx.sf, tw.sf, fmt)
            E4["plain"](qa, qb, out={"D": y2}, ts=(tx.ts * tw.ts).reshape(1), **kw)
            r_f2 = relL2(y2, ref)
            # dgrad: dX[M, K] = dY W   (A = dY.q [M, N], B = W.qt [K, N])
            dx = torch.empty(M, K, device=dev, dtype=torch.float32)
            C.quack_fp4(C.operand(td.row_triple(), fmt), C.operand(tw.col_triple(), fmt), dx)
            refd = R.dequant_copy(td.q, td.sf, td.ts, fmt, M, N).double() @ R.dequant_copy(tw.qt, tw.sft, tw.tst, fmt, K, N).double().t()
            r_d = relL2(dx, refd)
            # wgrad: dW[N, K] = dYt Xt  (contraction over padded / patch tokens)
            dw = torch.empty(N, K, device=dev, dtype=torch.float32)
            C.quack_fp4(C.operand(td.col_triple(), fmt), C.operand(tx.col_triple(), fmt), dw)
            Kt = tx.kt
            refw = R.dequant_copy(td.qt, td.sft, td.tst, fmt, N, Kt).double() @ R.dequant_copy(tx.qt, tx.sft, tx.tst, fmt, K, Kt).double().t()
            r_w = relL2(dw, refw)
            rec = {"shape": name, "M": M, "K": K, "N": N, "prefix": prefix is not None, "fmt": fmt, "fwd_relL2": r_f, "fwd_plain_ts_epilogue_relL2": r_f2, "dgrad_relL2": r_d, "wgrad_relL2": r_w, "Kt": Kt}
            out["cases"].append(rec)
            for k_ in ("fwd_relL2", "fwd_plain_ts_epilogue_relL2", "dgrad_relL2", "wgrad_relL2"):
                if not rec[k_] <= 2e-3:
                    FAILS.append({"case": f"gemm {name} {fmt} {k_}", "bad": {k_: rec[k_]}})
            # also: quantizer vs the true fp32 product (informational: quantization error of the whole pipeline)
            out["cases"][-1]["fwd_vs_fp32_true_relL2"] = relL2(y, X.float() @ W.t())
            out["cases"][-1]["dgrad_vs_fp32_true_relL2"] = relL2(dx, dY.float() @ W)
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("gemm")]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_div(quick):
    """Markstein division (r = RN(1/d), q = RN(y r), fma residual) used for v = x / (sf * ts) vs IEEE fp32 division: brute force, d with all mantissa patterns, y at the thresholds."""
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    ext = fp4q.extension()
    tot = bad = 0
    for seed in range(1 if quick else 4):
        b, n, *first = ext.div_check(3000 if quick else 20000, 1234 + seed)
        if b and "first_mismatch" not in out:
            out["first_mismatch"] = {"y": hex(first[0]), "d": hex(first[1]), "markstein": hex(first[2]), "ieee": hex(first[3])}
        bad += b
        tot += n
    out["samples"], out["mismatches"] = tot, bad
    out["cases"] = 1
    if bad:
        FAILS.append({"case": "div markstein vs ieee", "bad": {"mismatch": bad, "of": tot}})
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("div")]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_codes(quick):
    """element-level rounding: hardware RN (cvt.rn.satfinite.e2m1x2) vs the fp4_ref thresholds over all 2^32 fp32 patterns; SR code (bit trick) vs fp4_ref.e2m1_sr on random + special inputs."""
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    ext = fp4q.extension()
    out["rn_all_2^32_patterns_mismatches"] = ext.rn_check()
    if out["rn_all_2^32_patterns_mismatches"]:
        FAILS.append({"case": "codes rn exhaustive", "bad": {"mismatch": out["rn_all_2^32_patterns_mismatches"]}})
    n = 2_000_000 if quick else 20_000_000
    g = torch.Generator(device=dev).manual_seed(3)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    special = torch.cat([grid, -grid, grid * (1 + 2 ** -23), grid * (1 - 2 ** -23), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.999999, 6.000001, 7.0, 1e30, 1e-30, 1e-38, 0.9999999, 0.5000001, 0.4999999, -0.0], device=dev)])
    v = torch.cat([(torch.rand(n, device=dev, generator=g) * 14 - 7), torch.exp((torch.rand(n, device=dev, generator=g) * 40 - 30)) * (torch.randint(0, 2, (n,), device=dev, generator=g) * 2 - 1), special.repeat(1024)])
    u = torch.cat([torch.rand(2 * n, device=dev, generator=g), torch.rand(special.numel() * 1024, device=dev, generator=g)])
    u[: 4096] = 0.0
    u[4096: 8192] = 0.99999994
    kern = ext.sr_codes(v.contiguous(), u.contiguous())
    ref = R.e2m1_sr(v, u)
    bad = int((kern != ref).sum())
    out["sr_codes_samples"], out["sr_codes_mismatches"] = int(v.numel()), bad
    if bad:
        FAILS.append({"case": "codes sr vs fp4_ref.e2m1_sr", "bad": {"mismatch": bad}})
    out["cases"] = 2
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("codes")]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_headroom(quick):
    """headroom argument + saturation monitor (FP4State.roll / stats):
    (a) headroom=2.0 (default and explicit) bit-identical to the pre-change build (sha256 table of tests/headroom_baseline.py, 72 outputs);
    (b) delayed ts bits for headroom 8 / 3 / 0.5 (act + weight, both div modes): ts == fl(fl(headroom * prev) * fl(1/2688)) (resp. IEEE division) of the previous amax, q / qt consistent with fp4_ref given that ts;
    (c) monitor counters on a constructed amax sequence (sites with headroom 2 and 8, an equality case, a zero step, a never-called site, a weight site) == an independent host model;
    (d) CUDA graph (roll + act + weight, headroom 8) replay == eager twin incl. monitor counters, with the input changing between replays."""
    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    # (a) bit identity vs the pre-change build
    import hashlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("headroom_baseline", str(ROUND / "tests" / "headroom_baseline.py"))
    hb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hb)
    base = json.loads((ROUND / "profile" / "headroom_baseline_hashes.json").read_text())["hashes"]
    for tag, extra in (("default", {}), ("headroom=2.0", {"headroom": 2.0})):
        r = hb.run(extra)
        bad = [k for k in base if r.get(k) != base[k]]
        out[f"identical_to_prechange_{tag}"] = f"{len(base) - len(bad)}/{len(base)}"
        if bad:
            FAILS.append({"case": f"headroom pre-change identity {tag}", "bad": {"first": bad[:3]}})
        out["cases"] += 1
    # (b) ts bits
    two = torch.tensor(1.0 / 2688.0, dtype=torch.float32, device=dev)
    big = torch.tensor(2688.0, dtype=torch.float32, device=dev)
    xs = {"gauss": data("gauss", 1029, 1024, seed=50), "spike": data("spike3", 7203, 1024, seed=51)}
    for hr in (8.0, 3.0, 0.5, 2.0):
        for div_true in (False, True):
            for kind, x in xs.items():
                st = fp4q.FP4State(2)
                st.roll()
                k0 = fp4q.quant_act(x, "nv4", rht_col=True, amax="delayed", state=st, site=0, headroom=hr, div_true=div_true)  # step 0: exact init of prev
                prev_r, prev_c = st.amax[0, 0, 0].clone(), st.amax[0, 0, 1].clone()
                st.roll()
                x2 = (x.float() * 0.75).bfloat16()
                k1 = fp4q.quant_act(x2, "nv4", rht_col=True, amax="delayed", state=st, site=0, headroom=hr, div_true=div_true)
                for name, prev, got in (("ts", prev_r, k1.ts), ("tst", prev_c, k1.tst)):
                    a = prev * hr  # fp32 multiply, like the kernel
                    want = (a / big) if div_true else (a * two)
                    ok = torch.equal(got.view(torch.int32), want.reshape(1).view(torch.int32))
                    out["cases"] += 1
                    if not ok:
                        FAILS.append({"case": f"headroom ts bits hr={hr} div_true={div_true} {kind} {name}", "bad": {"got": float(got), "want": float(want)}})
                # q / qt of the delayed call == the explicit-ts call with the same ts (and fp4_ref given that ts)
                tsin = torch.cat([k1.ts.reshape(1), k1.tst.reshape(1)])
                k1b = fp4q.quant_act(x2, "nv4", rht_col=True, amax="ts", ts=tsin, div_true=div_true)
                bad = nbad(k1.q, k1b.q) + nbad(k1.qt, k1b.qt) + nbad(k1.sf, k1b.sf) + nbad(k1.sft, k1b.sft)
                out["cases"] += 1
                if bad:
                    FAILS.append({"case": f"headroom q equals ts-mode hr={hr} div_true={div_true} {kind}", "bad": {"bytes": bad}})
                # weight
                w = (torch.randn(2048, 1024, device=dev, generator=torch.Generator(device=dev).manual_seed(7)) * 0.02)
                stw = fp4q.FP4State(1)
                stw.roll()
                fp4q.quant_weight(w, "nv4", amax="delayed", state=stw, site=0, headroom=hr, div_true=div_true)
                pw = stw.amax[0, 0, 0].clone()
                stw.roll()
                kw = fp4q.quant_weight(w * 0.9, "nv4", amax="delayed", state=stw, site=0, headroom=hr, div_true=div_true)
                a = pw * hr
                want = (a / big) if div_true else (a * two)
                out["cases"] += 1
                if not torch.equal(kw.ts.view(torch.int32), want.reshape(1).view(torch.int32)):
                    FAILS.append({"case": f"headroom weight ts bits hr={hr} div_true={div_true}", "bad": {"got": float(kw.ts), "want": float(want)}})
                # q of the delayed weight == fp4_ref 2D weight quantization given the same ts (explicit-ts path of the kernel is gate-A tested against fp4_ref)
                kt = fp4q.quant_weight(w * 0.9, "nv4", amax="ts", ts=kw.ts, div_true=div_true)
                bad = nbad(kw.q, kt.q) + nbad(kw.qt, kt.qt) + nbad(kw.sf, kt.sf) + nbad(kw.sft, kt.sft)
                out["cases"] += 1
                if bad:
                    FAILS.append({"case": f"headroom weight q equals ts-mode hr={hr}", "bad": {"bytes": bad}})
    # headroom recorded per site in the state; default value never touches the device array
    st = fp4q.FP4State(3)
    st.roll()
    x = xs["gauss"]
    fp4q.quant_act(x, "nv4", amax="delayed", state=st, site=0)
    fp4q.quant_act(x, "nv4", amax="delayed", state=st, site=1, headroom=8.0)
    fp4q.quant_weight(torch.randn(256, 256, device=dev), "nv4", amax="delayed", state=st, site=2, headroom=4.0)
    out["headroom_per_site_recorded"] = [float(v) for v in st.headroom.cpu()]
    out["cases"] += 1
    if out["headroom_per_site_recorded"] != [2.0, 8.0, 4.0]:
        FAILS.append({"case": "headroom per-site record", "bad": {"got": out["headroom_per_site_recorded"]}})

    # (c) monitor on a constructed sequence vs an independent host model
    FLT_MAX = 3.402823466e38

    def host_model(amaxes, hr, first_call_init=True):
        """amaxes[t]: amax produced at step t (None = site not called).  Mirrors: roll at the start of every step, call (exact init of prev at the first call), final roll."""
        prev = nxt = 0.0
        calls = exceed = 0
        maxr = 0.0
        inited = False
        for a in list(amaxes) + [None]:  # last entry = the extra roll that flushes the last step
            # roll
            if prev > 0.0:
                calls += 1
                exceed += int(nxt > prev * hr)
                maxr = max(maxr, nxt / prev)
            elif nxt > 0.0:
                calls += 1
                exceed += 1
                maxr = FLT_MAX
            prev, nxt = nxt, 0.0
            if a is not None:  # call
                if not inited:
                    prev = a
                    inited = True
                nxt = max(nxt, a)
        return calls, exceed, maxr

    mult = [1.0, 1.0, 4.0, 1.0, 16.0, 16.0, 0.5, 1.0, 2.0, 0.0, 1.0, 1.0, 64.0, 1.0]  # ratios 1, 4, .25, 16, 1, .03, 2, 2 (equality at hr=2), 0 (zero step), inf (prev == 0), 1, 64, ...
    base_x = data("gauss", 1029, 1024, seed=60)
    xw = (torch.randn(512, 512, device=dev, generator=torch.Generator(device=dev).manual_seed(9)) * 0.02)
    st = fp4q.FP4State(4)  # site0 hr 2, site1 hr 8, site2 weight hr 2, site3 never called
    seqs = {0: [], 1: [], 2: []}
    for m in mult:
        st.roll()
        xm = (base_x.float() * m).bfloat16()  # exact (power of two scaling of bf16)
        fp4q.quant_act(xm, "nv4", rht_col=False, amax="delayed", state=st, site=0, headroom=2.0)
        fp4q.quant_act(xm, "nv4", rht_col=False, amax="delayed", state=st, site=1, headroom=8.0)
        fp4q.quant_weight(xw * m, "nv4", amax="delayed", state=st, site=2, headroom=2.0)
        am = float(xm.float().abs().amax())
        seqs[0].append(am)
        seqs[1].append(am)
        seqs[2].append(float((xw * m).abs().amax()))
    st.roll()  # flush
    sts = st.stats()
    out["monitor_stats"] = {k: sts[k] for k in ("total_calls", "total_exceed", "exceed_rate", "headroom")}
    for site, hr, slots in ((0, 2.0, ("row", "col")), (1, 8.0, ("row", "col")), (2, 2.0, ("row",))):
        exp = host_model(seqs[site], hr)
        for slot in slots:
            got = (sts[slot]["calls"][site], sts[slot]["exceed"][site], sts[slot]["max_ratio"][site])
            gotr = FLT_MAX if got[2] == float("inf") else got[2]
            ok = got[0] == exp[0] and got[1] == exp[1] and abs(gotr - exp[2]) <= 1e-6 * exp[2]
            out["cases"] += 1
            out[f"monitor_site{site}_{slot}"] = {"got": [got[0], got[1], got[2]], "host_model": [exp[0], exp[1], "inf" if exp[2] >= FLT_MAX else exp[2]]}
            if not ok:
                FAILS.append({"case": f"monitor site{site} {slot}", "bad": {"got": list(got), "want": list(exp)}})
    # the never-called site and the unused weight col slot stay at zero
    zero_ok = sts["row"]["calls"][3] == 0 and sts["col"]["calls"][3] == 0 and sts["col"]["calls"][2] == 0 and sts["row"]["exceed"][3] == 0
    out["cases"] += 1
    if not zero_ok:
        FAILS.append({"case": "monitor unused slots", "bad": {}})
    # the headroom-2 site must have counted the equality case (ratio exactly 2) as NOT exceeded and the 4 / 16 / 64 / inf ones as exceeded
    if not (sts["row"]["exceed"][0] > sts["row"]["exceed"][1] > 0):
        FAILS.append({"case": "monitor headroom 2 vs 8 ordering", "bad": {"e2": sts["row"]["exceed"][0], "e8": sts["row"]["exceed"][1]}})
    out["cases"] += 1
    # second sequence without a zero step: finite max ratio
    mult2 = [1.0, 1.0, 4.0, 1.0, 16.0, 16.0, 0.5, 1.0, 2.0, 1.0, 1.0, 64.0, 1.0, 0.25]
    st2 = fp4q.FP4State(2)
    seq2 = []
    for m in mult2:
        st2.roll()
        xm = (base_x.float() * m).bfloat16()
        fp4q.quant_act(xm, "nv4", rht_col=False, amax="delayed", state=st2, site=0, headroom=2.0)
        fp4q.quant_act(xm, "nv4", rht_col=False, amax="delayed", state=st2, site=1, headroom=8.0)
        seq2.append(float(xm.float().abs().amax()))
    st2.roll()
    s2 = st2.stats()
    for site, hr in ((0, 2.0), (1, 8.0)):
        exp = host_model(seq2, hr)
        got = (s2["row"]["calls"][site], s2["row"]["exceed"][site], s2["row"]["max_ratio"][site])
        ok = got[0] == exp[0] and got[1] == exp[1] and got[2] == exp[2] and got[2] == 64.0
        out["cases"] += 1
        out[f"monitor2_site{site}_row"] = {"got": list(got), "host_model": list(exp)}
        if not ok:
            FAILS.append({"case": f"monitor2 site{site}", "bad": {"got": list(got), "want": list(exp)}})
    # reset_stats
    st.reset_stats()
    s2 = st.stats()
    out["cases"] += 1
    if s2["total_calls"] != 0 or s2["total_exceed"] != 0:
        FAILS.append({"case": "monitor reset_stats", "bad": {}})

    # (d) graph replay == eager twin (input changes between replays through a static buffer)
    xin_g = base_x.clone()
    xin_e = base_x.clone()
    stg, ste = fp4q.FP4State(3), fp4q.FP4State(3)
    for s_, xin in ((stg, xin_g), (ste, xin_e)):  # eager warm-up / init of the two sites (outside the graph)
        s_.roll()
        fp4q.quant_act(xin, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="delayed", state=s_, site=0, sr_offset=1, headroom=8.0)
        fp4q.quant_weight(xw, "nv4", amax="delayed", state=s_, site=1, headroom=2.0)
    torch.cuda.synchronize()

    def step_fn(s_, xin):
        s_.roll()
        a = fp4q.quant_act(xin, "nv4", rht_col=True, sr_row=True, sr_col=True, amax="delayed", state=s_, site=0, sr_offset=1, headroom=8.0)
        w = fp4q.quant_weight(xw, "nv4", amax="delayed", state=s_, site=1, headroom=2.0)
        return a, w

    strm = torch.cuda.Stream()
    strm.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=strm):
        ga, gw = step_fn(stg, xin_g)
    # the capture itself runs no kernels; replay count == eager step count
    scales = [1.0, 12.0, 0.5, 1.0, 40.0, 1.0]
    for m in scales:
        xin_g.copy_((base_x.float() * m).bfloat16())
        g.replay()
        xin_e.copy_((base_x.float() * m).bfloat16())
        ea, ew = step_fn(ste, xin_e)
    torch.cuda.synchronize()
    eq = nbad(ga.q, ea.q) + nbad(ga.qt, ea.qt) + nbad(ga.sf, ea.sf) + nbad(ga.sft, ea.sft) + nbad(gw.q, ew.q) + nbad(gw.qt, ew.qt) == 0
    eq_state = torch.equal(stg.amax, ste.amax) and torch.equal(stg.parts, ste.parts) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step)
    eq_mon = torch.equal(stg.mon_cnt, ste.mon_cnt) and torch.equal(stg.mon_maxr, ste.mon_maxr) and torch.equal(stg.headroom, ste.headroom)
    out["graph"] = {"outputs_equal": bool(eq), "state_equal": bool(eq_state), "monitor_equal": bool(eq_mon), "exceed_total_graph": stg.stats()["total_exceed"], "exceed_total_eager": ste.stats()["total_exceed"],
                    "calls_total_graph": stg.stats()["total_calls"]}
    out["cases"] += 3
    for k_, v_ in (("outputs", eq), ("state", eq_state), ("monitor", eq_mon)):
        if not v_:
            FAILS.append({"case": f"headroom graph {k_}", "bad": {}})
    if stg.stats()["total_exceed"] == 0:
        FAILS.append({"case": "headroom graph test did not exercise an exceed", "bad": {}})
    out["fail_list"] = [f for f in FAILS if f["case"].startswith(("headroom", "monitor"))]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out



# ------------------------------------------------------------------------------------------------ round24 additions
def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    va = a.view(torch.int32) if a.dtype == torch.float32 else a.view(torch.uint8)
    vb = b.view(torch.int32) if b.dtype == torch.float32 else b.view(torch.uint8)
    return a.shape == b.shape and torch.equal(va, vb)


def part_fast(quick):
    """fast kernels (actf_kernel) == generic kernel (set_fast(False)) == round21 fp4q, bit for bit: shapes x data kinds x row / col / dual x RN / SR (Philox-10) x delayed (3 steps with roll,
    headroom 8) / exact; the amax state after fold() / roll(); the weight quantizer (row=True identical, row=False gives the identical W^T copy)."""
    from round21_fp4_quant_head.impl import fp4q as A

    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    fails = []
    shapes = [(7203, 1024), (7903, 3072), (1029, 768), (33, 128), (257, 192), (14406, 768), (7168, 4096), (128, 64), (300, 320), (15806, 2304)]
    kinds = ["gauss", "spike8", "spike3", "zeros", "tiny", "big", "negzero", "pow2"]

    def mk(kind, R, C, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(R, C, generator=g, device=dev)
        if kind in ("spike8", "spike3"):
            x[:, ::97] *= 8.0
            x[::53] *= 4.0
            if R > 5 and C > 5:
                x[3, 5] = 200.0
        if kind == "spike3":
            x[:, ::131] *= 1e3
            x[R // 2] *= 1e4
            x[2] *= 1e4
        elif kind == "zeros":
            x.zero_()
        elif kind == "tiny":
            x *= 1e-30
        elif kind == "big":
            x *= 1e4
        elif kind == "negzero":
            x = torch.where(torch.rand(R, C, generator=g, device=dev) < 0.3, torch.tensor(-0.0, device=dev), x)
        elif kind == "pow2":
            x = torch.sign(x) * torch.exp2(torch.round(torch.log2(x.abs().clamp_min(1e-6))))
        return x.to(torch.bfloat16)

    def cmpq(qa, qb, tag):
        for f in ("q", "sf", "qt", "sft", "ts", "tst"):
            if not _same(getattr(qa, f), getattr(qb, f)):
                fails.append(f"{tag}:{f}")

    for (R, C) in shapes:
        for ki, kind in enumerate(kinds):
            if (R > 8000 and ki > 2) or (quick and ki > 1):
                continue
            x = mk(kind, R, C, 1000 + ki)
            for (row, col) in ((True, True), (True, False), (False, True)):
                for sr in (False, True):
                    for mode in ("delayed", "exact"):
                        sa, sb_, sg = A.FP4State(4), fp4q.FP4State(4), fp4q.FP4State(4)
                        for st in (sa, sb_, sg):
                            st.seed.fill_(12345)
                        kw = dict(row=row, col=col, sr_row=sr, sr_col=sr, amax=mode, sr_offset=5)
                        if mode == "delayed":
                            kw.update(site=1)
                        hr = {"headroom": 8.0}
                        for stp in range(3 if mode == "delayed" else 1):
                            xs = x if stp == 0 else (x * (1.0 + 0.5 * stp)).to(torch.bfloat16)
                            qa = A.quant_act(xs, "nv4", state=sa, **kw, **hr)
                            qb = fp4q.quant_act(xs, "nv4", state=sb_, **kw, **hr)
                            fp4q.extension().set_fast(False)
                            qg = fp4q.quant_act(xs, "nv4", state=sg, **kw, **hr)
                            fp4q.extension().set_fast(True)
                            tag = f"[{R},{C}]{kind} row={row} col={col} sr={sr} {mode} step{stp}"
                            cmpq(qa, qb, tag + " r21a-vs-fast")
                            cmpq(qa, qg, tag + " r21a-vs-generic")
                            if mode == "delayed":
                                sb_.fold()
                                sg.fold()
                                if not (torch.equal(sa.amax, sb_.amax) and torch.equal(sa.amax, sg.amax)):
                                    fails.append(tag + " amax-state")
                                sa.roll()
                                sb_.roll()
                                sg.roll()
                                if not (torch.equal(sa.amax, sb_.amax) and torch.equal(sa.seed, sb_.seed)):
                                    fails.append(tag + " amax/seed after roll")
                            out["cases"] += 1
            del x
        torch.cuda.empty_cache()
    for (N_, K_) in ((3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (768, 768), (2304, 768), (256, 384)):
        wt = (torch.randn(N_, K_, device=dev) * 0.02).float()
        sa, sb2, sc2 = A.FP4State(3), fp4q.FP4State(3), fp4q.FP4State(3)
        for mode in ("delayed", "exact"):
            kwq = dict(amax=mode, state=sa, site=2) if mode == "delayed" else dict(amax="exact")
            a_ = A.quant_weight(wt, "nv4", **kwq)
            b_ = fp4q.quant_weight(wt, "nv4", row=True, **(dict(kwq, state=sb2) if mode == "delayed" else kwq))
            c_ = fp4q.quant_weight(wt, "nv4", row=False, **(dict(kwq, state=sc2) if mode == "delayed" else kwq))
            cmpq(a_, b_, f"w[{N_},{K_}] {mode} row=True")
            for f_ in ("qt", "sft", "ts", "tst"):
                if not _same(getattr(a_, f_), getattr(c_, f_)):
                    fails.append(f"w[{N_},{K_}] {mode} row=False:{f_}")
            if c_.q is not None or c_.sf is not None:
                fails.append(f"w[{N_},{K_}] row=False returned q / sf")
            out["cases"] += 1
    out["fail_list"] = [{"case": f, "bad": {}} for f in fails[:40]]
    out["fail"] = len(fails)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_philox7(quick):
    """Philox4x32-7 option: (1) in-kernel SR stream (fast kernel, philox_rounds=r) == explicit u = philox_uniform(rounds=r) through the generic kernel, bit for bit, r = 10 and 7;
    (2) 7 rounds changes only the SR noise (scale bytes, ts and RN unchanged); (3) statistics of the 7-round stream: moments, bit balance, 256-bin and 64x64 pair chi-square, serial /
    cross-word1 / cross-seed correlations, avalanche of counter / key bits (ideal 0.5).  BigCrush itself is not run: Salmon et al. 2011 report Philox4x32-7 as BigCrush-clean."""
    import numpy as np

    out = {"cases": 0, "fail_list": []}
    t0 = time.time()
    seed = torch.tensor([987654321098], dtype=torch.int64, device=dev)
    R_, C_ = 1029, 768
    x = data("gauss", R_, C_, seed=77)
    Kt = -(-R_ // 128) * 128
    for rounds in (10, 7):
        for (row, col) in ((True, True), (True, False), (False, True)):
            st1 = fp4q.FP4State(2)
            st1.roll()
            kfast = fp4q.quant_act(x, "nv4", row=row, col=col, sr_row=True, sr_col=True, amax="delayed", state=st1, site=0, sr_offset=3, seed=seed, philox_rounds=rounds)
            tsv = torch.cat([(kfast.ts if row else kfast.tst).reshape(1), (kfast.tst if col else kfast.ts).reshape(1)])
            u_row = fp4q.philox_uniform(R_ * C_, seed, 6, dev, rounds=rounds).view(R_, C_) if row else None
            u_col = fp4q.philox_uniform(C_ * Kt, seed, 7, dev, rounds=rounds).view(C_, Kt) if col else None
            kexp = fp4q.quant_act(x, "nv4", row=row, col=col, sr_row=row, sr_col=col, amax="ts", ts=tsv, u_row=u_row, u_col=u_col, sr_offset=3)
            for f in ("q", "sf", "qt", "sft"):
                ok = _same(getattr(kfast, f), getattr(kexp, f))
                out["cases"] += 1
                if not ok:
                    FAILS.append({"case": f"philox{rounds} in-kernel vs explicit u row={row} col={col} {f}", "bad": {}})
    a10 = fp4q.quant_act(x, "nv4", row=True, col=False, sr_row=True, amax="exact", seed=seed, sr_offset=3, philox_rounds=10)
    a7 = fp4q.quant_act(x, "nv4", row=True, col=False, sr_row=True, amax="exact", seed=seed, sr_offset=3, philox_rounds=7)
    rn = fp4q.quant_act(x, "nv4", row=True, col=False, amax="exact")
    out["q_bytes_changed_10_vs_7_frac"] = float((a10.q != a7.q).float().mean())
    out["scale_bytes_equal_10_vs_7"] = bool(torch.equal(a10.sf, a7.sf) and torch.equal(a10.ts, a7.ts))
    out["rn_unaffected_by_rounds"] = bool(torch.equal(fp4q.quant_act(x, "nv4", row=True, col=False, amax="exact", philox_rounds=7).q, rn.q))
    out["cases"] += 3
    if not (out["scale_bytes_equal_10_vs_7"] and out["rn_unaffected_by_rounds"] and out["q_bytes_changed_10_vs_7_frac"] > 0.1):
        FAILS.append({"case": "philox7 changes only the SR noise", "bad": {}})
    # statistics of the stream
    n = 1 << 24
    st_ = {}
    for rounds in (10, 7):
        u = fp4q.philox_uniform(n, seed, 11, dev, rounds=rounds)
        h = ((u * 2.0 ** 17 - 1.0) * 0.5).round().long()  # the 16 bit values
        bits = ((h.unsqueeze(1) >> torch.arange(16, device=dev)) & 1).float().mean(0)
        hist = torch.bincount(h >> 8, minlength=256).double()
        exp = n / 256.0
        chi2 = float(((hist - exp) ** 2 / exp).sum())
        uc = u - u.mean()
        ser = [float((uc[:-k] * uc[k:]).mean() / uc.var()) for k in (1, 2, 3, 4, 8, 16)]
        u2 = fp4q.philox_uniform(n, seed, 12, dev, rounds=rounds)
        u3 = fp4q.philox_uniform(n, seed + 1, 11, dev, rounds=rounds)

        def cc(a, b):
            return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

        cell = (h[0::2] >> 10) * 64 + (h[1::2] >> 10)
        hist2 = torch.bincount(cell, minlength=4096).double()
        exp2 = (n / 2) / 4096.0
        chi2_2d = float(((hist2 - exp2) ** 2 / exp2).sum())
        st_[rounds] = {"mean": float(u.mean()), "std": float(u.std()), "bit_balance_min": float(bits.min()), "bit_balance_max": float(bits.max()), "chi2_256bins": chi2, "chi2_256bins_dof": 255,
                       "chi2_pairs_4096cells": chi2_2d, "chi2_pairs_dof": 4095, "serial_corr": ser, "corr_other_word1": cc(u, u2), "corr_other_seed": cc(u, u3)}
        del u, u2, u3, h, cell
    out["stream_stats"] = st_
    s7 = st_[7]
    ok = (abs(s7["mean"] - 0.5) < 3e-4 and abs(s7["std"] - (1 / 12) ** 0.5) < 3e-4 and 0.499 < s7["bit_balance_min"] and s7["bit_balance_max"] < 0.501 and s7["chi2_256bins"] < 255 + 5 * (2 * 255) ** 0.5
          and s7["chi2_pairs_4096cells"] < 4095 + 5 * (2 * 4095) ** 0.5 and max(abs(v) for v in s7["serial_corr"]) < 5e-4 and abs(s7["corr_other_word1"]) < 5e-4 and abs(s7["corr_other_seed"]) < 5e-4)
    out["stream_checks_pass"] = bool(ok)
    out["cases"] += 1
    if not ok:
        FAILS.append({"case": "philox7 stream statistics", "bad": s7})

    def philox_rounds_np(c0, c1, k0, k1, rounds):
        M0, M1, W0, W1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57), np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
        m = np.uint64(0xFFFFFFFF)
        c0, c1, k0, k1 = [np.asarray(v, dtype=np.uint64) for v in (c0, c1, k0, k1)]
        c2 = np.zeros_like(c0)
        c3 = np.zeros_like(c0)
        for _ in range(rounds):
            p0, p1 = M0 * c0, M1 * c2
            c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m, p1 & m, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m
            k0, k1 = (k0 + W0) & m, (k1 + W1) & m
        return np.stack([c0, c1, c2, c3], 1)

    rng = np.random.default_rng(5)
    N = 100000
    base_in = {k: rng.integers(0, 2 ** 32, N, dtype=np.uint64) for k in ("c0", "c1", "k0", "k1")}
    aval = {}
    for rounds in (5, 6, 7, 10):
        base = philox_rounds_np(base_in["c0"], base_in["c1"], base_in["k0"], base_in["k1"], rounds)
        fl = []
        for what in ("c0", "c1", "k0", "k1"):
            for b in (0, 7, 13, 31):
                a_ = dict(base_in)
                a_[what] = a_[what] ^ np.uint64(1 << b)
                alt = philox_rounds_np(a_["c0"], a_["c1"], a_["k0"], a_["k1"], rounds)
                x_ = base ^ alt
                pc = np.zeros(x_.shape, dtype=np.float64)
                for _b in range(32):
                    pc += ((x_ >> np.uint64(_b)) & np.uint64(1)).astype(np.float64)
                fl.append(float(pc.mean() / 32.0))
        aval[rounds] = {"mean_flip_fraction": float(np.mean(fl)), "min": float(np.min(fl)), "max": float(np.max(fl))}
    out["avalanche_flip_fraction"] = aval
    out["cases"] += 1
    if not (abs(aval[7]["mean_flip_fraction"] - 0.5) < 2e-3 and aval[7]["min"] > 0.495 and aval[7]["max"] < 0.505):
        FAILS.append({"case": "philox7 avalanche", "bad": aval[7]})
    # SR unbiasedness with the 7-round stream: mean over seeds of the dequantised code value vs x / (sf ts) for non-saturating elements (normalised by the expected noise)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    xs_ = data("gauss", 256, 1024, seed=9)
    res = {}
    for rounds in (10, 7):
        acc = None
        nseed = 256
        for sd in range(nseed):
            sdt = torch.tensor([1000 + sd], dtype=torch.int64, device=dev)
            k = fp4q.quant_act(xs_, "nv4", row=True, col=False, sr_row=True, amax="exact", seed=sdt, sr_offset=1, philox_rounds=rounds)
            lo, hi = (k.q & 0xF).long(), (k.q >> 4).long()
            deq = grid[torch.stack([lo, hi], 2).reshape(k.q.shape[0], -1)]
            acc = deq if acc is None else acc + deq
            if sd == 0:
                sf0, ts0 = k.sf, k.ts
        res[rounds] = (acc / nseed)
    # compare the two means to each other and to the unrounded target using the RN scale of fp4_ref via dequantised RN as a sanity proxy
    rn_ = fp4q.quant_act(xs_, "nv4", row=True, col=False, amax="exact")
    lo, hi = (rn_.q & 0xF).long(), (rn_.q >> 4).long()
    rn_deq = grid[torch.stack([lo, hi], 2).reshape(rn_.q.shape[0], -1)]
    d10 = float((res[10] - rn_deq).abs().mean())
    d7 = float((res[7] - rn_deq).abs().mean())
    out["sr_mean_vs_rn_mean_abs_diff"] = {"rounds10": d10, "rounds7": d7}
    out["sr_mean_over_all_elements"] = {"rounds10": float(res[10].mean()), "rounds7": float(res[7].mean()), "rn": float(rn_deq.mean())}
    out["cases"] += 1
    if abs(d7 - d10) > 0.02 * max(d10, 1e-6) + 1e-3:
        FAILS.append({"case": "philox7 SR mean differs from philox10", "bad": {"d10": d10, "d7": d7}})
    out["fail_list"] = [f for f in FAILS if f["case"].startswith("philox")]
    out["fail"] = len(out["fail_list"])
    out["seconds"] = round(time.time() - t0, 1)
    return out



PARTS = {"rn": part_rn, "sr": part_sr, "exact": part_exact, "weight": part_weight, "state": part_state, "gemm": part_gemm, "div": part_div, "codes": part_codes, "headroom": part_headroom, "fast": part_fast, "philox7": part_philox7}


def merge_write(part, res):
    data_ = {}
    if OUT.exists():
        try:
            data_ = json.loads(OUT.read_text())
        except Exception:  # noqa: BLE001
            data_ = {}
    data_.setdefault("round", "round24_fp4_quant_opt")
    data_.setdefault("gate", "A (fp4q vs fp4_ref bit-exact)")
    data_.setdefault("parts", {})[part] = res
    data_["all_pass"] = all(v.get("fail", 1) == 0 for v in data_["parts"].values()) and len(data_["parts"]) == len(PARTS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data_, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    names = list(PARTS) if a.part == "all" else a.part.split(",")
    for n in names:
        with gpu_lock():
            FAILS.clear()
            t = time.time()
            res = PARTS[n](a.quick)
            res["fail"] = res.get("fail", len(res.get("fail_list", [])))
            print(f"[{n}] cases={res.get('cases') if not isinstance(res.get('cases'), list) else len(res['cases'])} fail={res['fail']} ({time.time() - t:.0f}s)")
            for f in res.get("fail_list", [])[:8]:
                print("   FAIL", json.dumps(f)[:300])
            if not a.quick:
                merge_write(n, res)
    print("DONE")


if __name__ == "__main__":
    main()
