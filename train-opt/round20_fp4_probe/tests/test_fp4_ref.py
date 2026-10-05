"""R20a self-test of impl/fp4_ref.py  ->  profile/fp4_ref_selftest.json

 (a) RN quantizer vs QuACK to_nvfp4 / to_mxfp4(floor) and Spark's CUDA quantizer: byte / scale agreement and explained differences
 (b) fp4_ref operands -> QuACK FP4 GEMM, cuBLASLt (F.scaled_mm), Spark fp4gemm, vs fp64 product of the dequantized operands (rel L2)
 (c) SR unbiasedness; (d) RHT orthogonality; (e) default RNG streams untouched; (f) 2D weight W / W^T identity; (g) padding rows
"""
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4_cands as C
import fp4_ref as R
from common.bench import gpu_lock

dev = "cuda"
OUT = {"results": {}}


def rel_l2(o, r):
    o, r = o.double().reshape(-1), r.double().reshape(-1)
    return float((o - r).norm() / r.norm().clamp_min(1e-30))


def data(kind, rows, cols, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device=dev)
    if kind == "spiky8":  # DINOv3-like: a few outlier channels x8 and a few outlier rows
        x[:, ::97] *= 8.0
        x[::53] *= 4.0
        x[3, 5] = 200.0
    elif kind == "tiny_blocks":  # some blocks 1e-5 of the rest -> tiny e4m3 scales
        x[:, 64:128] *= 1e-5
        x[:, 256:272] = 0
    elif kind == "zeros":
        x.zero_()
    return x.to(torch.bfloat16)


def agree(a, b):
    return float((a == b).float().mean())


with gpu_lock():
    torch.manual_seed(0)
    cpu0, cuda0 = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    from quack.blockscaled import quantize as qq
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig

    spark = None
    try:
        import spark_fp4_ext

        spark = spark_fp4_ext.extension()
    except Exception as e:
        OUT["spark_import_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # ---------------------------------------------------------------- (a) layout + RN agreement
    a = OUT["results"]["a_rn_vs_quack_spark"] = {}
    s2d = torch.randint(0, 255, (300, 20), dtype=torch.uint8, device=dev)
    a["blocked_layout_equals_quack_pack"] = bool(torch.equal(R.to_blocked(s2d), pack_scale_2d_to_blocked_contig(s2d).squeeze(0)))
    a["blocked_roundtrip"] = bool(torch.equal(R.from_blocked(R.to_blocked(s2d), 300, 20), s2d))
    for kind in ("gauss", "spiky8", "tiny_blocks"):
        x = data(kind, 2048, 1024)
        rec = {}
        q, sf, ts = R.quantize_matrix(x, "nv4")
        qd, sfq, tsq = qq.to_nvfp4(x, 16, ts.reshape(()))
        sfq_b = pack_scale_2d_to_blocked_contig(sfq.view(torch.uint8)).squeeze(0)
        rec["nv4_vs_quack"] = {"data_bytes_equal": agree(q, qd), "scale_bytes_equal": agree(sf, sfq_b), "ts_equal": bool(torch.equal(ts.reshape(()), tsq.reshape(()))),
                               "n_data_diff": int((q != qd).sum()), "n_scale_diff": int((sf != sfq_b).sum())}
        if spark is not None:
            ts_s = ts.clone()
            qs = torch.empty_like(q)
            sfs = torch.empty(spark.scale_bytes(2048, 1024, 0), dtype=torch.uint8, device=dev)
            spark.fp4quant(x.contiguous(), qs, sfs, ts_s, 0, None, None)
            rec["nv4_vs_spark"] = {"data_bytes_equal": agree(q, qs), "scale_bytes_equal": agree(sf.reshape(-1), sfs), "n_data_diff": int((q != qs).sum()), "n_scale_diff": int((sf.reshape(-1) != sfs).sum())}
            work = torch.zeros(2, dtype=torch.float32, device=dev)
            qs2 = torch.empty_like(q)
            sfs2 = torch.empty_like(sfs)
            spark.fp4quant(x.contiguous(), qs2, sfs2, None, 0, work, work[1:2])
            rec["nv4_spark_device_amax_ts_equal"] = bool(torch.equal(work[1:2], ts))
        q, sf, _ = R.quantize_matrix(x, "mx4", mx_round="floor")
        qd, sfq = qq.to_mxfp4(x, 32)
        rec["mx4_floor_vs_quack"] = {"data_bytes_equal": agree(q, qd), "scale_bytes_equal": agree(sf, pack_scale_2d_to_blocked_contig(sfq.view(torch.uint8)).squeeze(0)), "n_data_diff": int((q != qd).sum())}
        if spark is not None:
            qs = torch.empty_like(q)
            sfs = torch.empty(spark.scale_bytes(2048, 1024, 1), dtype=torch.uint8, device=dev)
            spark.fp4quant(x.contiguous(), qs, sfs, None, 1, None, None)
            rec["mx4_floor_vs_spark"] = {"data_bytes_equal": agree(q, qs), "scale_bytes_equal": agree(sf.reshape(-1), sfs), "n_data_diff": int((q != qs).sum())}
        q, sf, _ = R.quantize_matrix(x, "mx4")  # rceil
        sc = R.from_blocked(sf, 2048, 32)
        amax = x.float().reshape(2048, 32, 32).abs().amax(-1)
        sv = torch.exp2(sc.float() - 127)
        nz = amax > 0
        rec["mx4_rceil_properties"] = {"amax_over_scale_le_6": bool((amax[nz] / sv[nz] <= 6.0).all()), "scale_half_lt_amax_over_6": bool(((sv[nz] / 2) < amax[nz] / 6).all()),
                                       "zero_block_byte": int(sc[~nz].max()) if (~nz).any() else None}
        a[kind] = rec
    xz = data("zeros", 256, 128)
    q, sf, ts = R.quantize_matrix(xz, "nv4")
    qm, sfm, _ = R.quantize_matrix(xz, "mx4")
    a["zero_tensor"] = {"nv4_ts": float(ts), "nv4_scale_bytes_max": int(sf.max()), "nv4_data_max": int(q.max()), "mx4_scale_bytes_max": int(sfm.max()), "mx4_data_max": int(qm.max())}
    print(json.dumps(a, indent=None)[:3000], flush=True)

    # ---------------------------------------------------------------- (b) GEMM consumption
    b = OUT["results"]["b_gemm_vs_dequant_fp64"] = {}
    gen = R.make_gen(dev, 7)
    signs = R.rht_signs(device=dev)
    M, N, K = 7203, 4096, 1024
    X, dY = data("spiky8", M, K, 1), data("gauss", M, N, 2)
    W = (torch.randn(N, K, device=dev, generator=torch.Generator(device=dev).manual_seed(3)) * 0.05).to(torch.bfloat16)
    for fmt in ("nv4", "mx4"):
        tx = R.quantize(X, fmt)
        tw = R.quantize_weight(W, fmt, two_d=True)
        tdy = R.quantize(dY, fmt, sr=True, gen=gen, rht_col=True, signs=signs)
        txr = R.quantize(X, fmt, rht_col=True, signs=signs)
        cases = {
            "fwd": ((tx.q, tx.sf, tx.ts), (tw.q, tw.sf, tw.ts), R.dequant_row(tx, False), R.dequant_row(tw, False), (tx.ts, tw.ts)),
            "dgrad": ((tdy.q, tdy.sf, tdy.ts), (tw.qt, tw.sft, tw.tst), R.dequant_row(tdy, False), R.dequant_col(tw, False), (tdy.ts, tw.tst)),
            "wgrad_rht_sr": ((tdy.qt, tdy.sft, tdy.tst), (txr.qt, txr.sft, txr.tst), R.dequant_col(tdy, False), R.dequant_col(txr, False), (tdy.tst, txr.tst)),
        }
        for kind, (A_, B_, Ad, Bd, tss) in cases.items():
            ref = R.gemm_ref(Ad, Bd)
            if fmt == "nv4":
                ref = ref * (tss[0] * tss[1])
            rec = {"shape": [ref.shape[0], ref.shape[1], Ad.shape[1]]}
            try:
                ao, bo = C.operand(A_, fmt), C.operand(B_, fmt)
                dq = ao.dequantize(torch.float32)
                rec["quack_operand_dequant_equal"] = bool(torch.equal(dq, Ad * (tss[0] if fmt == "nv4" else 1.0)))
                out = torch.empty(ref.shape, device=dev, dtype=torch.float32)
                C.quack_fp4(ao, bo, out)
                rec["quack_relL2"] = rel_l2(out, ref)
            except Exception as e:
                rec["quack_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            try:
                o = C.scaled_mm_fp4(A_, B_, fmt, torch.bfloat16)
                rec["scaled_mm_bf16_relL2"] = rel_l2(o, ref)
            except Exception as e:
                rec["scaled_mm_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if spark is not None:
                try:
                    o = torch.empty(ref.shape, device=dev, dtype=torch.bfloat16)
                    C.spark_fp4(spark, A_, B_, o, fmt)
                    rec["spark_bf16_relL2"] = rel_l2(o, ref)
                except Exception as e:
                    rec["spark_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["bf16_rounding_floor_relL2"] = rel_l2(ref.to(torch.bfloat16), ref)
            b[f"{fmt}_{kind}"] = rec
            print(fmt, kind, {k: v for k, v in rec.items() if k != "shape"}, flush=True)

    # ---------------------------------------------------------------- (c) SR unbiasedness
    c = OUT["results"]["c_sr_unbiased"] = {}
    xs = data("gauss", 64, 1024, 5).float()
    nd = 2000
    for fmt in ("nv4", "mx4"):
        g2 = R.make_gen(dev, 11)
        q0, sf0, ts0 = R.quantize_matrix(xs, fmt)
        d_rn = R.dequant_copy(q0, sf0, ts0, fmt, 64, 1024)
        # saturation mask: |x| / (block scale * ts) > 6 clips (SR cannot be unbiased there)
        sc_only = R.dequant_copy(torch.full_like(q0, 0x22), sf0, ts0, fmt, 64, 1024)  # code 2 = 1.0 -> pure block scale * ts
        sat = (xs.abs() / sc_only) > 6.0
        acc = torch.zeros_like(xs)
        acc2 = torch.zeros_like(xs)
        for _ in range(nd):
            q1, sf1, ts1 = R.quantize_matrix(xs, fmt, sr=True, gen=g2)
            d = R.dequant_copy(q1, sf1, ts1, fmt, 64, 1024)
            acc += d
            acc2 += d * d
        mean = acc / nd
        var = (acc2 / nd - mean * mean).clamp_min(0)
        sd = var.sqrt() / nd ** 0.5
        ok = ~sat
        rms = lambda t: float(t.pow(2).mean().sqrt())
        zz = ((mean - xs) / sd.clamp_min(1e-30))[ok & (sd > 0)]
        c[fmt] = {"n_draws": nd, "frac_saturated(|x|/denom>6)": float(sat.float().mean()),
                  "SR_rms_bias_unsat_over_rms_x": rms((mean - xs)[ok]) / rms(xs), "SR_expected_noise_if_unbiased_over_rms_x": rms(sd[ok]) / rms(xs),
                  "SR_rms_bias_sat_over_rms_x_(clip_bias)": rms((mean - xs)[sat]) / rms(xs) if bool(sat.any()) else 0.0,
                  "RN_rms_err_over_rms_x": rms(d_rn - xs) / rms(xs),
                  "SR_signed_bias_unsat_over_mean_abs": float((mean - xs)[ok].sum() / xs.abs().sum()), "RN_signed_bias_over_mean_abs": float((d_rn - xs).sum() / xs.abs().sum()),
                  "SR_frac_abs_z_gt4_unsat": float((zz.abs() > 4).float().mean())}
        print("SR", fmt, c[fmt], flush=True)

    # ---------------------------------------------------------------- (d) RHT, (f) 2D weights, (g) padding
    d_ = OUT["results"]["d_rht"] = {}
    gg = R.make_gen(dev, 5)
    a_, b_ = torch.randn(256, 1024, device=dev, generator=gg), torch.randn(256, 1024, device=dev, generator=gg)
    ra, rb = R.rht_apply(a_, signs), R.rht_apply(b_, signs)
    d_["dot_preserved_rel"] = float(((ra * rb).sum() - (a_ * b_).sum()).abs() / (a_ * b_).abs().sum())
    d_["norm_preserved_rel"] = float((ra.norm() - a_.norm()).abs() / a_.norm())
    H = torch.tensor([[1.0]], device=dev)
    for _ in range(4):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    d_["fwht16_equals_matrix"] = bool(torch.allclose(R.fwht16(a_.reshape(-1, 16)), a_.reshape(-1, 16) @ H, atol=1e-5))
    d_["signs"] = [int(v) for v in signs.cpu()]
    f_ = OUT["results"]["f_weight_2d"] = {}
    for fmt in ("nv4", "mx4"):
        tw = R.quantize_weight(W, fmt, two_d=True)
        wr, wc = R.dequant_row(tw, False), R.dequant_col(tw, False)
        f_[fmt] = {"W_equals_WT_elementwise": bool(torch.equal(wr, wc.t())), "ts_equal": None if fmt == "mx4" else bool(torch.equal(tw.ts, tw.tst))}
        tw1 = R.quantize_weight(W, fmt, two_d=False)
        f_[fmt]["1D_W_equals_WT"] = bool(torch.equal(R.dequant_row(tw1, False), R.dequant_col(tw1, False).t()))
        f_[fmt]["relL2_W_quant_2D"] = rel_l2(R.dequant_row(tw), W.float())
        f_[fmt]["relL2_W_quant_1D"] = rel_l2(R.dequant_row(tw1), W.float())
    g_ = OUT["results"]["g_padding"] = {}
    for fmt in ("nv4", "mx4"):
        t = R.quantize(data("gauss", 7203, 256, 9), fmt)
        dc = R.dequant_col(t, False)
        g_[fmt] = {"rpad": t.rpad, "pad_cols_all_zero": bool((dc[:, 7203:] == 0).all()), "qt_shape": list(t.qt.shape), "sft_shape": list(t.sft.shape), "sf_shape": list(t.sf.shape)}

    # ---------------------------------------------------------------- (e) default generators untouched by fp4_ref (SR + RHT + weights)
    torch.manual_seed(0)
    cpu0, cuda0 = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    gx = R.make_gen(dev, 99)
    for fmt in ("nv4", "mx4"):
        R.quantize(data("gauss", 300, 256, 4), fmt, sr=True, gen=gx, rht_row=True, rht_col=True)
        R.quantize_weight(W[:256, :256].contiguous(), fmt, two_d=True, rht_row=True, rht_col=True)
        R.quantize_weight(W[:256, :256].contiguous(), fmt, two_d=False)
    OUT["results"]["e_default_rng_untouched"] = {"cpu": bool(torch.equal(cpu0, torch.get_rng_state())), "cuda": bool(torch.equal(cuda0, torch.cuda.get_rng_state()))}
    print(OUT["results"]["e_default_rng_untouched"])

(ROUND / "profile").mkdir(exist_ok=True)
(ROUND / "profile" / "fp4_ref_selftest.json").write_text(json.dumps(OUT, indent=1, default=str))
print("wrote profile/fp4_ref_selftest.json")
