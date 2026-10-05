

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


