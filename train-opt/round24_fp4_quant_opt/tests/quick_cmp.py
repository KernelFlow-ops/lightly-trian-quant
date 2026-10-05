"""quick bit-exact check: fp4q2 (fast kernels) vs fp4q2 generic (set_fast(False)) vs round21 fp4q, many shapes / data kinds / SR / headroom / delayed."""
import sys, time
from pathlib import Path
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
import torch
from common.bench import gpu_lock, assert_gpu_exclusive
from round21_fp4_quant_head.impl import fp4q as A
sys.path.insert(0, str(ROUND / "impl"))
import fp4q2 as B

dev = "cuda"
def data(kind, R, C, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(R, C, generator=g, device=dev)
    if kind in ("spike8", "spike3"):
        x[:, ::97] *= 8.0; x[::53] *= 4.0
        if R > 5 and C > 5: x[3, 5] = 200.0
    if kind == "spike3":
        x[:, ::131] *= 1e3; x[R // 2] *= 1e4; x[2] *= 1e4
    elif kind == "zeros": x.zero_()
    elif kind == "tiny": x *= 1e-30
    elif kind == "big": x *= 1e4
    elif kind == "negzero":
        x = torch.where(torch.rand(R, C, generator=g, device=dev) < 0.3, torch.tensor(-0.0, device=dev), x)
    elif kind == "pow2":
        x = torch.sign(x) * torch.exp2(torch.round(torch.log2(x.abs().clamp_min(1e-6))))
    return x.to(torch.bfloat16)

def same(a, b):
    if a is None or b is None: return a is None and b is None
    return a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a.view(torch.int32), b.view(torch.uint8) if b.dtype != torch.float32 else b.view(torch.int32))

def cmp(qa, qb, tag, fails):
    for f in ("q", "sf", "qt", "sft", "ts", "tst"):
        if not same(getattr(qa, f), getattr(qb, f)):
            fails.append(f"{tag}:{f}")

def main():
    fails, n = [], 0
    with gpu_lock():
        for _ in range(60):
            try: assert_gpu_exclusive(); break
            except RuntimeError as e: print("retry", str(e)[:80], flush=True); time.sleep(20)
        shapes = [(7203, 1024), (7903, 3072), (1029, 768), (33, 128), (257, 192), (14406, 768), (7168, 4096), (128, 64), (300, 320)]
        kinds = ["gauss", "spike8", "spike3", "zeros", "tiny", "big", "negzero", "pow2"]
        for (R, C) in shapes:
            for ki, kind in enumerate(kinds):
                if R > 8000 and ki > 2: continue
                x = data(kind, R, C, 1000 + ki)
                for (row, col) in ((True, True), (True, False), (False, True)):
                    for sr in (False, True):
                        for mode in ("delayed", "exact"):
                            sa, sb_, sg = A.FP4State(4), B.FP4State(4), B.FP4State(4)
                            for st in (sa, sb_, sg): st.seed.fill_(12345)
                            kw = dict(row=row, col=col, sr_row=sr, sr_col=sr, amax=mode, sr_offset=5)
                            if mode == "delayed": kw.update(site=1)
                            hr = {"headroom": 8.0}
                            steps = 3 if mode == "delayed" else 1
                            for stp in range(steps):
                                xs = x if stp == 0 else (x * (1.0 + 0.5 * stp)).to(torch.bfloat16)
                                qa = A.quant_act(xs, "nv4", state=sa, **kw, **hr)
                                qb = B.quant_act(xs, "nv4", state=sb_, **kw, **hr)
                                B.extension().set_fast(False)
                                qg = B.quant_act(xs, "nv4", state=sg, **kw, **hr)
                                B.extension().set_fast(True)
                                tag = f"[{R},{C}]{kind} row={row} col={col} sr={sr} {mode} step{stp}"
                                cmp(qa, qb, tag + " r21a-vs-fast", fails); cmp(qa, qg, tag + " r21a-vs-generic", fails)
                                if mode == "delayed":
                                    sb_.fold(); sg.fold()
                                    if not torch.equal(sa.amax, sb_.amax): fails.append(tag + " amax-state-fast")
                                    if not torch.equal(sa.amax, sg.amax): fails.append(tag + " amax-state-generic")
                                    sa.roll(); sb_.roll(); sg.roll()
                                    if not torch.equal(sa.amax, sb_.amax): fails.append(tag + " amax-after-roll")
                                    if not torch.equal(sa.seed, sb_.seed): fails.append(tag + " seed")
                            n += 1
                        # roll: second step with delayed amax from the first
                if len(fails) > 40: break
        # weights: row=True identical to round21, row=False gives the identical W^T copy and no row copy
        for (N_, K_) in ((3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (768, 768), (2304, 768), (256, 384)):
            wt = (torch.randn(N_, K_, device=dev) * 0.02).float()
            sa, sb2, sc2 = A.FP4State(3), B.FP4State(3), B.FP4State(3)
            for mode in ("delayed", "exact"):
                kwq = dict(amax=mode, state=sa, site=2) if mode == "delayed" else dict(amax="exact")
                a_ = A.quant_weight(wt, "nv4", **kwq)
                kwb = dict(kwq, state=sb2) if mode == "delayed" else kwq
                b_ = B.quant_weight(wt, "nv4", row=True, **kwb)
                kwc = dict(kwq, state=sc2) if mode == "delayed" else kwq
                c_ = B.quant_weight(wt, "nv4", row=False, **kwc)
                cmp(a_, b_, f"w[{N_},{K_}] {mode} row=True", fails)
                for f_ in ("qt", "sft", "ts", "tst"):
                    if not same(getattr(a_, f_), getattr(c_, f_)): fails.append(f"w[{N_},{K_}] {mode} row=False:{f_}")
                if c_.q is not None or c_.sf is not None: fails.append(f"w[{N_},{K_}] row=False returned q")
                n += 1
        print(f"cases {n} fails {len(fails)}", flush=True)
        for f in fails[:40]: print("FAIL", f)
        # weights
    print("DONE")
main()
