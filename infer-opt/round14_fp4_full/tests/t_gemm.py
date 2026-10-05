"""Round 14 step 2: op-level check + microbenchmark of the own FP4 GEMM (fp4g_core.cu) against the fp64 product of the dequantised operands, and against QuACK /
cuBLASLt at the same shapes.  python t_gemm.py check | sweep [fmt] [kind] | cmp
Correctness runs unlocked; timing runs under the dual GPU lock (graph of R weight copies, per-call time = graph time / R, like round10)."""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl"))
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import fp4util as U
import fp4gb
from fp4util import R, C

dev = "cuda"
torch.manual_seed(0)
SHAPES = {"vitl16": dict(d=1024, ff=4096, H=16), "vitb16": dict(d=768, ff=3072, H=12)}


def kind_nk(model, kind):
    d, ff = SHAPES[model]["d"], SHAPES[model]["ff"]
    return {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff)}[kind]


def heavy(M, K):
    x = torch.randn(M, K, device=dev)
    x[:, 5] *= 30
    x[:, K // 3] *= 12
    return x


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item(), torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def operands(fmt, M, N, K, nW=1):
    X = heavy(M, K)
    xq = R.quantize_matrix(X, fmt)
    ws = [R.quantize_matrix(torch.randn(N, K, device=dev) * 0.03, fmt) for _ in range(nW)]
    return X, xq, ws


def deq(t, fmt, rows, K):
    return R.dequant_copy(t[0], t[1], t[2], fmt, rows, K)


def tsprod(xq, wq):
    return (xq[2] * wq[2]).reshape(1).contiguous() if xq[2] is not None else None


def check(models=("vitl16",), tag=""):
    for model in models:
        for fmt, fi in (("nv4", 0), ("mx4", 1)):
            for kind in ("qkv", "proj", "fc1", "fc2"):
                for M in (2058, 2258):
                    N, K = kind_nk(model, kind)
                    X, xq, (wq,) = operands(fmt, M, N, K)
                    P = deq(xq, fmt, M, K).double() @ deq(wq, fmt, N, K).double().t()
                    ts = tsprod(xq, wq)
                    # the scale tensors of the quantiser are padded to 128 rows; the kernel reads whole 128-row blocks
                    kern = fp4gb.get(dict(fmt=fi, epi=0, bm=128, bn=128, bkb=64, st=4, wm=2, wn=4))
                    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                    kern(xq[0], wq[0], xq[1], wq[1], M=M, N=N, K=K, ts=ts, C=out)
                    r = rel(out.float(), P)
                    print(f"{model} {fmt} {kind:4s} M={M}: plain relL2={r[0]:.2e} max={r[1]:.2e} cos={r[2]:.7f}", flush=True)


def bench(callables, reps=20):
    for f in callables:
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in callables:
                f()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / reps * 1000 / len(callables))
    return statistics.median(ts)


def sweep(model, fmt, kind, M=2058, cfgs=None, nW=24):
    from lock2 import gpu_lock2
    fi = 0 if fmt == "nv4" else 1
    N, K = kind_nk(model, kind)
    X, xq, ws = operands(fmt, M, N, K, nW=nW)
    ts = [tsprod(xq, w) for w in ws]
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    P = deq(xq, fmt, M, K).double() @ deq(ws[0], fmt, N, K).double().t()
    res = []
    from concurrent.futures import ThreadPoolExecutor

    def _pc(cfg):
        try:
            fp4gb.compile_cubin(dict(cfg, fmt=fi, epi=0))
        except Exception as e:  # noqa: BLE001
            return str(e)[:200]
    with ThreadPoolExecutor(8) as ex:
        for cfg, err in zip(cfgs, ex.map(_pc, cfgs)):
            if err:
                print("compile failed", cfg, err, flush=True)
    with gpu_lock2():
        for cfg in cfgs:
            try:
                kern = fp4gb.get(dict(cfg, fmt=fi, epi=0))
            except Exception as e:  # noqa: BLE001
                print("skip", cfg, str(e)[:160], flush=True)
                continue
            kern(xq[0], ws[0][0], xq[1], ws[0][1], M=M, N=N, K=K, ts=ts[0], C=out)
            torch.cuda.synchronize()
            r = rel(out.float(), P)
            t = bench([(lambda w=w, tt=tt: kern(xq[0], w[0], xq[1], w[1], M=M, N=N, K=K, ts=tt, C=out)) for w, tt in zip(ws, ts)])
            tf = 2 * M * N * K / t / 1e6
            res.append((t, cfg, r[0]))
            print(f"{model} {fmt} {kind} M={M} {cfg}: {t:.1f} us ({tf:.0f} TFLOPS) relL2={r[0]:.1e} occ={kern.occupancy()}", flush=True)
    res.sort(key=lambda x: x[0])
    return res


def default_cfgs():
    out = []
    for bm, bn, wm, wn in ((128, 128, 2, 4), (128, 128, 4, 2), (128, 64, 4, 2), (64, 128, 2, 4), (64, 128, 2, 2), (64, 64, 2, 2), (128, 256, 2, 4), (64, 256, 2, 4)):
        for bkb in (64, 128):
            for st in (3, 4):
                out.append(dict(bm=bm, bn=bn, bkb=bkb, st=st, wm=wm, wn=wn, minb=1))
    return out


def cmp(model="vitl16", fmt="nv4", M=2058, nW=24):
    """QuACK (bias-less plain epilogue with ts) and cuBLASLt at the model shapes, same harness."""
    from lock2 import gpu_lock2
    calib = U.Calib()
    calib.on = False
    for kind in ("qkv", "proj", "fc1", "fc2"):
        N, K = kind_nk(model, kind)
        X, xq, ws = operands(fmt, M, N, K, nW=nW)
        x = U.FQ(xq[0], xq[1], xq[2], M, K)
        lins = []
        for w in ws:
            lin = U.FP4Linear(torch.zeros(N, K, device=dev) + 1e-3, None, fmt, calib)
            lin.q, lin.sf, lin.ts = w[0], w[1], (w[2] if w[2] is not None else torch.ones(1, device=dev))
            lin.qb = U._f4(lin.q).t()
            lin.tsp = (x.ts * lin.ts) if x.ts is not None else torch.ones(1, device=dev)
            lins.append(lin)
        with gpu_lock2():
            tq = bench([(lambda l=l: l.fwd_plain(x)) for l in lins])
            tc = bench([(lambda w=w: C.scaled_mm_fp4(xq, w, fmt, torch.bfloat16)) for w in ws])
        print(f"{model} {fmt} {kind} M={M}: quack_plain {tq:.1f} us  cublaslt {tc:.1f} us", flush=True)


# ------------------------------------------------------------------------------------------------ tile tuning with the real epilogues
TILES = [(128, 128, 2, 4), (128, 128, 4, 2), (128, 64, 4, 2), (64, 128, 2, 4), (64, 128, 4, 2), (64, 64, 2, 2)]


def tune_cfgs():
    out = []
    for bm, bn, wm, wn in TILES:
        for bkb, sts in ((64, (2, 3, 4, 5)), (128, (2,))):
            for st in sts:
                for minb in (1, 2):
                    out.append(dict(bm=bm, bn=bn, bkb=bkb, st=st, wm=wm, wn=wn, minb=minb))
    keep = []
    for c_ in out:
        sb = fp4gb.stage_bytes(dict(fp4gb.DEFAULT, **c_))
        if c_["st"] * sb <= 99 * 1024:
            keep.append(c_)
    return keep


KIND_EPI = {"qkv": 2, "proj": 1, "fc1": 3, "fc2": 1, "ct1": 0, "ct2": 0}


def kind_nk2(model, kind):
    d = SHAPES[model]["d"]
    if kind in ("ct1", "ct2"):
        return 4 * d, d
    return kind_nk(model, kind)


def tune_kind(model, fmt, kind, M, nW_target=192e6):
    from concurrent.futures import ThreadPoolExecutor
    from lock2 import gpu_lock2
    fi = 0 if fmt == "nv4" else 1
    epi = KIND_EPI[kind]
    N, K = kind_nk2(model, kind)
    d, H = SHAPES[model]["d"], SHAPES[model]["H"]
    Ntok = 1029 if M == 2058 else 1129
    nW = int(min(256, max(8, -(-nW_target // (N * K // 2)))))
    X, xq, ws0 = operands(fmt, M, N, K, nW=1)
    w0 = ws0[0]
    ws = [(w0[0].clone(), w0[1].clone()) for _ in range(nW)]
    ts = tsprod(xq, w0)
    bias, gamma, Rs = torch.randn(N, device=dev) * 0.1, torch.rand(N, device=dev), torch.randn(M, N, device=dev)
    sin, cos = torch.randn(1024, 64, device=dev), torch.randn(1024, 64, device=dev)
    vec = 16 if fmt == "nv4" else 32
    Qo = torch.zeros(M, N // 2, dtype=torch.uint8, device=dev)
    SFo = torch.zeros((M + 127) // 128, N // (vec * 4), 32, 4, 4, dtype=torch.uint8, device=dev)
    tso = torch.full((1,), 0.01, device=dev) if fmt == "nv4" else None
    C_out = {0: torch.empty(M, N, device=dev, dtype=torch.bfloat16), 1: torch.empty(M, N, device=dev),
             2: torch.empty(3, M // Ntok, Ntok, H, 64, device=dev, dtype=torch.float16), 3: None}[epi]
    cfgs = tune_cfgs()

    def _pc(cfg):
        try:
            fp4gb.compile_cubin(dict(cfg, fmt=fi, epi=epi))
        except Exception as e:  # noqa: BLE001
            return str(e)[-120:]
    with ThreadPoolExecutor(32) as ex:
        errs = list(ex.map(_pc, cfgs))
    res = []
    with gpu_lock2():
        for cfg, err in zip(cfgs, errs):
            if err:
                continue
            kern = fp4gb.get(dict(cfg, fmt=fi, epi=epi))
            kw = dict(M=M, N=N, K=K, ts=ts, bias=bias if epi in (1, 2, 3) else None, gamma=gamma if epi == 1 else None, R=Rs if epi == 1 else None, C=C_out,
                      sin=sin, cos=cos, Ntok=Ntok, prefix=Ntok - 1024, d=d, H=H, Qo=Qo if epi == 3 else None, SFo=SFo if epi == 3 else None, tso=tso if epi == 3 else None)
            try:
                t = bench([(lambda w=w: kern(xq[0], w[0], xq[1], w[1], **kw)) for w in ws], reps=6)
            except Exception as e:  # noqa: BLE001
                print("run failed", cfg, str(e)[:100], flush=True)
                continue
            res.append((t, cfg))
    res.sort(key=lambda x: x[0])
    return res


def tune(model, fmt, kinds=("qkv", "proj", "fc1", "fc2", "ct1", "ct2"), out="tuned14_new.json"):
    import os
    path = Path(out)
    tuned = json.loads(path.read_text()) if path.exists() else {}
    for kind in kinds:
        M = {"ct1": 2048, "ct2": 8192}.get(kind, 2058)
        res = tune_kind(model, fmt, kind, M)
        N, K = kind_nk2(model, kind)
        top = res[:3]
        print(f"TUNE {model} {fmt} {kind} M={M}: " + " | ".join(f"{t:.1f}us {json.dumps(c_)}" for t, c_ in top), flush=True)
        tuned[f"{model}|{fmt}|{kind}"] = top[0][1]
        tuned[f"{model}|{fmt}|{kind}|us"] = round(top[0][0], 2)
        path.write_text(json.dumps(tuned, indent=1, sort_keys=True))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("vitl16",))
    elif mode == "sweep":
        model, fmt, kind = sys.argv[2], sys.argv[3], sys.argv[4]
        res = sweep(model, fmt, kind, cfgs=default_cfgs())
        print("BEST", model, fmt, kind, json.dumps([(round(t, 1), c) for t, c, _ in res[:4]]), flush=True)
    elif mode == "tune":
        tune(sys.argv[2], sys.argv[3], tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("qkv", "proj", "fc1", "fc2", "ct1", "ct2"), out=sys.argv[5] if len(sys.argv) > 5 else "tuned14_new.json")
    elif mode == "cmp":
        cmp(sys.argv[2], sys.argv[3])
