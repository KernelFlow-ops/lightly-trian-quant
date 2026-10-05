"""Round 13 tile sweep for the own MXFP8 GEMM `mgx` (short: M = 2058 / 2258 backbone shapes + head ConvT shapes; ViT-L d=1024 and ViT-B d=768).

  python tune_mx.py --phase compile --n 48            # random valid configs per (model, kind); parallel nvcc -> cubins (CPU only, no lock)
  python tune_mx.py --phase time --top 8 --out impl/tuned_mx.json   # stage 1: time all compiled configs at the first M (graph replay, dual lock); stage 2: top-N at every M, 3 alternating rounds
"""
import argparse
import itertools
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
for p in (HERE / "impl", ROOT / "round10_lowp_probe" / "impl", ROOT / "common"):
    sys.path.insert(0, str(p))

MODELS = {"vitl16": (1024, 16), "vitb16": (768, 12)}
EPI = {"qkv": 2, "proj": 1, "fc1": 0, "fc2": 1, "ct1": 4, "ct2": 4}
MS = {"qkv": (2058, 2258), "proj": (2058, 2258), "fc1": (2058, 2258), "fc2": (2058, 2258), "ct1": (2048,), "ct2": (8192,)}
TMPJ = Path("/root/autodl-tmp/infer-opt-tmp/round13_mxfp8_full/tune_cfgs.json")


def shape(kind, d):
    return {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d), "ct1": (4 * d, d), "ct2": (4 * d, d)}[kind]


def valid(c, kind, d):
    N, K = shape(kind, d)
    import mgx
    if N % c["bn"] or K % c["bkb"] or c["bm"] % c["wm"] or c["bn"] % c["wn"]:
        return False
    wtm, wtn = c["bm"] // c["wm"], c["bn"] // c["wn"]
    if wtm % 16 or wtn % 16 or (wtn // 8) % 2:
        return False
    if c["wm"] * c["wn"] * 32 > 256 or mgx.smem_bytes(c) > 101376:
        return False
    if c["epi"] == 2 and (c["bn"] % 64 or d % c["bn"] or c["wn"] % (c["bn"] // 64) or (wtn // 8 // 2) * (c["wn"] // (c["bn"] // 64)) != 4):
        return False
    if c["epi"] == 0 and (wtn % 32):
        return False
    st_bytes = c["st"] * (c["bm"] + c["bn"]) * c["bkb"]
    if c["epi"] == 2 and c.get("stg") and st_bytes < c["bm"] * (c["bn"] + 8) * 2:
        return False
    if c["epi"] == 0 and st_bytes < c["bm"] * (c["bn"] + 16):
        return False
    return True


def gen(kind, d, n, seed):
    rnd = random.Random(seed)
    space = []
    for bm, bn, bkb, st, (wm, wn), minb, order, unr in itertools.product(
            (64, 80, 96, 112, 128, 160, 192, 256), (64, 128, 192, 256), (64, 128), (2, 3, 4),
            ((2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (1, 8), (8, 1), (2, 1), (1, 2)), (1, 2, 3), (0, 1), (1, 2)):
        for stg in ((0, 1) if EPI[kind] == 2 else (0,)):
            c = dict(epi=EPI[kind], bm=bm, bn=bn, bkb=bkb, st=st, wm=wm, wn=wn, minb=minb, order=order, unr=unr, nfull=1, stg=stg)
            if valid(c, kind, d):
                space.append(c)
    rnd.shuffle(space)
    return space[:n]


def phase_compile(a):
    import mgx
    cfgs = {}
    for mi_, (model, (d, H)) in enumerate(MODELS.items()):
        for ki, kind in enumerate(EPI):
            if model in a.models.split(",") and kind in a.kinds.split(","):
                cfgs[f"{model}|{kind}"] = gen(kind, d, a.n, seed=100 * (mi_ + 1) + ki)
    for model, (d, H) in MODELS.items():                      # always include the defaults
        for kind in EPI:
            if f"{model}|{kind}" not in cfgs:
                continue
            c = dict(mgx._key(mgx.DEFAULT[kind]))
            if valid(c, kind, d) and c not in cfgs[f"{model}|{kind}"]:
                cfgs[f"{model}|{kind}"].append(c)
    TMPJ.write_text(json.dumps(cfgs))
    uniq = {json.dumps(c, sort_keys=True): c for v in cfgs.values() for c in v}
    print(f"{len(uniq)} unique configs to compile", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        errs = list(ex.map(_job, list(uniq.values())))
    bad = [e for e in errs if e]
    print(f"compiled {len(uniq) - len(bad)} ok, {len(bad)} failed in {time.time() - t0:.0f}s", flush=True)
    for e in bad[:3]:
        print(e)


def _job(c):
    import mgx
    try:
        mgx.compile_cubin(c)
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)[-300:]


def operands(kind, d, H, M, dev="cuda"):
    import torch
    import mx13
    N, K = shape(kind, d)
    E = mx13.ext()
    Ntok = 1029 if M == 2058 else 1129
    X = torch.randn(M, K, device=dev) * 0.5
    xb, W = mx13.ActBuf(M, K), mx13.MXW(torch.randn(N, K, device=dev) * 0.03)
    E.quant_rows(X, xb.q, xb.sf)
    bias, g = torch.randn(N, device=dev) * 0.1, torch.rand(N, device=dev)
    R, out = torch.randn(M, N, device=dev), torch.empty(M, N, device=dev)
    ob = mx13.ActBuf(M, N)
    sin, cos = torch.randn(1024, 64, device=dev), torch.randn(1024, 64, device=dev)
    qo = torch.empty(3, max(M // Ntok, 1), Ntok, H, 64, device=dev, dtype=torch.float16)
    ob16 = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    def call(kern):
        if kind == "qkv":
            kern(xb, W, M, N, K, bias=bias, C=qo, sin=sin, cos=cos, Ntok=Ntok, prefix=Ntok - 1024, d=K, H=H)
        elif kind in ("proj", "fc2"):
            kern(xb, W, M, N, K, bias=bias, gamma=g, R=R, C=out)
        elif kind == "fc1":
            kern(xb, W, M, N, K, bias=bias, Cq=ob.q, Csf=ob.sf)
        else:
            kern(xb, W, M, N, K, C=ob16)
    return call


def time_cfg(call, kern, reps=10):
    import torch
    call(kern)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            call(kern)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / reps * 1e3)
    return sorted(ts)[2]


def phase_time(a):
    import torch
    import mgx
    from lock2 import gpu_lock2
    cfgs = json.loads(TMPJ.read_text())
    tuned = json.loads(Path(a.out).read_text()) if Path(a.out).exists() and a.merge else {}
    with gpu_lock2(), torch.inference_mode():
        for model, (d, H) in MODELS.items():
            if model not in a.models.split(","):
                continue
            for kind in EPI:
                if kind not in a.kinds.split(",") or f"{model}|{kind}" not in cfgs:
                    continue
                N, K = shape(kind, d)
                M0 = MS[kind][0]
                call = operands(kind, d, H, M0)
                res = []
                for c in cfgs[f"{model}|{kind}"][a.lo:a.hi]:
                    try:
                        kern = mgx.get(c)
                        us = time_cfg(call, kern)
                    except Exception as e:  # noqa: BLE001
                        print("fail", kind, c, str(e)[:100], flush=True)
                        continue
                    res.append((us, c))
                res.sort(key=lambda r: r[0])
                top = [c for _, c in res[: a.top]]
                print(f"[{model} {kind} M={M0}] stage1 best {res[0][0]:.1f} us ({len(res)} ok); top: " + ", ".join(f"{u:.1f}" for u, _ in res[:5]), flush=True)
                for M in MS[kind]:
                    call2 = operands(kind, d, H, M)
                    acc = {i: [] for i in range(len(top))}
                    kerns = [mgx.get(c) for c in top]
                    for r in range(3):
                        for i in (range(len(top)) if r % 2 == 0 else reversed(range(len(top)))):
                            acc[i].append(time_cfg(call2, kerns[i]))
                    sc = sorted(((sorted(v)[1], i) for i, v in acc.items()))
                    best_us, bi = sc[0]
                    key = f"{kind}|{N}|{K}|{M}"
                    if key in tuned and tuned[key]["us"] <= best_us:     # --merge: keep the faster of old / new
                        print(f"   M={M}: new best {best_us:.1f} us not better than stored {tuned[key]['us']:.1f} us", flush=True)
                        continue
                    tuned[key] = dict(cfg=top[bi], us=best_us, tflops=2 * M * N * K / best_us / 1e6, top3=[dict(us=u, cfg=top[i]) for u, i in sc[:3]])
                    print(f"   M={M}: best {best_us:.1f} us ({2 * M * N * K / best_us / 1e6:.0f} TFLOPS)  {top[bi]}", flush=True)
    Path(a.out).write_text(json.dumps(tuned, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--kinds", default=",".join(EPI))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10 ** 9)
    ap.add_argument("--out", default=str(HERE / "impl" / "tuned_mx.json"))
    a = ap.parse_args()
    {"compile": phase_compile, "time": phase_time}[a.phase](a)
