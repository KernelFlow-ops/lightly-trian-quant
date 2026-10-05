"""Tile sweep of the hand-written CUDA GEMM.  usage: python tune_mg.py <qkv|proj|fc1|fc2> <vitl16|vitb16> <M> [phase]
phase A: split-K = 1 sweep over tile/stages/warp layout/min-blocks;  phase B (proj/fc2): split-K over the top configs."""
import itertools, json, sys, time
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
from multiprocessing import Pool
import mgb

kind, model, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
d, ff, H = {"vitl16": (1024, 4096, 16), "vitb16": (768, 3072, 12)}[model]
Ntok = M // 2
N, K = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff)}[kind]
EPI = {"fc1": 0, "proj": 1, "fc2": 1, "qkv": 2}[kind]


def feasible(c):
    NT = c["wm"] * c["wn"] * 32
    if c["bm"] % c["wm"] or c["bn"] % c["wn"]:
        return False
    wtm, wtn = c["bm"] // c["wm"], c["bn"] // c["wn"]
    if wtm % 16 or wtn % 16:
        return False
    if (c["bm"] * c["bk"] // 8) % NT or (c["bn"] * c["bk"] // 8) % NT:
        return False
    if c["st"] * (c["bm"] + c["bn"]) * c["bk"] * 2 > 98 * 1024:
        return False
    if EPI == 2:
        if c["bn"] % 64 or d % c["bn"]:
            return False
        wph = c["wn"] // (c["bn"] // 64) if c["wn"] >= c["bn"] // 64 else 0
        if wph == 0 or c["wn"] % (c["bn"] // 64) or (wtn // 8 // 2) * wph != 4:
            return False
    if wtm // 16 * (wtn // 8) * 4 > 128:  # accumulators/thread
        return False
    return True


def flags(bm, bn):
    f = "-DORDER=1 -DUNR=2 -DFRAGALL=1"
    if EPI in (0, 2):
        f += " -DSTG=1"
    if N % bn == 0:
        f += " -DNFULL=1"
    return f


def grid_A():
    out = []
    for bm, bn, bk, st, wm, wn, minb in itertools.product((32, 64, 96, 128), (32, 64, 96, 128, 192), (32, 64), (2, 3, 4),
                                                           (1, 2, 4), (1, 2, 4), (3,)):
        c = dict(bm=bm, bn=bn, bk=bk, st=st, wm=wm, wn=wn, minb=minb, epi=EPI, rf32=1, splitk=1, extra=flags(bm, bn))
        if not feasible(c):
            continue
        smem = st * (bm + bn) * bk * 2
        if minb > max(1, 100 * 1024 // (smem + 1024)):
            continue
        out.append(c)
    return out


def comp(c):
    try:
        mgb.compile_cubin(c)
        return None
    except Exception as e:  # noqa
        return str(e)[-300:]


if __name__ == "__main__":
    import torch
    from kt import kt
    from bench import write_results
    phase = sys.argv[4] if len(sys.argv) > 4 else "A"
    dev, bf = "cuda", torch.bfloat16
    r = lambda *s: torch.randn(*s, device=dev).to(bf)
    if kind == "fc1":
        h, W, b = r(M, K), r(N, K) * K ** -0.5, r(N)
        C = torch.empty(M, N, device=dev, dtype=bf)
        mk = lambda k: (lambda: k(h, W, b, C=C))
    elif kind in ("proj", "fc2"):
        h, W, b, g, R = r(M, K), r(N, K) * K ** -0.5, r(N), torch.rand(N, device=dev), torch.randn(M, N, device=dev)
        C = torch.empty(M, N, device=dev)
        mk = lambda k: (lambda: k(h, W, b, g, R, C))
    else:
        h, W, b = r(M, K), r(N, K) * K ** -0.5, r(N)
        ang = torch.randn(1024, 32, device=dev)
        sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()
        C = torch.empty(3, 2, Ntok, H, 64, device=dev, dtype=bf)
        mk = lambda k: (lambda: k(h, W, b, C=C, sin=sin, cos=cos, Ntok=Ntok, prefix=Ntok - 1024, d=d, H=H))
    out = f"/root/lightly-train/infer-opt/round5_gemm_sm120/tunemg2_{kind}_{model}_M{M}.json"
    if phase == "A":
        cfgs = grid_A()
    else:
        prev = json.load(open(out))["rows"]
        base = [x["cfg"] for x in prev[:12]]
        cfgs = []
        for c in base:
            for S in (2, 3, 4, 6, 8):
                cc = dict(c, splitk=S, extra=flags(c["bm"], c["bn"]))
                if K // cc["bk"] >= S * 4:
                    cfgs.append(cc)
        # a few hand-picked wave-friendly tiles
        for bm, bn, wm, wn in ((64, 64, 2, 2), (128, 64, 2, 2), (64, 128, 2, 2), (64, 32, 2, 1), (32, 64, 1, 2)):
            for S in (2, 3, 4):
                cc = dict(bm=bm, bn=bn, bk=32, st=3, wm=wm, wn=wn, minb=3, epi=EPI, rf32=1, splitk=S, extra=flags(bm, bn))
                if feasible(cc):
                    cfgs.append(cc)
    print(len(cfgs), "configs phase", phase, flush=True)
    t0 = time.time()
    with Pool(16) as p:
        errs = p.map(comp, cfgs, chunksize=1)
    print("compile %.0fs, failures %d" % (time.time() - t0, sum(1 for e in errs if e)), flush=True)
    us_t = kt(lambda: torch.nn.functional.linear(h, W, b))["min"]
    print(f"torch linear: {us_t:.1f}us {2*M*N*K/us_t/1e6:.1f}TF", flush=True)
    rows = []
    for c, e in zip(cfgs, errs):
        if e:
            continue
        try:
            k = mgb.get(c)
            fn = mk(k)
            fn(); torch.cuda.synchronize()
            us = kt(fn, n=10, reps=5)["min"]
        except Exception as ex:  # noqa
            print("FAIL", c, str(ex)[:80].replace("\n", " "), flush=True)
            continue
        tf = 2 * M * N * K / us / 1e6
        rows.append({"us": us, "tf": tf, "cfg": c, "occ": k.occupancy()})
        print(f"{us:8.1f}us {tf:6.1f}TF occ={rows[-1]['occ']} {c}", flush=True)
    rows.sort(key=lambda x: x["us"])
    print("TOP10")
    for x in rows[:10]:
        print(f"{x['us']:8.1f}us {x['tf']:6.1f}TF occ={x['occ']} {x['cfg']}")
    if phase == "B":
        old = json.load(open(out))
        old["rows_splitk"] = rows
        json.dump(old, open(out, "w"), indent=1)
    else:
        json.dump({"torch_us": us_t, "rows": rows}, open(out, "w"), indent=1)
