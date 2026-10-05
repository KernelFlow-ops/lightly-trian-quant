"""Systematic TileLang tile sweep (Round 5).  usage: python tune5.py <qkv|proj|fc1|fc2> <vitl16|vitb16> <M> [--quick]
Compiles configs in a process pool (tilelang disk cache), times in-graph with the GPU lock."""
import itertools, json, sys, time, os
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
from multiprocessing import Pool

kind, model, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
quick = "--quick" in sys.argv
d, ff, H = {"vitl16": (1024, 4096, 16), "vitb16": (768, 3072, 12)}[model]
Ntok = M // 2
SHAPE = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff)}[kind]
N, K = SHAPE


def build(cfg):
    from impl import gemm_tl
    c = dict(cfg)
    if kind == "fc1":
        return gemm_tl.gemm_bias_gelu_kernel(M, N, K, **c)
    if kind in ("proj", "fc2"):
        return gemm_tl.gemm_bias_residual_kernel(M, N, K, res_dtype="float32", **c)
    return gemm_tl.qkv_rope_kernel(2, Ntok, d, H, Ntok - 1024, **c)


def warm(cfg):
    try:
        build(cfg)
        return None
    except Exception as e:  # noqa
        return str(e)[:150].replace("\n", " ")


def smem_bytes(c):
    b = c["stages"] * (c["bm"] + c["bn"]) * c["bk"] * 2
    if kind == "qkv":
        b += c["bm"] * c["bn"] * 4
    return b


def grid():
    out = []
    bms = (32, 64, 96, 128, 192, 256) if not quick else (64, 128)
    bns = (32, 64, 96, 128, 192, 256) if not quick else (64, 128)
    for bm, bn, bk, st, mode, swz, pol in itertools.product(
            bms, bns, (32, 64), (2, 3, 4, 5), ("ws", "cp"), (0, 8), ("square",)):
        if kind == "qkv" and (bn % 64 or d % bn):
            continue
        if N % bn and bn > 64 and kind != "qkv":
            pass
        th_opts = (256,) if mode == "ws" else (128, 256)
        for th in th_opts:
            c = dict(bm=bm, bn=bn, bk=bk, stages=st, threads=th, ws=(mode == "ws"), tma=(mode == "ws"), swz=swz, pol=pol)
            if smem_bytes(c) > 98 * 1024 - 1024:
                continue
            if (bm * bn) // th < 16 or (bm * bn) // th > 128:
                continue
            if pol != "square" and swz:  # keep grid moderate
                continue
            if bk == 64 and mode == "ws" and False:
                continue
            out.append(c)
    return out


if __name__ == "__main__":
    import torch
    from kt import kt
    from bench import write_results
    cfgs = grid()
    print(len(cfgs), "configs", flush=True)
    t0 = time.time()
    with Pool(16) as p:
        errs = p.map(warm, cfgs, chunksize=1)
    print("compile done %.0fs" % (time.time() - t0), flush=True)
    dev, bf = "cuda", torch.bfloat16
    r = lambda *s: torch.randn(*s, device=dev).to(bf)
    if kind == "fc1":
        h, W, b = r(M, K), r(N, K) * K ** -0.5, r(N)
        mk = lambda k: (lambda: k(h, W, b))
    elif kind in ("proj", "fc2"):
        h, W, b, res, g = r(M, K), r(N, K) * K ** -0.5, r(N), torch.randn(M, N, device=dev), torch.rand(N, device=dev)
        mk = lambda k: (lambda: k(h, W, b, g, res))
    else:
        h, W, b = r(2, Ntok, d).reshape(M, d), r(N, K) * K ** -0.5, r(N)
        ang = torch.randn(1024, 32, device=dev)
        sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()
        mk = lambda k: (lambda: k(h, W, b, sin, cos))
    # torch reference
    us_t = kt(lambda: torch.nn.functional.linear(h, W, b))["min"]
    print(f"torch linear (bias): {us_t:.1f}us {2*M*N*K/us_t/1e6:.1f}TF", flush=True)
    rows = []
    for c, e in zip(cfgs, errs):
        if e:
            continue
        try:
            k = build(c)
            fn = mk(k)
            fn(); torch.cuda.synchronize()
            us = kt(fn, n=10, reps=5)["min"]
        except Exception as ex:  # noqa
            print("FAIL", c, str(ex)[:80].replace("\n", " "), flush=True)
            continue
        tf = 2 * M * N * K / us / 1e6
        rows.append({"us": us, "tf": tf, "cfg": c})
        print(f"{us:8.1f}us {tf:6.1f}TF {c}", flush=True)
    rows.sort(key=lambda x: x["us"])
    print("TOP10")
    for x in rows[:10]:
        print(f"{x['us']:8.1f}us {x['tf']:6.1f}TF {x['cfg']}")
    write_results(f"/root/lightly-train/infer-opt/round5_gemm_sm120/tune5_{kind}_{model}_M{M}.json",
                  {"torch_us": us_t, "n_compile_fail": sum(1 for e in errs if e), "compile_errors": sorted(set(e for e in errs if e))[:10], "rows": rows})
