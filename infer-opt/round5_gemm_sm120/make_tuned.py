"""Final per-shape kernel selection (Round 5).  usage: python make_tuned.py <vitl16|vitb16> [M ...]

For every (kind, M) of the model (qkv / proj / fc1 / fc2) collect the best candidates of the sweeps
    tune5_*.json     TileLang (TMA+warp-specialised and cp.async paths, tile/stages/threads/swizzle sweep)
    tunemg3_*.json   hand-written CUDA kernel (geometry-driven tile sweep, split-K)
plus the Round-3 configuration and torch's F.linear, and time them ALL together with interleaved CUDA-graph replays (clock drift hits every
candidate equally).  The winner goes into impl/tuned_cfg.json (used by ops.py); the whole comparison goes into tuned_summary.json."""
import json
import sys
from pathlib import Path

HERE = Path("/root/lightly-train/infer-opt/round5_gemm_sm120")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import mgb
from kt import ab
from r3load import load_r3

model = sys.argv[1]
Ms = [int(a) for a in sys.argv[2:]] or [2058, 2258]
d, ff, H = {"vitl16": (1024, 4096, 16), "vitb16": (768, 3072, 12)}[model]
PEAK = 232.0
dev, bf = "cuda", torch.bfloat16
gt = None  # gemm_tl loaded lazily
r3 = load_r3()
R3CFG = r3.TILE_CFG


def tl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r5_gemm_tl", str(HERE / "impl" / "gemm_tl.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gt = tl()
SHAPE = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff)}
tuned_path = HERE / "impl" / "tuned_cfg.json"
tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
summary_path = HERE / "tuned_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}


def tl_cfg_from_sweep(c):
    return dict(bm=c["bm"], bn=c["bn"], bk=c["bk"], stages=c["stages"], threads=c["threads"], ws=c["ws"], tma=c["tma"], swz=c["swz"], pol=c["pol"])


def r3_cfg(kind):
    N, K = SHAPE[kind]
    key = {"qkv": f"qkv_{d}", "fc1": f"gelu_{N}_{K}", "proj": f"res_{N}_{K}", "fc2": f"res_{N}_{K}"}[kind]
    return R3CFG[key]


for M in Ms:
    Ntok = M // 2
    prefix = Ntok - 1024
    for kind in ("qkv", "proj", "fc1", "fc2"):
        N, K = SHAPE[kind]
        r = lambda *s: (torch.randn(*s, device=dev) * 0.03).to(bf)  # noqa: E731
        h, W, b = r(M, K), r(N, K), r(N)
        g = torch.rand(N, device=dev)
        R = torch.randn(M, N, device=dev)
        ang = torch.randn(1024, 32, device=dev)
        sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()
        C = torch.empty(M, N, device=dev, dtype=bf if kind == "fc1" else torch.float32)
        Oq = torch.empty(3, 2, Ntok, H, 64, device=dev, dtype=bf)
        cands = {}
        # ---- torch
        cands["torch F.linear (+bias only)"] = ("torch", None, lambda: torch.nn.functional.linear(h, W, b))
        # ---- round 3 config
        c3 = r3_cfg(kind)
        # ---- sweeps
        f5 = HERE / f"tune5_{kind}_{model}_M{M}.json"
        fm = HERE / f"tunemg3_{kind}_{model}_M{M}.json"
        tl_list = [("r3", c3)]
        if f5.exists():
            for x in json.loads(f5.read_text())["rows"][:4]:
                tl_list.append(("tl", tl_cfg_from_sweep(x["cfg"])))
        for tag, c in tl_list:
            try:
                if kind == "fc1":
                    k = gt.gemm_bias_gelu_kernel(M, N, K, **c); fn = (lambda k=k: k(h, W, b))
                elif kind in ("proj", "fc2"):
                    k = gt.gemm_bias_residual_kernel(M, N, K, res_dtype="float32", **c); fn = (lambda k=k: k(h, W, b, g, R))
                else:
                    k = gt.qkv_rope_kernel(2, Ntok, d, H, prefix, **c); fn = (lambda k=k: k(h, W, b, sin, cos))
                cands[f"TL[{tag}] bm{c['bm']} bn{c['bn']} bk{c['bk']} st{c.get('stages')} th{c['threads']} {'ws' if c.get('ws') else 'cp'}"] = ("tl", c, fn)
            except Exception as e:  # noqa
                print("TL candidate failed", tag, c, str(e)[:100])
        if fm.exists():
            dm = json.loads(fm.read_text())
            mg_rows = ([x["cfg"] for x in dm.get("rows", [])[:4]] + [x["cfg"] for x in dm.get("rows_splitk", [])[:4]]
                       + [x["cfg"] for x in dm.get("rows_refined", [])[:5]])
            seen = set()
            for c in mg_rows:
                key = json.dumps(c, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    k = mgb.get(c)
                    if kind == "fc1":
                        fn = (lambda k=k: k(h, W, b, C=C))
                    elif kind in ("proj", "fc2"):
                        fn = (lambda k=k: k(h, W, b, g, R, C))
                    else:
                        fn = (lambda k=k: k(h, W, b, C=Oq, sin=sin, cos=cos, Ntok=Ntok, prefix=prefix, d=d, H=H))
                    cands[f"mg bm{c['bm']} bn{c['bn']} bk{c['bk']} st{c['st']} w{c['wm']}x{c['wn']} S{c['splitk']}"] = ("mg", c, fn)
                except Exception as e:  # noqa
                    print("mg candidate failed", c, str(e)[:100])
        res = ab({k: v[2] for k, v in cands.items()}, n=10, rounds=7, reps=3)
        flops = 2 * M * N * K
        rows = sorted(((v["min"], k) for k, v in res.items()))
        print(f"== {model} {kind} M={M} N={N} K={K}")
        for us, k in rows[:6]:
            print(f"   {us:7.1f} us {flops/us/1e6:6.1f} TF ({100*flops/us/1e6/PEAK:4.1f}% of {PEAK:.0f})  {k}")
        tor = res["torch F.linear (+bias only)"]["min"]
        r3name = next(k for k in res if k.startswith("TL[r3]"))
        # winner among non-torch
        best_us, best_k = next((us, k) for us, k in rows if not k.startswith("torch"))
        impl, cfg, _ = cands[best_k]
        key = f"{kind}|{N}|{K}|{M}"
        tuned[key] = {"impl": impl, "cfg": cfg, "us": best_us}
        summary[f"{model}|{kind}|{M}"] = {"N": N, "K": K, "M": M, "flops": flops, "torch_us": tor, "r3_us": res[r3name]["min"], "r3_cfg": cands[r3name][1],
                                        "best": best_k, "best_us": best_us, "best_impl": impl, "best_cfg": cfg,
                                        "best_tflops": flops / best_us / 1e6, "best_pct_peak": 100 * flops / best_us / 1e6 / PEAK,
                                        "r3_tflops": flops / res[r3name]["min"] / 1e6, "r3_pct_peak": 100 * flops / res[r3name]["min"] / 1e6 / PEAK,
                                        "all": {k: v["min"] for k, v in res.items()}}
        tuned_path.write_text(json.dumps(tuned, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1))
        del cands
        torch.cuda.empty_cache()
