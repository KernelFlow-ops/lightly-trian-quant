"""Tile sweep for the mg_i8 prototype (copy of tune16.py) (CPU compile phase needs no GPU; time phase must run under the dual lock).

  python tune16.py --phase gen      # write the random config list per kind (seeded) to tmp/tune8_cfgs.json
  python tune16.py --phase compile  # parallel nvcc -> cubins (no GPU)
  python tune16.py --phase stage1 --Ms 2058     # time every compiled config once (graph replay) per (model, kind), keep the best 12
  python tune16.py --phase stage2 --Ms 2058,2258  # re-time best-of-stage1 + r5_v1 + cublas fp16-acc, 3 interleaved rounds -> impl/tuned8.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/tmp")

EPI = {"qkv": 2, "proj": 1, "fc1": 0, "fc2": 1}
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"


def gen(kind: str, n: int, seed: int, wide: bool = False) -> list[dict]:
    rnd = random.Random(seed)
    space = []
    BMS = (64, 80, 96, 112, 128, 160, 192, 256) if wide else (64, 80, 96, 112, 128, 160)
    BNS = (64, 96, 128, 192, 256) if wide else (64, 96, 128)
    STS = (2, 3, 4) if wide else (2, 3)
    WGS = ((2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (1, 8), (8, 1)) if wide else ((2, 2), (1, 4), (4, 1), (2, 4))
    for bm, bn, bk, st, (wm, wn), minb in itertools.product(BMS, BNS, (32, 64), STS, WGS, (1, 2, 3)):
        if bm % wm or bn % wn or (bm // wm) % 16 or (bn // wn) % 16 or ((bn // wn) // 8) % 2:
            continue
        if wm * wn * 32 > 256 or st * (bm + bn) * bk * 2 > 96 * 1024:
            continue
        if EPI[kind] == 2 and bn % 64:
            continue
        if kind in ("qkv",) and bn == 96:
            continue
        space.append((bm, bn, bk, st, wm, wn, minb))
    rnd.shuffle(space)
    out = []
    for bm, bn, bk, st, wm, wn, minb in space[:n]:
        for splitk in (1,):
            for stg in ((0, 1) if EPI[kind] != 1 else (0,)):
                ex = EX if not wide else f"-DORDER={rnd.choice((0, 1, 1))} -DUNR={rnd.choice((1, 2, 2, 4))} -DFRAGALL={rnd.choice((0, 1, 1))}"
                c = dict(epi=EPI[kind], bm=bm, bn=bn, bk=bk, st=st, wm=wm, wn=wn, minb=minb, splitk=splitk, rf32=1,
                         extra=ex + (" -DSTG=1" if stg else ""))
                out.append(c)
    return out


def with_nfull(cfg, N):
    c = dict(cfg)
    if N % c["bn"] == 0 and "-DNFULL=1" not in c["extra"]:
        c["extra"] = c["extra"] + " -DNFULL=1"
    return c


def _compile(cfg):
    import mg8b
    try:
        mg8b.compile_cubin(dict(cfg))
        return None
    except Exception as ex:  # noqa: BLE001
        return str(ex)[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--Ms", default="2058")
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--kinds", default="qkv,proj,fc1,fc2")
    ap.add_argument("--pkt1", action="store_true", help="stage2 restricted to configs with pkt == 1 -> impl/tuned8_pkt1.json")
    a = ap.parse_args()
    import bench10 as b
    cfgf = TMP / "tune8_cfgs.json"
    if a.phase == "gen2":      # extend the pool with a wider random batch (more tiles / stages / loop orders), keep the old configs
        cfgs = json.loads(cfgf.read_text())
        for i, k in enumerate(EPI):
            have = {json.dumps(c, sort_keys=True) for c in cfgs[k]}
            cfgs[k] += [c for c in gen(k, a.n, 777 + i, wide=True) if json.dumps(c, sort_keys=True) not in have]
        cfgf.write_text(json.dumps(cfgs))
        print({k: len(v) for k, v in cfgs.items()})
        return
    if a.phase == "gen":
        cfgs = {k: gen(k, a.n, 1234 + i) for i, k in enumerate(EPI)}
        # R5's tuned MG configs and defaults are always in the pool
        r5 = json.loads(Path("/root/lightly-train/infer-opt/round5_gemm_sm120/impl/tuned_cfg.json").read_text())
        for key, ent in r5.items():
            if ent["impl"] == "mg":
                kind = key.split("|")[0]
                c = {k: v for k, v in ent["cfg"].items() if k != "extra"}
                c["extra"] = ent["cfg"].get("extra", EX).replace("-DNFULL=1", "")
                cfgs[kind].append(c)
        cfgf.write_text(json.dumps(cfgs))
        print({k: len(v) for k, v in cfgs.items()})
        return
    cfgs = json.loads(cfgf.read_text())
    if a.phase == "compile":
        jobs = []
        for model in a.models.split(","):
            for kind in a.kinds.split(","):
                N, K = b.kind_nk(model, kind)
                for c in cfgs[kind]:
                    jobs.append(with_nfull(c, N))
        uniq = {json.dumps(j, sort_keys=True): j for j in jobs}
        print("compiling", len(uniq), "configs", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=48) as ex:
            res = list(ex.map(_compile, uniq.values()))
        bad = [(c, r) for c, r in zip(uniq.values(), res) if r]
        print(f"done in {time.time() - t0:.0f}s, failed {len(bad)}", flush=True)
        for c, r in bad[:5]:
            print(c, r)
        (TMP / "tune8_bad.json").write_text(json.dumps([c for c, _ in bad]))
        return
    # ---- timing phases
    import cands_base as cb
    import cands_i8 as c16
    bad = {json.dumps(c, sort_keys=True) for c in json.loads((TMP / "tune8_bad.json").read_text())}
    Ms = [int(x) for x in a.Ms.split(",")]
    out_path = TMP / "tune8_stage1.json"
    s1 = json.loads(out_path.read_text()) if out_path.exists() else {}
    tuned_path = HERE / "impl" / ("tuned8_pkt1.json" if a.pkt1 else "tuned8.json")
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    print("other gpu procs:", b.other_gpu_procs(), flush=True)
    for model in a.models.split(","):
        for kind in a.kinds.split(","):
            N, K = b.kind_nk(model, kind)
            for M in Ms:
                shape = dict(model=model, kind=kind, M=M, N=N, K=K)
                key = f"{kind}|{N}|{K}|{M}"
                if a.phase == "stage1":
                    pool = [with_nfull(c, N) for c in cfgs[kind]]
                    pool = [c for c in pool if json.dumps(c, sort_keys=True) not in bad]
                else:
                    # best of stage 1 for the nearest M that has one, plus the already-chosen tuned cfg
                    src = [k for k in s1 if k.startswith(f"{kind}|{N}|{K}|") and (k.endswith("|pkt1") == a.pkt1)]
                    pool = []
                    for k in src:
                        pool += [e["cfg"] for e in s1[k][:8]]
                    uniq = {json.dumps(c, sort_keys=True): c for c in pool}
                    pool = list(uniq.values())
                t0 = time.time()
                cands = [b.Cand(f"c{i}", c16.make_mg8(c)) for i, c in enumerate(pool)]
                if a.phase == "stage2":
                    cands += cb.base_cands(["r5_v1", "cublas_fp16acc", "cublas_bf16"])
                # time in chunks so that the graphs of all candidates need not be resident at once
                res = {}
                step = 16 if a.phase == "stage1" else 64
                for i in range(0, len(cands), step):
                    chunk = cands[i:i + step]
                    if a.phase == "stage2" and i:
                        chunk = chunk + cb.base_cands(["r5_v1"])
                    res.update(b.time_shape(shape, chunk, rounds=1 if a.phase == "stage1" else 3, target_ms=12 if a.phase == "stage1" else 30, verbose=False))
                rows = []
                for i, c in enumerate(pool):
                    r = res.get(f"c{i}", {})
                    if "us" in r:
                        rows.append({"us": r["us"], "tflops": r["tflops"], "cfg": c})
                rows.sort(key=lambda e: e["us"])
                if a.phase == "stage1":
                    s1[key] = rows[:12]
                    s1[key + "|pkt1"] = [e for e in rows if e["cfg"].get("pkt", 1) == 1][:12]
                    out_path.write_text(json.dumps(s1))
                    print(f"{model} {kind} M={M}: {len(rows)}/{len(pool)} ok, best {rows[0]['us']:.1f}us ({rows[0]['tflops']:.0f}TF) {rows[0]['cfg']}  [{time.time() - t0:.0f}s]", flush=True)
                else:
                    base = {n: res[n]["us"] for n in ("r5_v1", "cublas_fp16acc", "cublas_bf16") if n in res and "us" in res[n]}
                    tuned[key] = {"cfg": rows[0]["cfg"], "us": rows[0]["us"], "tflops": rows[0]["tflops"], "baseline_us": base,
                                  "top3": [{"us": e["us"], "cfg": e["cfg"]} for e in rows[:3]]}
                    tuned_path.write_text(json.dumps(tuned, indent=1))
                    print(f"{model} {kind} M={M}: mg16 best {rows[0]['us']:.1f}us ({rows[0]['tflops']:.0f}TF) vs {base}  cfg={rows[0]['cfg']}  [{time.time() - t0:.0f}s]", flush=True)
                import torch
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
