"""R21b gate A: the REAL-kernel FP4 head (``impl/head_fp4.py``) against references.

  part gemm   the head's nv4 GEMMs (dgrad / wgrad / fwd, both nv4 GEMM flavours) vs the dequantized fp32 product (relative L2 <= 2e-3), real head shapes of both models
  part head   RN variant (+ amax=exact) vs the round20b simulated head (``fp4_head``): logits and ALL parameter / input gradients, cosine >= 0.9999 (gauss and scale-8 spike data,
              C = 1024 and 768); empty policy == r14's head bit for bit; SR variant: statistical consistency (error vs the bf16 head and std over seeds, kernel Philox vs the simulated generator)
  part graph  capture of fwd + bwd (+ roll) of 4 predicts into a CUDA graph == eager (RN, delayed amax: bitwise), the state machine (first call eager, once-per-step weight quantization)

  python tests/test_head_fp4.py --part all [--quick]      -> profile/gateA_head_fp4.json (parts merged)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

from common.bench import assert_gpu_exclusive, gpu_lock  # noqa: E402
from round14_head_sm120.impl import head as h14  # noqa: E402
from round20_fp4_probe.impl import fp4_head as FH20  # noqa: E402
from round20_fp4_probe.impl import fp4_policy as P  # noqa: E402
from round20_fp4_probe.impl import fp4_ref as R  # noqa: E402
from round20_fp4_probe.impl import fp4_sim as S20  # noqa: E402
from round21_fp4_quant_head.impl import fp4q, head_fp4  # noqa: E402

dev = "cuda"
OUT = ROUND / "profile" / "gateA_head_fp4.json"
MODELS = {"vitl16": dict(C=1024, B=7), "vitb16": dict(C=768, B=14)}


def relL2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if float(d) > 0 else 1.0


# ------------------------------------------------------------------------------------------------ part gemm
def part_gemm(quick):
    out = {"cases": [], "fail_list": []}
    g = torch.Generator(device=dev).manual_seed(0)
    shapes = []
    for model, c in MODELS.items():
        C = c["C"]
        M1 = c["B"] * 1024
        shapes += [(f"{model}/convT1", M1, 4 * C, C), (f"{model}/convT2", 4 * M1, 4 * C, C)]
    if quick:
        shapes = shapes[:2]
    for name, M, N, K in shapes:   # Z [M, N] = X [M, K] Wm^T
        for data in ("gauss", "spike"):
            X = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
            dZ = (torch.randn(M, N, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
            if data == "spike":
                X[::997] *= 8
                dZ[:, ::61] *= 8
            W = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
            st = fp4q.FP4State(4)
            tw = fp4q.quant_weight(W, "nv4", amax="exact")
            tx = fp4q.quant_act(X, "nv4", row=True, col=True, amax="exact")
            td = fp4q.quant_act(dZ, "nv4", row=True, col=True, sr_row=True, sr_col=True, amax="exact", seed=st.seed, sr_offset=1)
            deq = lambda t, q, sf, ts, r, c: R.dequant_copy(q, sf, ts, "nv4", r, c, apply_ts=True).double()   # noqa: E731
            ref_f = deq(tx, tx.q, tx.sf, tx.ts, M, K) @ deq(tw, tw.q, tw.sf, tw.ts, N, K).t()
            ref_d = deq(td, td.q, td.sf, td.ts, M, N) @ deq(tw, tw.qt, tw.sft, tw.tst, K, N).t()
            ref_w = deq(td, td.qt, td.sft, td.tst, N, td.kt) @ deq(tx, tx.qt, tx.sft, tx.tst, K, tx.kt).t()
            for impl in ("epi", "operand"):
                yf = head_fp4.nv4_gemm(tx.q, tx.sf, tx.ts, tw.q, tw.sf, tw.ts, torch.empty(M, N, device=dev, dtype=torch.bfloat16), impl)
                dx = head_fp4.nv4_gemm(td.q, td.sf, td.ts, tw.qt, tw.sft, tw.tst, torch.empty(M, K, device=dev, dtype=torch.bfloat16), impl)
                dw = head_fp4.nv4_gemm(td.qt, td.sft, td.tst, tx.qt, tx.sft, tx.tst, torch.empty(N, K, device=dev, dtype=torch.float32), impl)
                rec = {"case": name, "data": data, "impl": impl, "M": M, "N": N, "K": K, "Kt": tx.kt, "fwd_relL2": relL2(yf, ref_f), "dgrad_relL2": relL2(dx, ref_d), "wgrad_relL2": relL2(dw, ref_w)}
                # bf16 outputs have a 2^-9 rounding error by themselves: judge them against that floor; the fp32 wgrad against 2e-3
                rec["pass"] = bool(rec["fwd_relL2"] <= 2e-3 and rec["dgrad_relL2"] <= 2e-3 and rec["wgrad_relL2"] <= 2e-3)
                out["cases"].append(rec)
                if not rec["pass"]:
                    out["fail_list"].append(rec)
                print("gemm", name, data, impl, {k: f"{v:.2e}" for k, v in rec.items() if k.endswith("relL2")}, flush=True)
            del X, dZ, W, tw, tx, td, ref_f, ref_d, ref_w
            torch.cuda.empty_cache()
    out["fail"] = len(out["fail_list"])
    return out


# ------------------------------------------------------------------------------------------------ part head
def make_inputs(C, B, data, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, device=dev, generator=g) * sc   # noqa: E731
    tok = r(B, 1024, C)
    if data == "spike":
        tok[:, ::37] *= 8
        tok[..., ::64] *= 4
    P_ = dict(
        tok=tok.to(torch.bfloat16), e=r(B, 100, C, sc=0.5).to(torch.bfloat16),
        w1=r(C, C, 2, 2, sc=0.03), b1=r(C, sc=0.1), d1=r(C, 1, 3, 3, sc=0.2), g1=1 + r(C, sc=0.1), be1=r(C, sc=0.1),
        w2=r(C, C, 2, 2, sc=0.03), b2=r(C, sc=0.1), d2=r(C, 1, 3, 3, sc=0.2), g2=1 + r(C, sc=0.1), be2=r(C, sc=0.1),
    )
    glog = r(B, 100, 128, 128, sc=1e-2)
    return P_, glog


NAMES = ("tok", "e", "w1", "b1", "d1", "g1", "be1", "w2", "b2", "d2", "g2", "be2")


def run_head(kind, inp, glog, handle=None):
    """-> {'logits', 'grads': {name: tensor}} for kind in r14 | r20 | r21 (fresh leaf copies of the inputs)."""
    leaves = {k: v.detach().clone().requires_grad_(True) for k, v in inp.items()}
    args = [leaves[k] for k in NAMES]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        if kind == "r14":
            logits = h14._Upscale.apply(*args, 1e-6, 1e-6, 32, False)
        elif kind == "r20":
            logits = FH20._Upscale.apply(*args, 1e-6, 1e-6, 32, handle, False)
        else:
            logits = head_fp4._Upscale.apply(*args, 1e-6, 1e-6, 32, handle, False)
    (logits * glog).sum().backward()
    return {"logits": logits.detach().clone(), "grads": {k: leaves[k].grad.detach().clone() for k in NAMES}}


def stub20(text, seed=777):
    pol = P.parse(text)
    return SimpleNamespace(policy=pol, sim=S20.Sim(device=dev, seed=seed, amax=pol.amax), head_call=0)


def stub21(text, **kw):
    ctx = SimpleNamespace(model=SimpleNamespace(num_joint_blocks=3))
    return head_fp4.Handle(ctx, text, **kw)


def compare(a, b):
    row = {"logits_cos": cosine(a["logits"], b["logits"]), "logits_relL2": relL2(a["logits"], b["logits"])}
    cs = {k: cosine(a["grads"][k], b["grads"][k]) for k in NAMES}
    rl = {k: relL2(a["grads"][k], b["grads"][k]) for k in NAMES}
    row["grad_cos_min"] = min(cs.values())
    row["grad_cos_min_name"] = min(cs, key=cs.get)
    row["grad_relL2_max"] = max(rl.values())
    row["grad_cos"] = {k: round(v, 7) for k, v in cs.items()}
    row["grad_relL2"] = {k: float(f"{v:.3e}") for k, v in rl.items()}
    return row


def part_head(quick):
    out = {"rn": [], "bf16_identity": [], "sr": [], "fail_list": []}
    cases = [("vitl16", "gauss"), ("vitl16", "spike"), ("vitb16", "gauss"), ("vitb16", "spike")]
    if quick:
        cases = cases[:1]
    RN = "head.f=mx8,head.dw=nv4+rn,amax=exact"
    for model, data in cases:
        C = MODELS[model]["C"]
        B = 2
        inp, glog = make_inputs(C, B, data)
        # (1) RN variants vs the simulated head
        for tag, text in (("H3rn", RN), ("H1rn", "head.dw=nv4+rn,amax=exact"), ("H2", "head.dw=mx8")):
            r21 = run_head("r21", inp, glog, stub21(text))
            r20 = run_head("r20", inp, glog, stub20(text))
            row = {"model": model, "data": data, "policy": text, "tag": tag, **compare(r21, r20)}
            row["pass"] = bool(row["logits_cos"] >= 0.9999 and row["grad_cos_min"] >= 0.9999)
            out["rn"].append(row)
            if not row["pass"]:
                out["fail_list"].append(row)
            print("head RN", model, data, tag, f"logits cos {row['logits_cos']:.7f} relL2 {row['logits_relL2']:.2e}  grad cos min {row['grad_cos_min']:.7f} ({row['grad_cos_min_name']}) relL2 max {row['grad_relL2_max']:.2e}", flush=True)
        # (2) empty policy == r14 head, bit for bit
        r14 = run_head("r14", inp, glog)
        r21 = run_head("r21", inp, glog, stub21(""))
        same = bool(torch.equal(r14["logits"], r21["logits"]) and all(torch.equal(r14["grads"][k], r21["grads"][k]) for k in NAMES))
        row = {"model": model, "data": data, "bitwise_equal_to_r14": same, **{k: v for k, v in compare(r21, r14).items() if k in ("logits_cos", "grad_cos_min", "grad_relL2_max")}}
        out["bf16_identity"].append(row)
        if not same:
            out["fail_list"].append({"bf16_identity": row})
        print("head bf16 identity", model, data, same, flush=True)
        # (3) SR statistics (gauss only: the data kind does not matter for the noise statistics)
        if data == "gauss":
            n = 3 if quick else 6
            SRT = "head.f=mx8,head.dw=nv4+sr,amax=exact"
            kr, sm = [], []
            h21 = stub21(SRT)
            for s in range(n):
                h21.reset_state(full=True)
                h21.state.seed.fill_(1000 + s)
                kr.append(run_head("r21", inp, glog, h21))
                sm.append(run_head("r20", inp, glog, stub20("head.f=mx8,head.dw=nv4+sr,amax=exact", seed=2000 + s)))
            rows = {}
            for name in ("w1", "w2", "tok"):
                ref = r14["grads"][name].double()
                sc = lambda L: torch.stack([x["grads"][name].double() for x in L])   # noqa: E731
                K_, S_ = sc(kr), sc(sm)
                err = lambda T: float(((T - ref) ** 2).flatten(1).sum(1).sqrt().mean() / ref.norm())   # noqa: E731
                sd = lambda T: float(T.var(0).mean().sqrt() / ref.std())   # noqa: E731
                rows[name] = {"err_vs_bf16_kernel": err(K_), "err_vs_bf16_sim": err(S_), "err_ratio": err(K_) / err(S_), "sd_over_seeds_kernel": sd(K_), "sd_over_seeds_sim": sd(S_), "sd_ratio": sd(K_) / sd(S_),
                              "mean_of_n_err_kernel": float((K_.mean(0) - ref).norm() / ref.norm()), "mean_of_n_err_sim": float((S_.mean(0) - ref).norm() / ref.norm())}
            ok = all(0.9 <= r["err_ratio"] <= 1.1 and 0.9 <= r["sd_ratio"] <= 1.1 for r in rows.values())
            # distinct seeds must give distinct results (the SR noise is live), the same seed the same result
            h21.reset_state(full=True)
            h21.state.seed.fill_(1000)
            again = run_head("r21", inp, glog, h21)
            rep = bool(torch.equal(again["grads"]["w1"], kr[0]["grads"]["w1"]))
            diff = not torch.equal(kr[0]["grads"]["w1"], kr[1]["grads"]["w1"])
            row = {"model": model, "n_seeds": n, "per_grad": rows, "same_seed_bitwise": rep, "other_seed_differs": diff, "pass": bool(ok and rep and diff)}
            out["sr"].append(row)
            if not row["pass"]:
                out["fail_list"].append({"sr": row})
            print("head SR", model, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rows.items()}), "same_seed", rep, "other_seed_differs", diff, flush=True)
        del inp, glog, r14, r21
        torch.cuda.empty_cache()
    out["fail"] = len(out["fail_list"])
    return out


# ------------------------------------------------------------------------------------------------ part graph
def part_graph(quick):
    out = {"cases": [], "fail_list": []}
    for model in ("vitl16", "vitb16") if not quick else ("vitb16",):
        C, B = MODELS[model]["C"], 2
        inp, glog = make_inputs(C, B, "spike")
        text = "head.f=mx8,head.dw=nv4+sr"      # SR is part of the test: seed / offsets are deterministic, so the graph must equal the eager step bit for bit
        h = stub21(text)
        leaves = {k: v.detach().clone().requires_grad_(True) for k, v in inp.items()}

        def one_step():
            for v in leaves.values():
                v.grad = None
            outs = []
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                for _ in range(h.n_calls):
                    outs.append(head_fp4._Upscale.apply(*[leaves[k] for k in NAMES], 1e-6, 1e-6, 32, h, False))
            sum((o * glog).sum() for o in outs).backward()
            h.end_step()      # roll
            return [o.detach() for o in outs]

        # eager reference: first step (init is exact) on a fresh state; then the second step (delayed amax from step 1)
        h.reset_state(full=True)
        e1 = one_step()
        g1 = {k: leaves[k].grad.detach().clone() for k in NAMES}
        e2 = one_step()
        g2 = {k: leaves[k].grad.detach().clone() for k in NAMES}
        # graph: eager warm-up on a side stream (init; == StepGraphRunner.warm: fwd + bwd, no roll), restore the state to "initialised, step 0", capture, replay twice
        h.reset_state(full=True)
        leaves = {k: v.detach().clone().requires_grad_(True) for k, v in inp.items()}      # fresh leaves: their AccumulateGrad nodes belong to the side stream
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                for _ in range(2):
                    outs = [head_fp4._Upscale.apply(*[leaves[k] for k in NAMES], 1e-6, 1e-6, 32, h, False) for _ in range(h.n_calls)]
                    sum((o * glog).sum() for o in outs).backward()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        h.reset_state(full=False)
        for v in leaves.values():
            v.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gouts = []
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                for _ in range(h.n_calls):
                    gouts.append(head_fp4._Upscale.apply(*[leaves[k] for k in NAMES], 1e-6, 1e-6, 32, h, False))
            sum((o * glog).sum() for o in gouts).backward()
            gg = {k: leaves[k].grad for k in NAMES}
            h.end_step()
        h.reset_state(full=False)
        graph.replay()
        torch.cuda.synchronize()
        r1 = {k: gg[k].clone() for k in NAMES}
        o1 = [o.clone() for o in gouts]
        graph.replay()
        torch.cuda.synchronize()
        r2 = {k: gg[k].clone() for k in NAMES}
        o2 = [o.clone() for o in gouts]
        b1 = all(torch.equal(a, b) for a, b in zip(e1, o1)) and all(torch.equal(g1[k], r1[k]) for k in NAMES)
        b2 = all(torch.equal(a, b) for a, b in zip(e2, o2)) and all(torch.equal(g2[k], r2[k]) for k in NAMES)
        row = {"model": model, "step1_bitwise": bool(b1), "step2_bitwise": bool(b2),
               "step1_grad_cos_min": min(cosine(g1[k], r1[k]) for k in NAMES), "step2_grad_cos_min": min(cosine(g2[k], r2[k]) for k in NAMES),
               "step2_grad_relL2_max": max(relL2(r2[k], g2[k]) for k in NAMES), "step_counter": int(h.state.step.item()), "stats": {k: v for k, v in h.state.stats().items() if k in ("total_calls", "total_exceed", "exceed_rate")}}
        out["cases"].append(row)
        if not (row["step1_bitwise"] and row["step2_bitwise"]):
            out["fail_list"].append(row)
        print("graph", row, flush=True)
        del graph
        torch.cuda.empty_cache()
    out["fail"] = len(out["fail_list"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("all", "gemm", "head", "graph"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    parts = ("gemm", "head", "graph") if args.part == "all" else (args.part,)
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    with gpu_lock():
        for _ in range(45):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError:
                time.sleep(2)
        else:
            assert_gpu_exclusive()
        for p in parts:
            t0 = time.time()
            res = {"gemm": part_gemm, "head": part_head, "graph": part_graph}[p](args.quick)
            res["seconds"] = round(time.time() - t0, 1)
            res["quick"] = bool(args.quick)
            rep[p] = res
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(rep, indent=1, default=str))
            print(f"== part {p}: fail {res['fail']} ({res['seconds']}s)", flush=True)
    rep["all_pass"] = bool(all(rep[p]["fail"] == 0 for p in ("gemm", "head", "graph") if p in rep))
    OUT.write_text(json.dumps(rep, indent=1, default=str))
    print("DONE all_pass", rep["all_pass"], flush=True)


if __name__ == "__main__":
    main()
