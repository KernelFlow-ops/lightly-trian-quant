"""R20a GEMM candidate matrix on the REAL training shapes (CUDA-graph replay, candidates interleaved in one process, median of 3 rounds, dual GPU lock).

  python tests/bench_fp4_gemm.py --model vitl16 --what bb:7203   -> profile/fp4_gemm_matrix_vitl16_bb7203.json   (backbone, one M, 4 linears x f/d/w)
  python tests/bench_fp4_gemm.py --model vitl16 --what head      -> ConvT1/ConvT2 f/d/w

All GEMMs are TN (A [M,K] . B[N,K]^T).  Current FINAL implementation = r17 MXFP8 native layouts (+ r13 epilogues), qkv fwd bf16, head bf16 QuACK / cuBLAS.
FP4 candidates: QuACK nv4 / mx4 (plain, and with the round13 epilogues; nv4 uses the ts-scalar epilogues of impl/fp4_epi.py), cuBLASLt (F.scaled_mm, NVFP4 only),
Spark fp4gemm (bf16 out only, K % 256, TMA variant needs M,N % 128).  Per-step weighting is done in tests/summarize_r20a.py.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4_cands as C
import fp4_epi
import fp4_ref as R
from common.bench import gpu_lock

dev = "cuda"
CFG = {"vitl16": dict(d=1024, Mp=(7203, 7903), head=(7168, 28672)), "vitb16": dict(d=768, Mp=(14406, 15806), head=(14336, 57344))}


def pad(n, m):
    return -(-n // m) * m


# ------------------------------------------------------------------------------------------------ timing
def make_graph(fn, inner=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    return g


def interleave(cands, rounds=3, reps=10, inner=5):
    graphs, errors = {}, {}
    for n, f in cands.items():
        try:
            graphs[n] = make_graph(f, inner)
        except Exception as e:  # noqa: BLE001
            errors[n] = f"{type(e).__name__}: {str(e)[:240]}"
            torch.cuda.synchronize()
    times = {n: [] for n in graphs}
    order = list(graphs)
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for r in range(rounds):
        for n in (order if r % 2 == 0 else order[::-1]):
            g = graphs[n]
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            ev[0].record()
            for _ in range(reps):
                g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            times[n].append(ev[0].elapsed_time(ev[1]) * 1000.0 / (reps * inner))
    return {n: statistics.median(t) for n, t in times.items()}, errors


# ------------------------------------------------------------------------------------------------ operand prep
def pad_rows(q, rows):
    return torch.nn.functional.pad(q, (0, 0, 0, rows - q.shape[0])).contiguous()


def pad_k(q, sf, kp, fmt):
    """pad the contraction axis (token axis of the transposed copies) with zero data / zero scales to kp elements."""
    vec = R.VEC[fmt]
    q2 = torch.nn.functional.pad(q, (0, kp // 2 - q.shape[1])).contiguous()
    rk = -(-kp // (4 * vec))
    sf2 = torch.nn.functional.pad(sf, (0, 0, 0, 0, 0, 0, 0, rk - sf.shape[1])).contiguous() if rk > sf.shape[1] else sf
    return q2, sf2


class Spark:
    def __init__(self):
        import spark_fp4_ext

        self.ext = spark_fp4_ext.extension()

    def variant(self, M, N, K):
        if M % 128 == 0 and N % 128 == 0 and (M // 128) * (N // 128) >= 170 and self.ext.supports(M, N, K, 2):
            return 2
        return 1

    def fn(self, a, b, fmt, M, N, K, out):
        v = self.variant(M, N, K)
        return (lambda: C.spark_fp4(self.ext, a, b, out, fmt, v)), v


def fp4_prep(X, dY, W, Mp):
    """quantize with fp4_ref (RN; timing is data independent).  Returns {fmt: (tx, tdy, tw)}."""
    res = {}
    for fmt in ("nv4", "mx4"):
        res[fmt] = (R.quantize(X, fmt, rpad=Mp), R.quantize(dY, fmt, rpad=Mp), R.quantize_weight(W, fmt, two_d=True))
    return res


def run_case(model, M, name, N, K, kind, spark, mx8, g13, g17, *, head=False, X, dY, W, F4, MX8, extras):
    """build the candidate dict of one (linear, kind) and time it."""
    Mp = pad(M, 128)
    dev = "cuda"
    cands = {}
    flops = 2.0 * M * N * K
    epi = None if head else {"qkv": "bias", "proj": "res", "fc1": "gelu", "fc2": "res"}[name]
    xq, dyq, wq = MX8
    bias = extras["bias"][:N] if kind == "fwd" else None
    # ---------------------------------------------------------------- forward
    if kind == "fwd":
        gamma, resid = extras["gamma"][:N], extras["resid"][:, :N].contiguous()
        b_row, g_row = bias.reshape(1, -1), gamma.reshape(1, -1)
        E13, E4 = C.epilogues(), fp4_epi.nv4_epilogues()
        y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        y2 = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        o32 = torch.empty(M, N, device=dev, dtype=torch.float32)
        if head:
            cands["cur_FINAL(bf16)"] = lambda: E13["gemm"](X, W.t(), out=y)
            cands["mx8"] = lambda: E13["gemm"](xq.a_row(), wq.b_row_t(), out=y2)
        elif name == "qkv":
            cands["cur_FINAL(bf16_epi)"] = lambda: g13.linear_bias(X, W, bias, "qkv")
            cands["mx8_epi"] = lambda: g17.fwd_bias(xq, wq, bias)
        else:
            cands["cur_FINAL(mx8_epi)"] = {"proj": lambda: g17.fwd_res(xq, wq, bias, gamma, resid), "fc1": lambda: g17.fwd_gelu(xq, wq, bias),
                                           "fc2": lambda: g17.fwd_res(xq, wq, bias, gamma, resid)}[name]
            cands["bf16_epi"] = {"proj": lambda: g13.linear_res(X, W, bias, gamma, resid, "proj"), "fc1": lambda: g13.linear_gelu(X, W, bias),
                                 "fc2": lambda: g13.linear_res(X, W, bias, gamma, resid, "fc2")}[name]
        for fmt in ("nv4", "mx4"):
            tx, _, tw = F4[fmt]
            A_, B_ = (tx.q, tx.sf, tx.ts), (tw.q, tw.sf, tw.ts)
            ao, bo = C.operand(A_, fmt), C.operand(B_, fmt)
            cands[f"{fmt}_quack_plain"] = (lambda ao=ao, bo=bo: C.quack_fp4(ao, bo, y))
            if not head:
                qa, qb = tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t()
                kw = C.epi_kw(tx.sf, tw.sf, fmt)
                if fmt == "nv4":
                    ts = (tx.ts * tw.ts).reshape(1)
                    e = {"bias": lambda: E4["bias"](qa, qb, out={"D": y2}, bias=b_row, ts=ts, **kw),
                         "res": lambda: E4["res"](qa, qb, resid, out={"D": o32, "y": y2}, bias=b_row, gamma=g_row, ts=ts, **kw),
                         "gelu": lambda: E4["gelu"](qa, qb, out={"D": y2, "h": y}, bias=b_row, ts=ts, **kw)}[epi]
                    cands["nv4_quack_epi"] = e
                else:  # mx4: the round13 epilogues unchanged (no per-tensor scale)
                    cands["mx4_quack_epi"] = {"bias": lambda: E13["bias"](qa, qb, out={"D": y2}, bias=b_row, **kw),
                                              "res": lambda: E13["res"](qa, qb, resid, out={"D": o32, "y": y2}, bias=b_row, gamma=g_row, **kw),
                                              "gelu": lambda: E13["gelu"](qa, qb, out={"D": y2, "h": y}, bias=b_row, **kw)}[epi]
            if fmt == "nv4":
                cands["nv4_cublaslt_plain"] = lambda A_=A_, B_=B_: C.scaled_mm_fp4(A_, B_, "nv4", torch.bfloat16)
            if spark is not None:
                ok = N % 64 == 0 and K % 256 == 0
                if ok:
                    cands[f"{fmt}_spark"], v = spark.fn(A_, B_, fmt, M, N, K, y)
                    if M % 128 != 0:
                        Ap = (pad_rows(tx.q, Mp), tx.sf, tx.ts)
                        yp = torch.empty(Mp, N, device=dev, dtype=torch.bfloat16)
                        cands[f"{fmt}_spark_padM"], _ = spark.fn(Ap, B_, fmt, Mp, N, K, yp)
        flops_eff = flops
    # ---------------------------------------------------------------- dgrad:  out[M,K] = dY[M,N] . Wt[K,N]^T
    elif kind == "dgrad":
        E13, E4 = C.epilogues(), fp4_epi.nv4_epilogues()
        dx = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        dx2 = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        z = extras["z"][:, :K].contiguous() if (not head and name == "fc2") else None
        if head:
            cands["cur_FINAL(bf16)"] = lambda: E13["gemm"](dY, W, out=dx)
            cands["mx8"] = lambda: g17.dgrad(dyq, wq)
        elif name == "fc2":
            cands["cur_FINAL(mx8_dgelu)"] = lambda: g17.dgrad_dgelu(dyq, wq, z)
            cands["bf16_dgelu"] = lambda: g13.dgrad_dgelu(dY, W, z)
        else:
            cands["cur_FINAL(mx8)"] = lambda: g17.dgrad(dyq, wq)
            cands["bf16"] = lambda: g13.dgrad(dY, W, name)
        for fmt in ("nv4", "mx4"):
            _, tdy, tw = F4[fmt]
            A_, B_ = (tdy.q, tdy.sf, tdy.ts), (tw.qt, tw.sft, tw.tst)
            ao, bo = C.operand(A_, fmt), C.operand(B_, fmt)
            cands[f"{fmt}_quack_plain"] = (lambda ao=ao, bo=bo: C.quack_fp4(ao, bo, dx))
            if (not head) and name == "fc2":
                qa, qb = tdy.q.view(torch.float4_e2m1fn_x2), tw.qt.view(torch.float4_e2m1fn_x2).t()
                kw = C.epi_kw(tdy.sf, tw.sft, fmt)
                if fmt == "nv4":
                    ts = (tdy.ts * tw.tst).reshape(1)
                    cands["nv4_quack_epi(dgelu)"] = lambda: E4["dgelu"](qa, qb, z, out={"D": dx2}, ts=ts, **kw)
                else:
                    cands["mx4_quack_epi(dgelu)"] = lambda: E13["dgelu"](qa, qb, z, out={"D": dx2}, **kw)
            if fmt == "nv4":
                cands["nv4_cublaslt_plain"] = lambda A_=A_, B_=B_: C.scaled_mm_fp4(A_, B_, "nv4", torch.bfloat16)
            if spark is not None and K % 64 == 0 and N % 256 == 0:  # Spark: out cols (=K here) % 64, contraction (=N here) % 256
                cands[f"{fmt}_spark"], _ = spark.fn(A_, B_, fmt, M, K, N, dx)
                if M % 128 != 0:
                    Ap = (pad_rows(tdy.q, Mp), tdy.sf, tdy.ts)
                    dxp = torch.empty(Mp, K, device=dev, dtype=torch.bfloat16)
                    cands[f"{fmt}_spark_padM"], _ = spark.fn(Ap, B_, fmt, Mp, K, N, dxp)
    # ---------------------------------------------------------------- wgrad:  out[N,K] fp32 = dYt[N,Mp] . Xt[K,Mp]^T
    else:
        E13 = C.epilogues()
        dw = torch.empty(N, K, device=dev, dtype=torch.float32)
        dw2 = torch.empty(N, K, device=dev, dtype=torch.float32)
        cands["cur_FINAL(mx8_native)"] = lambda: g17.wgrad(dyq, xq)
        cands["bf16_torch_f32"] = lambda: torch.mm(dY.t(), X, out_dtype=torch.float32)
        cands["bf16_quack_f32"] = lambda: E13["gemm"](dY.t(), X, out=dw)
        cands["bf16_quack_f32_splitk"] = lambda: E13["gemm"](dY.t(), X, out=dw2, split_k=None)
        if not head:  # prefix-token side GEMM (cls + 4 registers per image; joint blocks: + 100 queries): gather + bf16 GEMM with fp32 out, then the add into the FP4 result
            n_img = 1129 if M % 1129 == 0 and M % 1029 != 0 else 1029
            Bimg = M // n_img
            for pref in ((5,) if n_img == 1029 else (5, 105)):
                idx = (torch.arange(Bimg, device=dev)[:, None] * n_img + torch.arange(pref, device=dev)[None, :]).reshape(-1)
                side = torch.empty(N, K, device=dev, dtype=torch.float32)
                cands[f"prefix{pref}_side_gather+mm_f32"] = (lambda idx=idx: torch.mm(dY.index_select(0, idx).t(), X.index_select(0, idx), out_dtype=torch.float32))
                cands[f"prefix{pref}_side_gather+mm_f32+add"] = (lambda idx=idx, side=side: dw.add_(torch.mm(dY.index_select(0, idx).t(), X.index_select(0, idx), out_dtype=torch.float32)))
        for fmt in ("nv4", "mx4"):
            tx, tdy, _ = F4[fmt]
            A_, B_ = (tdy.qt, tdy.sft, tdy.tst), (tx.qt, tx.sft, tx.tst)
            ao, bo = C.operand(A_, fmt), C.operand(B_, fmt)
            cands[f"{fmt}_quack_f32"] = (lambda ao=ao, bo=bo: C.quack_fp4(ao, bo, dw))
            cands[f"{fmt}_quack_f32_splitk"] = (lambda ao=ao, bo=bo: (E13["gemm"](ao, bo.mT, out=dw2, split_k=None)))
            if fmt == "nv4":
                cands["nv4_cublaslt_f32"] = lambda A_=A_, B_=B_: C.scaled_mm_fp4(A_, B_, "nv4", torch.float32)
                cands["nv4_cublaslt_bf16"] = lambda A_=A_, B_=B_: C.scaled_mm_fp4(A_, B_, "nv4", torch.bfloat16)
            if spark is not None:
                kp = pad(Mp, 256)
                qa, sa = pad_k(tdy.qt, tdy.sft, kp, fmt)
                qb, sb = pad_k(tx.qt, tx.sft, kp, fmt)
                o = torch.empty(N, K, device=dev, dtype=torch.bfloat16)
                cands[f"{fmt}_spark_bf16out_K{kp}"], _ = spark.fn((qa, sa, tdy.tst), (qb, sb, tx.tst), fmt, N, K, kp, o)
    us, errs = interleave(cands)
    rec = {"M": M, "N": N, "K": K, "flops": flops, "us": us, "tflops": {k: flops / v / 1e6 for k, v in us.items()}, "errors": errs}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--what", required=True, help="bb:<M> or head")
    ap.add_argument("--only", default=None, help="comma list of linears (bb) / stage ids (head)")
    ap.add_argument("--no-spark", action="store_true")
    args = ap.parse_args()
    cfg = CFG[args.model]
    d = cfg["d"]
    out_path = ROUND / "profile" / f"fp4_gemm_matrix_{args.model}_{args.what.replace(':', '')}.json"
    from round13_gemm_fusion_sm120.impl import gemm_ops as g13
    from round17_mxfp8_sm120.impl import gemm_mx8 as g17
    from round17_mxfp8_sm120.impl import mx8

    res = {"model": args.model, "what": args.what, "cases": {}}
    with gpu_lock():
        spark = None if args.no_spark else Spark()
        g = torch.Generator(device=dev).manual_seed(0)
        if args.what == "head":
            jobs = [(f"convT{i + 1}", M, 4 * (d), d) for i, M in enumerate(cfg["head"])]
        else:
            M = int(args.what.split(":")[1])
            lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
            jobs = [(n, M, lin[n][0], lin[n][1]) for n in lin if (args.only is None or n in args.only.split(","))]
        for name, M, N, K in jobs:
            Mp = pad(M, 128)
            head = args.what == "head"
            X = torch.randn(M, K, device=dev, generator=g).bfloat16()
            dY = torch.randn(M, N, device=dev, generator=g).bfloat16()
            W = (torch.randn(N, K, device=dev, generator=g) * 0.05).bfloat16()
            extras = {"bias": torch.randn(max(N, K), device=dev, generator=g), "gamma": torch.rand(max(N, K), device=dev, generator=g) * 0.1 + 0.05,
                      "resid": torch.randn(M, max(N, K), device=dev, generator=g), "z": torch.randn(M, max(N, K), device=dev, generator=g).bfloat16()}
            F4 = fp4_prep(X, dY, W, Mp)
            MX8 = (mx8.quantize(X, mpad=Mp), mx8.quantize(dY, mpad=Mp), mx8.quantize(W))
            for kind in ("fwd", "dgrad", "wgrad"):
                key = f"{name}/{kind}"
                try:
                    rec = run_case(args.model, M, name, N, K, kind, spark, mx8, g13, g17, head=head, X=X, dY=dY, W=W, F4=F4, MX8=MX8, extras=extras)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    rec = {"M": M, "N": N, "K": K, "fatal": f"{type(e).__name__}: {str(e)[:300]}"}
                res["cases"][key] = rec
                best = sorted(rec.get("us", {}).items(), key=lambda kv: kv[1])[:4]
                print(f"[{args.model} {args.what}] {key} M={M} N={N} K={K}: " + ", ".join(f"{k}={v:.0f}us/{rec['flops'] / v / 1e6:.0f}TF" for k, v in best), flush=True)
                if rec.get("errors"):
                    print("   errors:", {k: v[:90] for k, v in rec["errors"].items()}, flush=True)
                out_path.write_text(json.dumps(res, indent=1))
            del X, dY, W, extras, F4, MX8
            torch.cuda.empty_cache()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
