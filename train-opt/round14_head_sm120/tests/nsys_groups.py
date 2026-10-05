"""Group the GPU kernels of one captured predict (fwd/bwd NVTX ranges, profile_head.py) by sub-op.

Usage: python nsys_groups.py <prefix> [<prefix> ...]      (reads <prefix>.sqlite; averages the captured steps)
Groups (by kernel name): ConvT/Linear/einsum GEMM, stage elementwise (GELU, dwconv, LayerNorm, bias add, fused stage kernels),
MXFP8 quantize, copies/reductions/other.
"""
import re
import sqlite3
import sys
from collections import defaultdict

GROUPS = [
    ("GEMM (ConvT / einsum / Linear)", r"cudnn::Kernel|xmma|quackgemm|cutlass::Kernel2|cutlass_|cublas|gemm"),
    ("stage elementwise (GELU/dwconv/LN/bias; fused)", r"Gelu|layer_norm|GammaBeta|conv2d_c1|dgrad2d_c1|wgrad2d_c1|fwd_kernel|bwd1_kernel|bwd2_kernel|_fwd_kernel|_bwd1|_bwd2"),
    ("MXFP8 quantize", r"quant_dual"),
]


def classify(name: str) -> str:
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return "copies / reductions / other"


def one(prefix):
    con = sqlite3.connect(f"{prefix}.sqlite")
    names = {i: v for i, v in con.execute("SELECT id,value FROM StringIds")}
    runtime = {c: s for c, s in con.execute("SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    nv = [(s, e, t if t is not None else names.get(tid, "")) for s, e, t, tid in con.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    steps = sorted((s, e) for s, e, n in nv if re.fullmatch(r"step_\d+", n))
    tags = sorted((s, e, n) for s, e, n in nv if n in ("fwd", "bwd"))
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for s, e, c, n in con.execute("SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        launch = runtime.get(c, s)
        if not any(ss <= launch <= se for ss, se in steps):
            continue
        tag = next((t for ts, te, t in tags if ts <= launch <= te), "?")
        g = classify(names.get(n, "?"))
        tot[(tag, g)] += (e - s) / 1e3
        cnt[(tag, g)] += 1
    ns = max(len(steps), 1)
    return {k: (v / ns, cnt[k] / ns) for k, v in tot.items()}, ns


def main():
    for prefix in sys.argv[1:]:
        res, ns = one(prefix)
        print(f"== {prefix.split('/')[-1]} (avg of {ns} steps), us per predict")
        allg = [g for g, _ in GROUPS] + ["copies / reductions / other"]
        print(f"{'group':50s} {'fwd us':>9s} {'(n)':>5s} {'bwd us':>9s} {'(n)':>5s} {'total us':>9s}")
        sf = sb = 0.0
        for g in allg:
            f, nf = res.get(("fwd", g), (0.0, 0))
            b, nb = res.get(("bwd", g), (0.0, 0))
            sf += f
            sb += b
            print(f"{g:50s} {f:9.1f} {nf:5.1f} {b:9.1f} {nb:5.1f} {f + b:9.1f}")
        print(f"{'TOTAL':50s} {sf:9.1f} {'':5s} {sb:9.1f} {'':5s} {sf + sb:9.1f}")


if __name__ == "__main__":
    main()
