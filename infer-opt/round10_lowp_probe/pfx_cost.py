"""Cost of keeping the prefix rows on the bf16 path: one extra tiny bf16 GEMM per quantised GEMM on the gathered prefix rows
(2 crops x 5 rows = 10 rows in normal layers, 2 x 105 = 210 rows in the joint layers), cuBLAS bf16, 16 back-to-back calls per graph.
The scatter of its output rows into the main GEMM output is NOT measured (assumed to fold into the epilogue; budget +2 us in speed_model)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
import torch
import torch.nn.functional as F
import bench10 as b

res = {}
for model in b.MODELS:
    for kind in ("qkv", "proj", "fc1", "fc2"):
        N, K = b.kind_nk(model, kind)
        for M, rows in ((2058, 10), (2258, 210)):
            R = b.R_FOR(N * K * 2)
            X = torch.randn(rows, K, device="cuda").bfloat16()
            Ws = [torch.randn(N, K, device="cuda").bfloat16() * 0.03 for _ in range(R)]
            c = b.Cand("pfx_bf16", lambda s, X=X, Ws=Ws: [(lambda W=W: F.linear(X, W)) for W in Ws])
            shape = dict(model=model, kind=kind, M=M, N=N, K=K)
            r = b.time_shape(shape, [c], rounds=3, verbose=False)
            res[f"{model}|{kind}|{M}"] = {"pfx_bf16": {k: v for k, v in r["pfx_bf16"].items() if k in ("us", "foreign_gpu_procs")}, "rows": rows}
            print(model, kind, M, rows, round(r["pfx_bf16"]["us"], 1), flush=True)
b.merge_results(HERE / "results_pfx.json", res)
