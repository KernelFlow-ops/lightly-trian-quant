"""Mainloop efficiency probe: own mgx (EPI 4, plain bf16) vs cuBLASLt `_scaled_mm` on a big GEMM (no wave / prologue / epilogue effects to speak of).
  python tests/bench_big.py M N K [json cfg ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
for p in (HERE / "impl", ROOT / "common"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
import mx13  # noqa: E402
import mgx  # noqa: E402
from lock2 import gpu_lock2  # noqa: E402

M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfgs = [json.loads(a) for a in sys.argv[4:]] or [mgx.DEFAULT["plain"]]
E = mx13.ext()
X = torch.randn(M, K, device="cuda") * 0.5
xb, W = mx13.ActBuf(M, K), mx13.MXW(torch.randn(N, K, device="cuda") * 0.03)
E.quant_rows(X, xb.q, xb.sf)
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)


def timeit(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
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


with gpu_lock2(), torch.inference_mode():
    fl = 2 * M * N * K
    us = timeit(lambda: mx13.mm(xb, W))
    print(f"cuBLASLt  M={M} N={N} K={K}: {us:.1f} us  {fl / us / 1e6:.0f} TFLOPS")
    for c in cfgs:
        c = dict(c, epi=4)
        kern = mgx.get(c)
        us = timeit(lambda: kern(xb, W, M, N, K, C=out))
        print(f"mgx {json.dumps(c)}: {us:.1f} us  {fl / us / 1e6:.0f} TFLOPS")
