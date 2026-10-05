"""Tile-config exploration for the TileLang GEMM kernels (mainloop variants). Timing under GPU lock.
usage: python tune_gemm.py <gelu|res_fc2|res_proj|qkv> [model]"""
import itertools
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import torch
from bench import time_fn, write_results
from roofline import waves

from impl import ops

which = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "vitl16"
d, ff, H = {"vitl16": (1024, 4096, 16), "vitb16": (768, 3072, 12)}[model]
Ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 1029
B = 2
M = B * Ntok
dev, bf = "cuda", torch.bfloat16
r = lambda *s: (torch.randn(*s, device=dev)).to(bf)  # noqa: E731
if which == "gelu":
    K_, Nn = d, ff
    h, W, b = r(M, d), r(ff, d) * d ** -0.5, r(ff)
    mk = lambda c: (lambda: ops.gemm_bias_gelu(h, W, b, cfg=c))  # noqa: E731
elif which in ("res_fc2", "res_proj"):
    K_, Nn = (ff if which == "res_fc2" else d), d
    h, W, b, res = r(M, K_), r(Nn, K_) * K_ ** -0.5, r(Nn), r(M, Nn)
    mk = lambda c: (lambda: ops.gemm_bias_residual(h, W, b, res, cfg=c))  # noqa: E731
else:
    K_, Nn = d, 3 * d
    h, W, b = r(B, Ntok, d), r(3 * d, d) * d ** -0.5, r(3 * d)
    ang = torch.randn(1024, 32, device=dev)
    sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()
    mk = lambda c: (lambda: ops.qkv_rope(h, W, b, sin, cos, 5, H, cfg=c))  # noqa: E731

grid = []
for bm, bn in ((128, 128), (128, 64), (64, 128), (64, 64), (128, 32)):
    for bk, st in ((32, 3), (32, 4)):
        for th, ws, tma in ((256, True, True), (128, False, False), (128, True, True)):
            if which == "qkv" and bn not in (64, 128):
                continue
            if (bm * bn) // th < 16:
                continue
            grid.append(dict(bm=bm, bn=bn, bk=bk, stages=st, threads=th, ws=ws, tma=tma))
print(len(grid), "configs", flush=True)
rows = []
for c in grid:
    try:
        fn = mk(c)
        fn()
        torch.cuda.synchronize()
        us = time_fn(fn, warmup=20, iters=50, repeats=3)["gpu_ms_p50"] * 1e3
    except Exception as e:  # noqa: BLE001
        print("FAIL", c, str(e)[:100].replace("\n", " "), flush=True)
        continue
    w = waves(M, Nn, c["bm"], c["bn"])
    tf = 2 * M * Nn * K_ / us / 1e6
    rows.append((us, c))
    print(f"{us:8.1f}us {tf:6.1f}TF tiles={w['tiles']:5d} waves={w['waves']:5.2f} {c}", flush=True)
rows.sort(key=lambda x: x[0])
print("TOP5")
for us, c in rows[:5]:
    print(f"{us:8.1f}us {c}")
write_results(f"/root/lightly-train/infer-opt/round3_fusion/tune_{which}_{model}_N{Ntok}.json",
              {"rows": [{"us": u, "cfg": c} for u, c in rows]})
