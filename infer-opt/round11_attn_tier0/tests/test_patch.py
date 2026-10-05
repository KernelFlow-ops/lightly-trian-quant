"""Correctness + microbench of the patch-embed GEMM vs the reference bb.prepare_tokens_with_masks (bf16 autocast) + .float().

  python tests/test_patch.py --model vitl16 [--tune]
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs
import patch_embed as pe

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitl16")
ap.add_argument("--tune", action="store_true")
a = ap.parse_args()

model = c.mf.load_model(a.model)
bb = model.backbone


def ref(x):
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        t, _ = bb.prepare_tokens_with_masks(x)
    return t.float().contiguous()


def bench(fn, iters=200):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters // 10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / iters)
    return sorted(ts)[len(ts) // 2]


with gpu_lock2(), torch.inference_mode():
    print("other gpu procs:", other_gpu_procs())
    x = torch.randn(2, 3, 512, 512, device="cuda").bfloat16()
    r = ref(x)
    g = pe.PatchEmbedGemm(bb)
    o = g(x)
    d = (o - r).abs()
    ne = (o != r)
    print(f"{a.model}: shape {tuple(o.shape)}  max_abs {d.max().item():.4g}  mean_abs {d.mean().item():.4g}  "
          f"frac_differing {ne.float().mean().item():.5f}  prefix_equal {torch.equal(o[:, :5], r[:, :5])}  "
          f"cos {torch.nn.functional.cosine_similarity(o.double().flatten(), r.double().flatten(), dim=0).item():.8f}")
    # rel error on differing elements (bf16 ulp = 2^-8 relative)
    rel = (d / r.abs().clamp_min(1e-6))[ne]
    print("  differing elems: max rel %.4g (bf16 ulp 3.9e-3), count %d of %d" % (rel.max().item() if rel.numel() else 0, ne.sum().item(), ne.numel()))
    # real frame (kept in bf16 as the pipeline feeds it)
    t_ref = bench(lambda: ref(x))
    t_new = bench(lambda: g(x))
    print(f"graph time/call: reference {t_ref:.1f} us   gemm {t_new:.1f} us   (other procs {other_gpu_procs()})")
    if a.tune:
        cfgs = [dict(BM=64, BN=64, BK=32, ST=4, WM=2, WN=2, MINB=3), dict(BM=64, BN=64, BK=64, ST=3, WM=2, WN=2, MINB=2),
                dict(BM=64, BN=128, BK=32, ST=3, WM=2, WN=2, MINB=2), dict(BM=128, BN=64, BK=32, ST=3, WM=2, WN=2, MINB=2),
                dict(BM=64, BN=64, BK=32, ST=3, WM=2, WN=1, MINB=4), dict(BM=128, BN=128, BK=32, ST=3, WM=2, WN=4, MINB=2)]
        for cfg in cfgs:
            try:
                gg = pe.PatchEmbedGemm(bb, **cfg)
                oo = gg(x)
                err = (oo - r).abs().max().item()
                t = bench(lambda: gg(x))
                print(f"{t:7.1f} us err {err:.3g} {cfg}", flush=True)
            except Exception as e:  # noqa: BLE001
                print("fail", cfg, str(e)[:200], flush=True)
