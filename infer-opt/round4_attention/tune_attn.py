"""Tune BLOCK_M/BLOCK_N/stages/threads of the TileLang attention kernels (GPU time via CUDA-graph replay)."""
import argparse, itertools, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, str(HERE.parent / "common"))
import torch
import ops
from bench import gpu_lock, write_results

def gpu_time_us(fn, reps=20, iters=7):
    """Median GPU time per call: capture `reps` calls into one CUDA graph, replay `iters` times."""
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters + 2):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    ts = sorted(ts[2:])
    return ts[len(ts) // 2]

def make(kind, B, H, N):
    torch.manual_seed(0)
    qkv = torch.randn(B, N, 3, H, 64, device="cuda", dtype=torch.bfloat16) * 2
    sin = torch.randn(1024, 64, device="cuda"); sin[:, 32:] = sin[:, :32]
    cos = torch.randn(1024, 64, device="cuda"); cos[:, 32:] = cos[:, :32]
    prefix = N - 1024
    if kind == "plain_qkv":
        return lambda cfg: (lambda: ops.attention_qkv(qkv, cfg))
    if kind == "rope_qkv":
        return lambda cfg: (lambda: ops.attention_rope_qkv(qkv, sin, cos, prefix, cfg))
    q, k, v = [qkv[:, :, i].transpose(1, 2).contiguous() for i in range(3)]
    if kind == "plain":
        return lambda cfg: (lambda: ops.attention(q, k, v, cfg))
    return lambda cfg: (lambda: ops.attention_rope(q, k, v, sin, cos, prefix, cfg))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="plain_qkv"); ap.add_argument("--N", type=int, default=1029)
    ap.add_argument("--H", type=int, default=16); ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--bm", default="64,128"); ap.add_argument("--bn", default="32,64,128")
    ap.add_argument("--stages", default="1,2,3"); ap.add_argument("--threads", default="128,256")
    a = ap.parse_args()
    mk = make(a.kind, a.B, a.H, a.N)
    flops = 4 * a.B * a.H * a.N * a.N * 64
    res = []
    for bm, bn, st, th in itertools.product(*[[int(x) for x in s.split(",")] for s in (a.bm, a.bn, a.stages, a.threads)]):
        cfg = dict(bm=bm, bn=bn, stages=st, threads=th)
        try:
            fn = mk(cfg)
            fn()  # compile outside the lock
        except Exception as e:  # noqa: BLE001
            print(cfg, "FAILED", str(e).splitlines()[-1][:100], flush=True); continue
        with gpu_lock():
            us = gpu_time_us(fn)
        res.append({**cfg, "us": us, "tflops": flops / us / 1e6})
        print(cfg, f"{us:.1f} us  {flops / us / 1e6:.1f} TFLOPS", flush=True)
    res.sort(key=lambda r: r["us"])
    print("BEST", res[:3])
    out = HERE / "tune"; out.mkdir(exist_ok=True)
    json.dump(res, open(out / f"{a.kind}_B{a.B}_H{a.H}_N{a.N}.json", "w"), indent=1)
