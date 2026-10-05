"""Round 11 part 2: attention candidate matrix, speed + operator-level accuracy.

  python attn_matrix.py --cands flash,tl,sage2_fp8,... --shapes 2x16x1029,... --out profile/attn_matrix_speed.json
  python attn_matrix.py --acc --real qkv_real_vitl16.pt --cands ...            # operator accuracy on captured model qkv
Speed: CUDA-graph replay (10 calls per graph), candidates interleaved in one process, median of 3 rounds, dual GPU lock; reported both
`kernel` (prep excluded: what a fused qkv epilogue would leave) and `full` (everything between the qkv GEMM and the proj GEMM).
Accuracy: error of O vs an fp64 attention of the same bf16 q/k/v.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs
import attn_cands as ac

ap = argparse.ArgumentParser()
ap.add_argument("--cands", default="flash,tl,sage2_fp8,sage2_fp8_f32,sage2_pv16,sage2_pv16_f32,sage3,a16k0,a16k1,a16k2,a16k3,a16k4")
ap.add_argument("--shapes", default="2x16x1029,2x16x1129,2x12x1029,2x12x1129,4x16x1029,4x16x1129,4x12x1029,4x12x1129")
ap.add_argument("--out", default="")
ap.add_argument("--acc", action="store_true")
ap.add_argument("--real", default="")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--a16cfg", default="")
a = ap.parse_args()
cands = a.cands.split(",")


def flops(B, H, N):
    return 4 * B * H * N * N * 64


def graph_of(fn, rep=10):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(rep):
            fn()
    return g


def time_graphs(graphs, rep=10, iters=100, rounds=3):
    ts = [[] for _ in graphs]
    for _ in range(rounds):
        for i, g in enumerate(graphs):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(iters // rep):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts[i].append(e0.elapsed_time(e1) * 1e3 / iters)
    return [statistics.median(t) for t in ts]


def ref_attn(qkv):
    q, k, v = [t.double().permute(0, 2, 1, 3) for t in (qkv[0], qkv[1], qkv[2])]
    out = []
    for b in range(q.shape[0]):
        s = torch.softmax(q[b] @ k[b].transpose(-1, -2) * 0.125, -1)
        out.append(s @ v[b])
    return torch.stack(out).permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


def err(o, ref):
    d = o.double() - ref
    return dict(rel_l2=(d.norm() / ref.norm()).item(), max_abs=d.abs().max().item(),
                cos=torch.nn.functional.cosine_similarity(o.double().flatten(), ref.flatten(), dim=0).item(), finite=bool(torch.isfinite(o.float()).all()))


# build/compile everything outside the lock
splits = {}
for n in cands:
    try:
        splits[n] = ac.get_split(n)
    except Exception as e:  # noqa: BLE001
        print(f"candidate {n} unavailable: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("candidates:", list(splits), flush=True)

result = {"env": {"time": time.strftime("%Y-%m-%d %H:%M:%S")}, "speed": {}, "acc": {}}
with gpu_lock2(), torch.inference_mode():
    result["env"]["other_gpu_procs"] = other_gpu_procs()
    print("other gpu procs:", result["env"]["other_gpu_procs"], flush=True)
    if not a.acc:
        for sh in a.shapes.split(","):
            B, H, N = [int(x) for x in sh.split("x")]
            torch.manual_seed(0)
            qkv = torch.randn(3, B, N, H, 64, device="cuda").bfloat16().contiguous()
            kg, fg, names = [], [], []
            for n, (prep, kern) in splits.items():
                try:
                    st = prep(qkv)
                    kern(st)
                    torch.cuda.synchronize()
                    kg.append(graph_of(lambda st=st, kern=kern: kern(st)))
                    fg.append(graph_of(lambda prep=prep, kern=kern: kern(prep(qkv))))
                    names.append(n)
                except Exception as e:  # noqa: BLE001
                    print(f"  {sh} {n}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
            tk = time_graphs(kg, rounds=a.rounds)
            tf = time_graphs(fg, rounds=a.rounds)
            fl = flops(B, H, N) / 1e12
            result["speed"][sh] = {n: {"kernel_us": k, "full_us": f, "kernel_TFLOPS": fl / (k * 1e-6), "full_TFLOPS": fl / (f * 1e-6)}
                                   for n, k, f in zip(names, tk, tf)}
            print(sh, "  ".join(f"{n}: {k:.1f}/{f:.1f}" for n, k, f in zip(names, tk, tf)), flush=True)
            if a.out:
                Path(a.out).write_text(json.dumps(result, indent=1))
    else:
        data = torch.load(a.real)
        for key, qkv in data.items():
            qkv = qkv.cuda().contiguous()
            ref = ref_attn(qkv)
            result["acc"][key] = {}
            for n, (prep, kern) in splits.items():
                try:
                    result["acc"][key][n] = err(kern(prep(qkv)), ref)
                except Exception as e:  # noqa: BLE001
                    result["acc"][key][n] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
            print(key, "  ".join(f"{n}: {v.get('rel_l2', float('nan')):.2e}" for n, v in result["acc"][key].items()), flush=True)
            if a.out:
                Path(a.out).write_text(json.dumps(result, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(result, indent=1))
