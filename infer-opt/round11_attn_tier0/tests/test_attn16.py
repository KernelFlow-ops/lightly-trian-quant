"""Correctness (vs fp64 reference on the kernel's own inputs) and graph-replay speed of the FP16-accumulate FA2 prototype.

  python tests/test_attn16.py [--cfgs '[{"CW":8,"STAGES":4,"MINB":1}]'] [--kinds 0,1,2,3,4] [--speed] [--real]
Inputs: random (scale 1), spiky (scale 8 on q/k, DINOv3-register-like) and, with --real, captured model qkv (qkv_real_<model>.pt).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs
import attn16

ap = argparse.ArgumentParser()
ap.add_argument("--cfgs", default='[{"CW":8,"STAGES":4,"MINB":1}]')
ap.add_argument("--kinds", default="0,1,2,3,4")
ap.add_argument("--speed", action="store_true")
ap.add_argument("--real", default="")
ap.add_argument("--shapes", default="2x16x1029,2x16x1129,2x12x1029,2x12x1129")
a = ap.parse_args()
cfgs = json.loads(a.cfgs)
kinds = [int(x) for x in a.kinds.split(",")]
exts = [(cfg, attn16.ext(**cfg)) for cfg in cfgs]       # compile outside the lock
print("compiled", len(exts), flush=True)


def ref_attn(qkv):
    """fp64 reference: qkv [3,B,N,H,64] -> [B,N,H*64]"""
    q, k, v = [t.double().permute(0, 2, 1, 3) for t in (qkv[0], qkv[1], qkv[2])]    # [B,H,N,D]
    out = []
    for b in range(q.shape[0]):
        s = torch.softmax(q[b] @ k[b].transpose(-1, -2) * 0.125, -1)
        out.append((s @ v[b]))                                                     # [H,N,D]
    o = torch.stack(out)                                                           # [B,H,N,D]
    return o.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


def flash(qkv):
    B, N, H = qkv.shape[1:4]
    q = qkv.transpose(2, 3)
    o = F.scaled_dot_product_attention(q[0], q[1], q[2])
    return o.transpose(1, 2).reshape(B, N, H * 64)


def metrics(o, ref):
    d = (o.double() - ref)
    return dict(max_abs=d.abs().max().item(), rel_l2=(d.norm() / ref.norm()).item(),
                cos=torch.nn.functional.cosine_similarity(o.double().flatten(), ref.flatten(), dim=0).item())


def bench(fns, iters=100, reps=3):
    gs = []
    for f in fns:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                f()
        gs.append(g)
    ts = [[] for _ in fns]
    for _ in range(reps):
        for i, g in enumerate(gs):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(iters // 10):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts[i].append(e0.elapsed_time(e1) * 1e3 / iters)
    return [sorted(t)[len(t) // 2] for t in ts]


def cases():
    torch.manual_seed(0)
    for sh in a.shapes.split(","):
        B, H, N = [int(x) for x in sh.split("x")]
        for name, qs in (("rand", 1.0), ("spiky8", 8.0)):
            qkv = torch.randn(3, B, N, H, 64, device="cuda")
            qkv[0] *= qs
            qkv[1] *= qs
            yield f"{sh} {name}", qkv.bfloat16().contiguous()
    if a.real:
        d = torch.load(a.real)
        for key, qkv in d.items():
            yield f"real {key}", qkv.cuda().contiguous()


with gpu_lock2(), torch.inference_mode():
    print("other gpu procs:", other_gpu_procs())
    for name, qkv in cases():
        ref = ref_attn(qkv)
        qkv16 = qkv.half().contiguous()
        line = f"{name:28s} flash(bf16) rel_l2 {metrics(flash(qkv), ref)['rel_l2']:.2e} |"
        for cfg, e in exts:
            for k in kinds:
                x = qkv if k == 0 else qkv16
                try:
                    m = metrics(e.attn16_fwd(x, k), ref)
                    line += f" k{k} rel_l2 {m['rel_l2']:.2e} max {m['max_abs']:.2e} |"
                except Exception as ex:  # noqa: BLE001
                    line += f" k{k} ERR {str(ex)[:60]} |"
        print(line, flush=True)
        if a.speed and "rand" in name:
            fns = [lambda: flash(qkv)]
            labels = ["flash"]
            for cfg, e in exts:
                for k in kinds:
                    x = qkv if k == 0 else qkv16
                    fns.append((lambda e=e, x=x, k=k: e.attn16_fwd(x, k)))
                    labels.append(f"{cfg} k{k}")
            ts = bench(fns)
            tf = 4 * qkv.shape[1] * qkv.shape[3] * qkv.shape[2] ** 2 * 64 / 1e12
            print("   speed us:", "  ".join(f"{l}: {t:.1f} ({tf / (t * 1e-6):.0f} TF)" for l, t in zip(labels, ts)), flush=True)
