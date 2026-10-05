"""post_a2 vs round7 post_a on real model outputs: lr error, active-query statistics, skipping error, graph timing.

  python tests/test_post.py --model vitl16 [--cfgs '[{"NG":8}]']
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs

HERE = Path(__file__).resolve().parents[1] / "impl"


def load_ext(**cfg):
    from torch.utils.cpp_extension import load
    tag = hashlib.md5(repr(sorted(cfg.items())).encode()).hexdigest()[:8]
    build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_post_" + tag)
    build.mkdir(parents=True, exist_ok=True)
    return load(name="r11_post_" + tag, sources=[str(HERE / "postfuse_ext11.cu")], build_directory=str(build),
                extra_cuda_cflags=["-O3", "-arch=sm_120a"] + [f"-D{k}={v}" for k, v in cfg.items()], verbose=False)


def bench(fn, iters=100, reps=5):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters // 10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / iters)
    return sorted(ts)[len(ts) // 2]


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitl16")
ap.add_argument("--cfgs", default='[{"NG":8}]')
ap.add_argument("--eps", default="1e-4,1e-3,1e-2")
a = ap.parse_args()
cfgs = json.loads(a.cfgs)
frames = c.mf.load_frames()
e7 = c.pf.ext()
exts = [(cfg, load_ext(**cfg)) for cfg in cfgs]       # compile outside the GPU lock
print("compiled", len(exts), flush=True)
with gpu_lock2(), torch.inference_mode():
    print("other gpu procs:", other_gpu_procs())
    model = c.mf.load_model(a.model)
    fn, pcls = c.make_variant(a.model, model, "r5_v1")
    pipe = c.make_pipeline(model, fn, pcls, slots=1)
    for fi in (0, 3, 7):
        pipe.pinned_np[0][...] = frames[fi]
        pipe.frame_gpu[0].copy_(pipe.pinned[0])
        pipe._prep_launch(0)
        ml, cl = pipe.model_fn(pipe.batch)
        ml, cl = ml.contiguous(), cl.contiguous()
        lr_old = torch.zeros_like(pipe.lr)
        e7.post_a_launch(ml, cl, lr_old, pipe.starts, pipe.crop)
        pr = torch.softmax(cl.float(), -1)[..., :2].sum(-1)       # [2,Q] p0+p1
        for eps in (1e-4, 1e-3, 1e-2):
            print(f"  frame {fi} eps {eps:g}: active queries per crop {[(pr[ci] >= eps).sum().item() for ci in range(2)]} / {pr.shape[1]}  "
                  f"skipped mass bound {[pr[ci][pr[ci] < eps].sum().item() for ci in range(2)]}")
        t_old = bench(lambda: e7.post_a_launch(ml, cl, lr_old, pipe.starts, pipe.crop))
        for cfg, e in exts:
            lr_new = torch.zeros_like(lr_old)
            e.post_a2_launch(ml, cl, lr_new, pipe.starts, pipe.crop, 0.0)
            d = (lr_new - lr_old).abs()
            line = f"frame {fi} {cfg}: old {t_old:6.1f} us  new {bench(lambda: e.post_a2_launch(ml, cl, lr_new, pipe.starts, pipe.crop, 0.0)):6.1f} us  max_abs {d.max().item():.3g} mean_abs {d.mean().item():.3g}"
            for eps in [float(x) for x in a.eps.split(",")]:
                lr_s = torch.zeros_like(lr_old)
                e.post_a2_launch(ml, cl, lr_s, pipe.starts, pipe.crop, eps)
                ds = (lr_s - lr_old).abs()
                t_s = bench(lambda: e.post_a2_launch(ml, cl, lr_s, pipe.starts, pipe.crop, eps))
                line += f" | eps {eps:g}: {t_s:5.1f} us max_abs {ds.max().item():.3g}"
            print(line, flush=True)
