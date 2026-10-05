"""Candidate matrix, group "torch": SDPA backends (flash / cudnn / efficient) and the default / r4 attn_core paths.

fwd = forward with autograd graph recording, bwd = autograd.grad with retain_graph (includes the
unbind/stack gradient copy and inverse RoPE when RoPE is part of the candidate), fwd+bwd = both.
Run under the GPU lock:  gpu_run.sh <log> python tests/bench_torch.py --model vitl16
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl

BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "auto": None}


def core(backend, mask_kind, mask_bool, mask_bias, h):
    def f(qkv):
        b, n = qkv.shape[:2]
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        m = None
        if mask_kind == "bool_expand":
            m = mask_bool[:, None].expand(-1, h, -1, -1)
        elif mask_kind == "bool_b1nn":
            m = mask_bool[:, None]
        elif mask_kind == "bias_b1nn":
            m = mask_bias[:, None]
        elif mask_kind == "bias_expand":
            m = mask_bias[:, None].expand(-1, h, -1, -1)
        if backend is None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        else:
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return o.transpose(1, 2).reshape(b, n, -1)
    return f


def measure(name, f, x, dy, results, meta, flops_f, flops_b):
    rec = dict(meta, variant=name)
    try:
        y = f(x)
        torch.autograd.grad(y, x, dy, retain_graph=True)
        torch.cuda.synchronize()
    except Exception as e:  # unsupported combination
        rec["error"] = str(e).split("\n")[0][:200]
        results.append(rec)
        print(json.dumps(rec), flush=True)
        return None
    return (name, f, rec)


def run_group(cands, x, dy, meta, flops_f, flops_b, results, iters):
    ready = []
    for name, f in cands.items():
        r = measure(name, f, x, dy, results, meta, flops_f, flops_b)
        if r:
            ready.append(r)
    fwd = bl.interleaved({n: (lambda f=f: f(x)) for n, f, _ in ready}, iters=iters)
    ys = {n: f(x) for n, f, _ in ready}
    bwd = bl.interleaved({n: (lambda n=n: torch.autograd.grad(ys[n], x, dy, retain_graph=True)) for n, _, _ in ready}, iters=iters)
    both = bl.interleaved({n: (lambda f=f: torch.autograd.grad(f(x), x, dy)) for n, f, _ in ready}, iters=iters)
    for n, f, rec in ready:
        rec.update(fwd_us=fwd[n], bwd_us=bwd[n], fwd_bwd_us=both[n],
                   fwd_tflops=flops_f / fwd[n] / 1e6, bwd_tflops=flops_b / bwd[n] / 1e6,
                   fwd_bwd_tflops=(flops_f + flops_b) / both[n] / 1e6)
        results.append(rec)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vitl16")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    bl.require_lock()
    bl.warm_gpu()
    B, H = bl.SHAPES[a.model]
    torch.manual_seed(0)
    results = []
    for kind, n in (("plain", bl.N_PLAIN), ("joint", bl.N_JOINT)):
        x = torch.randn(B, n, 3, H, bl.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(B, n, H * bl.D, device="cuda", dtype=torch.bfloat16)
        fl_f, fl_b = bl.flops_fwd(B, H, n), bl.flops_bwd(B, H, n)
        meta = {"model": a.model, "kind": kind, "B": B, "H": H, "N": n, "group": "torch"}
        if kind == "plain":
            cands = {f"sdpa_{b}_nomask": core(be, "none", None, None, H) for b, be in BACKENDS.items()}
            run_group(cands, x, dy, dict(meta, mask="none"), fl_f, fl_b, results, a.iters)
        else:
            logits = torch.randn(B, 100, 128, 128, device="cuda")
            for phase, disabled in (("p1.0", None), ("p0.5", torch.rand(B, 100, device="cuda") > 0.5), ("p0.0_dense_true", torch.ones(B, 100, dtype=torch.bool, device="cuda"))):
                mb = bl.dense_mask(logits, n, disabled)
                mbias = torch.zeros(B, n, n, device="cuda", dtype=torch.bfloat16).masked_fill_(~mb, float("-inf"))
                cands = {}
                for b, be in BACKENDS.items():
                    for mk in ("bool_expand", "bool_b1nn", "bias_b1nn"):
                        cands[f"sdpa_{b}_{mk}"] = core(be, mk, mb, mbias, H)
                run_group(cands, x, dy, dict(meta, mask=phase), fl_f, fl_b, results, a.iters)
            run_group({"sdpa_auto_nomask": core(None, "none", None, None, H)}, x, dy, dict(meta, mask="none"), fl_f, fl_b, results, a.iters)
    if a.output:
        a.output.write_text(json.dumps({"results": results, "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
