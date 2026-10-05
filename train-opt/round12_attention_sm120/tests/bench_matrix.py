"""Candidate matrix at ops.attn_core level (packed qkv [B,N,3C] bf16 in, [B,N,C] out, RoPE included) on the real shapes.

fwd = forward with autograd graph, bwd = autograd.grad(retain_graph) of that graph, fwd+bwd = both (CUDA events, interleaved candidates,
median of 3 x 30 launches).  TFLOPS use the useful (unpadded) FLOPs: fwd 4*B*H*N^2*64, bwd 10*B*H*N^2*64.

Groups:  attn_core  - full-op candidates: torch default (T-BF16 reference path), round4 sdpa_rope, TB1 (round11: triton rope + flash /
                      qsplit cuDNN), round12 fused (+ no-rope and device-prob variants)
         core       - RoPE excluded, third party kernels on q/k/v views of the packed tensor: torch flash / cuDNN, cuDNN-frontend sm120 DSL
                      (fwd+bwd), FA4 cute sm120, Spark attention (kernel only and with the layout copies a packed-qkv caller pays)
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import attn_ext  # noqa: E402
from round12_attention_sm120.impl.bitmask import QueryBitmask  # noqa: E402
from round12_attention_sm120.impl.fused import make_attn_core as make_fused  # noqa: E402
from common.ops import default_ops  # noqa: E402


class DummyCtx:
    pass


def t_us(rows, key):
    return rows.get(key)


def measure(cands, qkv, dy, flops_f, flops_b, iters):
    """cands: name -> fn(qkv) -> out [B,N,C].  Returns name -> dict."""
    ready, out = {}, {}
    for name, f in cands.items():
        try:
            y = f(qkv)
            torch.autograd.grad(y, qkv, dy, retain_graph=True)
            torch.cuda.synchronize()
            ready[name] = f
        except Exception as e:  # unsupported combination
            out[name] = {"error": str(e).split("\n")[0][:160]}
    ys = {}
    fwd = bl.interleaved({n: (lambda f=f: f(qkv)) for n, f in ready.items()}, iters=iters)
    ys = {n: f(qkv) for n, f in ready.items()}
    bwd = bl.interleaved({n: (lambda n=n: torch.autograd.grad(ys[n], qkv, dy, retain_graph=True)) for n in ready}, iters=iters)
    both = bl.interleaved({n: (lambda f=f: torch.autograd.grad(f(qkv), qkv, dy)) for n, f in ready.items()}, iters=iters)
    for n in ready:
        out[n] = {"fwd_us": fwd[n], "bwd_us": bwd[n], "fwd_bwd_us": both[n], "fwd_tflops": flops_f / fwd[n] / 1e6, "bwd_tflops": flops_b / bwd[n] / 1e6,
                  "fwd_bwd_tflops": (flops_f + flops_b) / both[n] / 1e6}
    del ys
    return out


def make_inputs(B, H, N, joint, seed=0):
    torch.manual_seed(seed)
    qkv = torch.randn(B, N, 3 * H * 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(B, N, H * 64, device="cuda", dtype=torch.bfloat16)
    rope = bl.make_rope()
    return qkv, dy, rope


def attn_core_group(model, B, H, iters, results):
    from round4_attention.impl.rope import sdpa_rope
    from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core as make_tb1

    ctx = DummyCtx()
    base = default_ops().attn_core
    tb1 = make_tb1(AttnCfg(rope="triton", plain="flash", joint="cudnn", qsplit=True, nq=100), fallback=base)
    fused = make_fused()
    for kind, N in (("plain", 1029), ("joint", 1129)):
        qkv, dy, rope = make_inputs(B, H, N, kind == "joint")
        fl_f, fl_b = bl.flops_fwd(B, H, N), bl.flops_bwd(B, H, N)
        prefix = N - 1024
        phases = [("none", None)] if kind == "plain" else [("p1.0", 1.0), ("p0.5", 0.5), ("p0.0", 0.0)]
        for phase, prob in phases:
            cands = {}
            if kind == "plain":
                cands["torch_default(T-BF16 ref)"] = lambda x: base(ctx, x, rope, None, H)
                cands["r4_sdpa_rope"] = lambda x: sdpa_rope(x.reshape(B, N, 3, H, 64), rope, None)
                cands["TB1_attn(triton rope+flash)"] = lambda x: tb1(ctx, x, rope, None, H)
                cands["fused"] = lambda x: fused(ctx, x, rope, None, H)
                cands["fused_norope"] = lambda x: fused(ctx, x, None, None, H)
            else:
                g = torch.Generator(device="cuda").manual_seed(1)
                logits = torch.randn(B, 100, 128, 128, device="cuda", generator=g)
                ext = attn_ext.load_ext()
                rnd = torch.rand(B, 100, device="cuda", generator=g)
                pr = torch.tensor(prob, device="cuda")
                bits = ext.build_bits(logits, rnd, pr)  # p>=1 -> no row is re-enabled (rand < 1)
                qb = QueryBitmask(bits, prefix, 100)
                dense = qb.to_dense(N)  # source layout [B,N,N]
                dense4 = dense[:, None]  # round1 broadcast layout [B,1,N,N]
                cands["torch_default(T-BF16 ref)"] = lambda x: base(ctx, x, rope, dense, H)
                cands["r4_sdpa_rope"] = lambda x: sdpa_rope(x.reshape(B, N, 3, H, 64), rope, dense)
                if prob == 0.0:  # round1 mask_skip: mask dropped at host probability 0
                    cands["TB1_attn(mask_skip->flash)"] = lambda x: tb1(ctx, x, rope, None, H)
                    cands["fused(mask dropped)"] = lambda x: fused(ctx, x, rope, None, H)
                    cands["fused_devprob(bits, all rows on)"] = lambda x: fused(ctx, x, rope, qb, H)
                else:
                    cands["TB1_attn(qsplit flash+cudnn)"] = lambda x: tb1(ctx, x, rope, dense4, H)
                    cands["fused"] = lambda x: fused(ctx, x, rope, qb, H)
            res = measure(cands, qkv, dy, fl_f, fl_b, iters)
            for name, r in res.items():
                rec = dict(model=model, B=B, H=H, N=N, kind=kind, phase=phase, group="attn_core", variant=name, **r)
                results.append(rec)
                if "error" in r:
                    print(f"{model} {kind:5s} {phase:5s} {name:34s} ERROR {r['error'][:80]}", flush=True)
                else:
                    print(f"{model} {kind:5s} {phase:5s} {name:34s} fwd {r['fwd_us']:7.1f} bwd {r['bwd_us']:7.1f} f+b {r['fwd_bwd_us']:7.1f} us | {r['fwd_bwd_tflops']:5.1f} TF", flush=True)
        del qkv, dy
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ core (rope excluded) third-party candidates
def core_views(qkv, B, N, H):
    q, k, v = (t.transpose(1, 2) for t in qkv.view(B, N, 3, H, 64).unbind(2))  # [B,H,N,64] strided views
    return q, k, v


def load_fa4():
    root = "/root/autodl-tmp/train-opt-tmp/third_party/flash-attention"
    pkg = types.ModuleType("flash_attn")
    pkg.__path__ = [root + "/flash_attn"]
    sys.modules["flash_attn"] = pkg
    from flash_attn.cute.interface import flash_attn_func

    return flash_attn_func


class CudnnDsl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, B, N, H):
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120

        q, k, v = core_views(qkv, B, N, H)
        r = sdpa_fwd_wrapper_dsl_sm120(q, k, v)
        o, lse = r["o_tensor"], r["lse_tensor"]
        ctx.save_for_backward(qkv, o, lse)
        ctx.dims = (B, N, H)
        return o.transpose(1, 2).reshape(B, N, H * 64)

    @staticmethod
    def backward(ctx, dy):
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_dsl_sm120

        qkv, o, lse = ctx.saved_tensors
        B, N, H = ctx.dims
        q, k, v = core_views(qkv, B, N, H)
        do = dy.view(B, N, H, 64).transpose(1, 2)
        g = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, lse.unsqueeze(-1))
        dqkv = torch.stack((g["dq_tensor"].transpose(1, 2), g["dk_tensor"].transpose(1, 2), g["dv_tensor"].transpose(1, 2)), 2)
        return dqkv.reshape(B, N, 3 * H * 64), None, None, None


class SparkAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, B, N, H, spark, copy_layout):
        q, k, v = (t.contiguous() for t in core_views(qkv, B, N, H))  # [B,H,N,64] contiguous: the layout copies a packed-qkv caller pays
        o, lse = spark.attention_fwd(q, k, v, False, -1)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.spark = spark
        return o.transpose(1, 2).reshape(B, N, H * 64)

    @staticmethod
    def backward(ctx, dy):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, N, _ = q.shape
        do = dy.view(B, N, H, 64).transpose(1, 2).contiguous()
        dq, dk, dv = ctx.spark.attention_bwd(q, k, v, o, do, lse, False, -1, False)
        dqkv = torch.stack((dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)), 2)
        return dqkv.reshape(B, N, 3 * H * 64), None, None, None, None, None


def core_group(model, B, H, iters, results, which):
    ctx = DummyCtx()
    fa4 = None
    spark = None
    if "fa4" in which:
        try:
            fa4 = load_fa4()
        except Exception as e:
            print("FA4 import failed:", str(e)[:200])
            results.append(dict(model=model, group="core", variant="fa4", error=str(e)[:200]))
    if "spark" in which:
        from spark_ext import load_spark

        spark = load_spark()
    fused = make_fused()
    for kind, N in (("plain", 1029), ("joint", 1129)):
        qkv, dy, _ = make_inputs(B, H, N, kind == "joint")
        fl_f, fl_b = bl.flops_fwd(B, H, N), bl.flops_bwd(B, H, N)

        def sdpa(backend):
            def f(x):
                q, k, v = core_views(x, B, N, H)
                with sdpa_kernel(backend):
                    o = F.scaled_dot_product_attention(q, k, v)
                return o.transpose(1, 2).reshape(B, N, H * 64)
            return f

        cands = {
            "torch_flash": sdpa(SDPBackend.FLASH_ATTENTION),
            "torch_cudnn": sdpa(SDPBackend.CUDNN_ATTENTION),
            "fused_norope(no mask)": lambda x: fused(ctx, x, None, None, H),
            "cudnn_dsl_sm120": lambda x: CudnnDsl.apply(x, B, N, H),
        }
        if fa4 is not None:
            def fa4f(x):
                q, k, v = (t for t in x.view(B, N, 3, H, 64).unbind(2))
                o = fa4(q, k, v)
                o = o[0] if isinstance(o, tuple) else o
                return o.reshape(B, N, H * 64)
            cands["fa4_cute_sm120"] = fa4f
        if spark is not None:
            cands["spark(var5/var3)+layout copies"] = lambda x: SparkAttn.apply(x, B, N, H, spark, True)
        res = measure(cands, qkv, dy, fl_f, fl_b, iters)
        for name, r in res.items():
            rec = dict(model=model, B=B, H=H, N=N, kind=kind, phase="nomask", group="core", variant=name, **r)
            results.append(rec)
            if "error" in r:
                print(f"{model} {kind:5s} nomask {name:34s} ERROR {r['error'][:100]}", flush=True)
            else:
                print(f"{model} {kind:5s} nomask {name:34s} fwd {r['fwd_us']:7.1f} bwd {r['bwd_us']:7.1f} f+b {r['fwd_bwd_us']:7.1f} us | {r['fwd_bwd_tflops']:5.1f} TF", flush=True)
        del qkv, dy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vitl16")
    p.add_argument("--groups", default="attn_core,core")
    p.add_argument("--core", default="fa4,spark")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bl.require_lock()
    bl.warm_gpu(2.0)
    B, H = bl.SHAPES[a.model]
    results = []
    if "attn_core" in a.groups:
        attn_core_group(a.model, B, H, a.iters, results)
    if "core" in a.groups:
        core_group(a.model, B, H, a.iters, results, a.core)
    a.output.write_text(json.dumps({"results": results, "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
