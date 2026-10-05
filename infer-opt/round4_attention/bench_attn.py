"""Microbench: TileLang attention vs SDPA backends on the real model shapes (GPU time, CUDA-graph replay).

Compared per (H, N):
  sdpa-{flash,efficient,cudnn}   on strided q/k/v views of the packed qkv (what the model feeds SDPA), output as [B,N,H*D]
  sdpa-*+rope4                   the R1 RoPE (roll/mul/addcmul/copy in place) followed by SDPA  == what fused rope replaces
  tl-plain_qkv / tl-rope_qkv     TileLang kernels on the packed layout with tuned configs
  tl-plain / tl-rope             TileLang on [B,H,N,D] contiguous inputs (interface of the brief)
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE))
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from bench import gpu_lock, write_results, env_info
from tune_attn import gpu_time_us
import ops

import ops as _ops
TUNED = {}

def rope4(qkv, sin_s, cos, hd=64):
    B, N = qkv.shape[:2]; hw = sin_s.shape[0]
    qk = qkv[:, N - hw:, :2]
    xr = torch.roll(qk, hd // 2, -1)
    qk.copy_(torch.addcmul(qk * cos, xr, sin_s))

def main():
    B = 2
    results = {}
    for H in (16, 12):
        for N in (1029, 1129):
            torch.manual_seed(0)
            qkv = torch.randn(B, N, 3, H, 64, device="cuda", dtype=torch.bfloat16) * 2
            sin = torch.randn(1024, 64, device="cuda"); sin[:, 32:] = sin[:, :32]
            cos = torch.randn(1024, 64, device="cuda"); cos[:, 32:] = cos[:, :32]
            sign = torch.ones(64, device="cuda"); sign[:32] = -1
            sin_s = (sin * sign)[:, None, None, :].contiguous(); cos_b = cos[:, None, None, :].contiguous()
            prefix = N - 1024
            q, k, v = qkv.unbind(2)
            qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            qc, kc, vc = [t.contiguous() for t in (qb, kb, vb)]
            flops = 4 * B * H * N * N * 64
            row = {}
            def add(name, fn):
                with gpu_lock():
                    us = gpu_time_us(fn)
                row[name] = {"us": us, "tflops": flops / us / 1e6}
                print(f"H{H} N{N} {name:28s} {us:7.1f} us  {flops / us / 1e6:6.1f} TFLOPS", flush=True)
            for nm, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                def f(be=be):
                    with sdpa_kernel(be):
                        return F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2).reshape(B, N, H * 64)
                def fr(be=be):
                    rope4(qkv, sin_s, cos_b)
                    return f()
                add(f"sdpa-{nm}", f)
                add(f"sdpa-{nm}+rope4", fr)
            for kind in ("plain_qkv", "rope_qkv", "plain", "rope"):
                cfg = None  # ops picks the tuned config
                if kind == "plain_qkv": fn = lambda cfg=cfg: ops.attention_qkv(qkv, cfg)
                elif kind == "rope_qkv": fn = lambda cfg=cfg: ops.attention_rope_qkv(qkv, sin, cos, prefix, cfg)
                elif kind == "plain": fn = lambda cfg=cfg: ops.attention(qc, kc, vc, cfg)
                else: fn = lambda cfg=cfg: ops.attention_rope(qc, kc, vc, sin, cos, prefix, cfg)
                add(f"tl-{kind}", fn); row[f"tl-{kind}"]["cfg"] = ops._cfg(kind if kind.endswith("qkv") else kind + "_qkv", N, H, None)
            results[f"H{H}_N{N}"] = row
    write_results(HERE / "results.json", {"microbench_attention": results, "env": env_info()})

if __name__ == "__main__":
    main()
