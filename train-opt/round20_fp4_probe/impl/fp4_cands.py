"""FP4 GEMM candidate wrappers (all TN: ``out[M,N] = A[M,K] . B[N,K]^T``, both operands K-major, blocked scales, per-tensor scales device-side).

  quack_fp4(a, b, out)      QuACK 0.6.5 SM120 CuTe-DSL GEMM, ``mma.sync kind::mxf4nvf4.block_scale`` (fp32 or bf16 out, tuned config cached on disk)
  scaled_mm_fp4(a, b, out_dtype)  cuBLASLt through ``torch.nn.functional.scaled_mm`` (NVFP4: BlockWise1x16 + TensorWise; MXFP4: BlockWise1x32)
  spark_fp4(a, b, out)      Spark ``fp4gemm`` (bf16 out only, K % 256, variant 2 = TMA needs M, N % 128)
  quack_fp4_epi(...)        QuACK with the round13 custom epilogues (bias / res / gelu / dgelu); NVFP4's per-tensor scale folded in as ``alpha``-like epilogue arg

``a`` / ``b`` are ``(q [R,K/2] u8, sf [rm,rk,32,4,4] u8, ts [1] fp32 or None)`` triples, ``fmt`` in {"nv4", "mx4"}.
"""
from __future__ import annotations

import os

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

import torch
import torch.nn.functional as F

_QF = {}


def qfmt(fmt):
    from quack.blockscaled import MXFP4, NVFP4

    return NVFP4 if fmt == "nv4" else MXFP4


def operand(trip, fmt):
    """QuACK ``BlockScaledOperand`` [R, K] (quant_dim -1) from a (q, sf, ts) triple."""
    from quack.blockscaled import BlockScaledOperand

    q, sf, ts = trip
    f = qfmt(fmt)
    return BlockScaledOperand.from_parts(q.view(torch.float4_e2m1fn_x2), sf.view(f.scale_dtype), f, per_tensor_scale=ts if fmt == "nv4" else None, quant_dim=-1)


def quack_fp4(a_op, b_op, out):
    """``out = A . B^T``; a_op [M,K], b_op [N,K] containers (the .mT view is taken here)."""
    from quack.gemm_interface import gemm

    gemm(a_op, b_op.mT, out=out)
    return out


def scaled_mm_fp4(a, b, fmt, out_dtype=torch.bfloat16):
    qa, sfa, tsa = a
    qb, sfb, tsb = b
    fa, fb = qa.view(torch.float4_e2m1fn_x2), qb.view(torch.float4_e2m1fn_x2)
    if fmt == "nv4":
        ea, eb = sfa.view(torch.float8_e4m3fn), sfb.view(torch.float8_e4m3fn)
        return F.scaled_mm(fa, fb.t(), [ea, tsa.reshape(1)], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           [eb, tsb.reshape(1)], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                           swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE], output_dtype=out_dtype)
    ea, eb = sfa.view(torch.float8_e8m0fnu), sfb.view(torch.float8_e8m0fnu)
    return F.scaled_mm(fa, fb.t(), ea, F.ScalingType.BlockWise1x32, eb, F.ScalingType.BlockWise1x32,
                       swizzle_a=F.SwizzleType.SWIZZLE_32_4_4, swizzle_b=F.SwizzleType.SWIZZLE_32_4_4, output_dtype=out_dtype)


def spark_fp4(ext, a, b, out, fmt, variant=-1):
    qa, sfa, tsa = a
    qb, sfb, tsb = b
    ext.fp4gemm(qa, qb, out, sfa, sfb, tsa if fmt == "nv4" else None, tsb if fmt == "nv4" else None, 0 if fmt == "nv4" else 1, variant)
    return out


def epilogues():
    """round13 epilogue frontends (bias | res | gelu | dgelu); same objects the MXFP8 path uses."""
    from round13_gemm_fusion_sm120.impl import gemm_ops as g13

    return g13._quack()


def epi_kw(a_sf, b_sf, fmt):
    f = qfmt(fmt).name
    return {"SFA": a_sf.view(qfmt(fmt).scale_dtype), "SFB": b_sf.view(qfmt(fmt).scale_dtype), "bs_format_a": f, "bs_format_b": f}
