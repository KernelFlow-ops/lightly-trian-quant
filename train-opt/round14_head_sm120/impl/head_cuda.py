"""CUDA (sm_120) implementation of the three fused head-stage kernels (``head_kernels.cu``), same API as ``head_triton``.

JIT-built with torch.utils.cpp_extension into the round14 tmp build dir.  ``cpt`` = channels per thread (CTA = C/cpt threads),
``pw`` = pixels of a row per CTA, ``rpb`` = image rows per CTA (backward), ``minb`` = launch-bound CTAs/SM.
Defaults come from the stage-kernel sweep (tests/test_triton_stage.py --impl cuda --bench, NOTES section 4).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_HERE = Path(__file__).resolve().parent
_BUILD = Path("/root/autodl-tmp/train-opt-tmp/round14_head_sm120/build/head_kernels")
CFG = {
    "fwd_cpt": int(os.environ.get("R14_FWD_CPT", "8")), "fwd_pw": int(os.environ.get("R14_FWD_PW", "64")),
    "bwd1_cpt": int(os.environ.get("R14_BWD1_CPT", "4")), "bwd1_minb": int(os.environ.get("R14_BWD1_MINB", "2")),
    "bwd2_cpt": int(os.environ.get("R14_BWD2_CPT", "4")), "bwd2_minb": int(os.environ.get("R14_BWD2_MINB", "2")),
    "rpb": int(os.environ.get("R14_RPB", "1")),
}


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "8")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _BUILD.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="trainopt_r14_head_kernels",
            sources=[str(_HERE / "head_kernels.cu")],
            build_directory=str(_BUILD),
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "--use_fast_math", "-Xptxas", "-v"],
            verbose=bool(int(os.environ.get("R14_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def _pw(w2, pw):
    pw = pw or w2
    return pw if w2 % pw == 0 else w2


def fwd(Z, bias, dwt, gam, bet, B, h, w, eps, PW=None, cpt=None):
    """Z [B*h*w,4C] bf16; bias/gam/bet [C] fp32; dwt [9,C] fp32 -> (Y [B*H*W,C] bf16, mean [B*H*W], rstd [B*H*W])."""
    y, mean, rstd = extension().stage_fwd(Z, bias, dwt, gam, bet, B, h, w, float(eps), _pw(2 * w, PW or CFG["fwd_pw"]), cpt or CFG["fwd_cpt"])
    return y, mean, rstd


def bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, PW=None, cpt=None, rpb=None, minb=None):
    """-> (dU [B*H*W,C] bf16, dgamma [C] fp32, dbeta [C] fp32)."""
    rpb = rpb or CFG["rpb"]
    if (B * 2 * h) % rpb:
        rpb = 1
    du, part = extension().stage_bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, _pw(2 * w, PW), cpt or CFG["bwd1_cpt"], rpb, minb or CFG["bwd1_minb"])
    s = part.sum(0)
    return du, s[0], s[1]


def bwd2(Z, bias, dwt, du, B, h, w, PW=None, cpt=None, rpb=None, minb=None):
    """-> (dZ [B*h*w,4C] bf16, ddw [9,C] fp32, dbias [C] fp32)."""
    rpb = rpb or CFG["rpb"]
    if (B * 2 * h) % rpb:
        rpb = 1
    dz, pdw, pdb = extension().stage_bwd2(Z, bias, dwt, du, B, h, w, _pw(2 * w, PW), cpt or CFG["bwd2_cpt"], rpb, minb or CFG["bwd2_minb"])
    return dz, pdw.sum(0), pdb.sum(0)
