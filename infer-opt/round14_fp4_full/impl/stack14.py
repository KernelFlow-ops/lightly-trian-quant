"""Round 14 whole-model FP4 stack on top of r12 (model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits)).

Step 1 ("a" variants, assembled from existing pieces):
  r14_nv4a / r14_mx4a = r12 with the four backbone GEMM kinds on every layer and both head ConvT GEMMs in NVFP4 / MXFP4.
    LN (fp32 -> bf16, ln14.cu) -> fp4q2 quantiser (static per-tensor scale for nv4) -> QuACK block-scaled GEMM with a ts-aware fused epilogue
    qkv : QuACK bias epilogue (bf16) -> rope14.cu (RoPE + [3,B,N,H,64] layout, fp16 out)
    attn: round-11 attn16 kernel (kind 4, fp16 in, bf16 out; layers 0-1 with the 1/8 Q pre-scale like r12) -> fp4q2 quantiser
    proj / fc2: QuACK gamma + fp32 residual epilogue ; fc1: QuACK bias + GELU epilogue (bf16) -> fp4q2 quantiser
  Weights are quantised once (1D scales along K, exact per-tensor scale); the activation per-tensor scale of nv4 is static: measured on the first
  (eager) call of the model_fn (headroom 2), never recomputed (accuracy is not gated in round 14).
Everything else (patch-embed GEMM, residual stream fp32, LN statistics, RoPE fp32, attention, head dw_ln / mask head / einsum, post) is r12 / r11.

Variant string "r14_<fmt><impl>[+flag...]": fmt in {nv4, mx4}; impl "a" = step 1.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch

import r11_common as c            # sys.path for round5 / 7 / 9 / 11 (read-only imports)
import head11
import patch_embed as pe_mod
import r11_model
from round5_gemm_sm120.impl import ops as r5

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fp4util as U  # noqa: E402

R11_IMPL = Path("/root/lightly-train/infer-opt/round11_attn_tier0/impl")
TMP = U.TMP
_EXT: dict = {}


def _load(name, src, flags=()):
    if name not in _EXT:
        from torch.utils.cpp_extension import load

        tag = hashlib.md5((Path(src).read_bytes() + repr(flags).encode())).hexdigest()[:8]
        build = TMP / "build" / f"{name}_{tag}"
        build.mkdir(parents=True, exist_ok=True)
        _EXT[name] = load(name=f"r14_{name}_{tag}", sources=[str(src)], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a", "--expt-relaxed-constexpr", "-std=c++20"] + list(flags),
                          extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _EXT[name]


def ln_ext():
    return _load("ln14", _HERE / "ln14.cu")


def rope_ext():
    return _load("rope14", _HERE / "rope14.cu")


def attn_ext():
    """round-11 attn16 kernel renamed to namespace a14 (fp16 in, bf16 out)"""
    return _load("attn14", _HERE / "attn14.cu", ("-DCW=8", "-DSTAGES=4", "-DMINB=1"))


class Layer14:
    def __init__(self, block, fmt, calib):
        w = r5.prepare_block_weights(block, ln_fold=False)
        self.ln1 = (w["ln1_w"], w["ln1_b"], w["ln1_eps"])
        self.ln2 = (w["ln2_w"], w["ln2_b"], w["ln2_eps"])
        self.qkv = U.FP4Linear(w["Wqkv"], w["bqkv"], fmt, calib)
        self.proj = U.FP4Linear(w["Wproj"], w["bproj"], fmt, calib, gamma=w["g1"])
        self.fc1 = U.FP4Linear(w["W1"], w["b1"], fmt, calib)
        self.fc2 = U.FP4Linear(w["W2"], w["b2"], fmt, calib, gamma=w["g2"])
        self.s_ln1, self.s_o, self.s_ln2, self.s_gelu = (U.Site(fmt, calib) for _ in range(4))


def block14a(x, L: Layer14, sin, cos, prefix, heads, attn_pre):
    """12 kernels: LN | quant | qkv GEMM | RoPE | attention | quant | proj+res | LN | quant | fc1+GELU | quant | fc2+res."""
    B, N, d = x.shape
    M = B * N
    ln, at, rp = ln_ext(), attn_ext(), rope_ext()
    h = ln.ln_bf16(x, *L.ln1[:2], L.ln1[2])
    y = L.qkv.fwd_bias(L.s_ln1.quant(h.view(M, d)))
    qkv = rp.rope14(y, sin, cos, N, prefix, d, heads)
    o = at.attn16_fwd(qkv, 4, attn_pre)                                   # [B,N,d] bf16
    x = L.proj.fwd_res(L.s_o.quant(o.view(M, d)), x.view(M, d)).view(B, N, d)
    h = ln.ln_bf16(x, *L.ln2[:2], L.ln2[2])
    a = L.fc1.fwd_gelu(L.s_ln2.quant(h.view(M, d)))
    return L.fc2.fwd_res(L.s_gelu.quant(a), x.view(M, d)).view(B, N, d)


class FastHead14(head11.FastHead11):
    """round-11 head with both ConvT GEMMs in FP4 (plain GEMM, bf16 out -> dw_ln unchanged)."""

    def __init__(self, model, fmt, calib):
        super().__init__(model)
        self.lin = [U.FP4Linear(st["wr"].float().t().contiguous(), None, fmt, calib) for st in self.stages]   # [4C, C]
        self.sites = [U.Site(fmt, calib) for _ in self.stages]

    def upscale_nhwc(self, x_tok, B, Hs, Ws):
        z = x_tok
        for i, st in enumerate(self.stages):
            g = self.lin[i].fwd_plain(self.sites[i].quant(z))
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=torch.bfloat16, device=x_tok.device)
            self.ext.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out


def make_model_fn_a(name, model, fmt, flags):
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    calib = U.Calib()
    U._CLEAR[0] = ln_ext().clear_cuda_error
    L = [Layer14(b, fmt, calib) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    sin, cos = sin.float().contiguous(), cos.float().contiguous()
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks
    pe = pe_mod.PatchEmbedGemm(bb, **r11_model.PCFG[0])
    fh = FastHead14(model, fmt, calib)

    def model_fn(x):
        if calib.on and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("r14: the first call (static-scale calibration) must run eagerly, before CUDA graph capture")
        t = pe(x)
        for i in range(nb):
            if i == first_joint:
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = block14a(t, L[i], sin, cos, t.shape[1] - 1024, heads, 0.125 if i <= 1 else 1.0)
        out = fh.predict_tail(t, grid)
        calib.on = False
        return out

    model_fn._keepalive = (L, pe, fh, calib)
    return model_fn


def make_variant(name, model, variant):
    parts = variant.split("+")
    base = parts[0]
    assert base.startswith("r14_"), variant
    tag = base[4:]
    fmt = tag[:3]
    impl = tag[3:]
    assert fmt in ("nv4", "mx4"), variant
    flags = set(parts[1:])
    if impl == "a":
        return make_model_fn_a(name, model, fmt, flags), r11_model.Pipeline11
    if impl == "":                                   # step 2: own kernel + fused producers / epilogues
        import stack14b
        return stack14b.make_model_fn_b(name, model, fmt, flags), r11_model.Pipeline11
    raise ValueError(variant)
