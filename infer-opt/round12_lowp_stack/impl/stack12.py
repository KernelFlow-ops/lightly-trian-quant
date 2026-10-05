"""Round 12 integrated low-precision stack  (model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits)).

  r12 = round-11 bf16-equivalent items (patch-embed GEMM, dw_ln, tail, post_a2)
      + FP16 backbone: LN -> fp16, fp16-accumulate GEMMs with fp32 promotion (round-10 mg_f16acc kernel, R5 epilogues),
        FP16-accumulate attention (round-11 attn16 kernel, kind 4; layers 0-1 with a 1/8 Q pre-scale = the gated "a16pol" policy),
        fp16 handed from producer to consumer with no cast kernels (qkv epilogue -> attention -> proj; LN -> qkv / fc1; fc1 -> fc2).
  The residual stream, LN statistics, RoPE, softmax statistics and LayerScale stay fp32 (PLAN_v2 sec 2.2).

Variant string "r12[+flag...]":
  nopatch   keep the cuDNN patch-embed of r5_v1 (round 11 found ViT-L's fragile frame reacts to the ulp-level change)
  head16    head ConvT GEMMs on the fp16-accumulate kernel (bf16 output, so dw_ln is unchanged)
  pkt1      tile table that promotes to fp32 every k-tile (tuned16_pkt1.json) -- set through env MG16_TUNED before import
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
import mg16  # noqa: E402  (round-12 copy of the round-10 prototype wrappers, own build dir)

F16, BF = torch.float16, torch.bfloat16
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round12_lowp_stack")
_EXT: dict = {}


def _load(name, src, flags=()):
    if name not in _EXT:
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        from torch.utils.cpp_extension import load

        tag = hashlib.md5((src.read_bytes() + repr(flags).encode())).hexdigest()[:8]
        build = TMP / f"build_{name}_{tag}"
        build.mkdir(parents=True, exist_ok=True)
        _EXT[name] = load(name=f"r12_{name}_{tag}", sources=[str(src)], build_directory=str(build),
                          extra_cuda_cflags=["-O3", "-arch=sm_120a", "--expt-relaxed-constexpr", "-std=c++20"] + list(flags),
                          extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _EXT[name]


def ln_ext():
    return _load("ln16", _HERE / "ln16.cu")


def attn_ext():
    return _load("attn16o", _HERE / "attn16o.cu", ("-DCW=8", "-DSTAGES=4", "-DMINB=1"))


@torch.no_grad()
def prepare_block_weights16(block) -> dict:
    """r5 weights + fp16 copies of the four GEMM weights (same values as the bf16 ones: what the round-10 gates were run with)."""
    w = r5.prepare_block_weights(block, ln_fold=False)
    for k in ("Wqkv", "Wproj", "W1", "W2"):
        w[k + "16"] = w[k].float().to(F16).contiguous()
    return w


def block16(x, w, sin, cos, prefix, heads, attn_pre):
    """7 kernels, fp16 between them: LN1 | qkv+RoPE | attention | proj+residual | LN2 | fc1+GELU | fc2+residual."""
    ln, at = ln_ext(), attn_ext()
    h = ln.ln16(x, w["ln1_w"], w["ln1_b"], w["ln1_eps"])
    qkv = mg16.qkv_rope16(h, w["Wqkv16"], w["bqkv"], sin, cos, prefix, heads, out_dtype=F16)
    o = at.attn16_fwd(qkv.transpose(2, 3), 4, attn_pre)                      # physical [3,B,N,H,64] fp16 -> [B,N,d] fp16
    x = mg16.gemm_bias_residual16(o, w["Wproj16"], w["bproj"], x, w["g1"])
    h = ln.ln16(x, w["ln2_w"], w["ln2_b"], w["ln2_eps"])
    a = mg16.gemm_bias_gelu16(h, w["W116"], w["b1"], out_dtype=F16)
    return mg16.gemm_bias_residual16(a, w["W216"], w["b2"], x, w["g2"])


class FastHead12(head11.FastHead11):
    """round-11 head with the two ConvT GEMMs on the fp16-accumulate kernel (bf16 output -> dw_ln unchanged)."""

    def __init__(self, model):
        super().__init__(model)
        for st in self.stages:
            st["wr16"] = st["wr"].float().t().contiguous().to(F16)      # [4C, C]: mg16 computes A @ W^T

    def upscale_nhwc(self, x_tok, B, Hs, Ws):
        z = x_tok
        for i, st in enumerate(self.stages):
            g = mg16.gemm_plain16(z.to(F16), st["wr16"], out_dtype=BF, kind="ct1" if i == 0 else "ct2")
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=BF, device=x_tok.device)
            self.ext.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out


def make_model_fn_r12(name, model, flags):
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    W = [prepare_block_weights16(b) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks
    pe = None if "nopatch" in flags else pe_mod.PatchEmbedGemm(bb, **r11_model.PCFG[0])
    fh = FastHead12(model) if "head16" in flags else head11.FastHead11(model)

    def model_fn(x):
        if pe is not None:
            t = pe(x)
        else:
            with torch.autocast("cuda", dtype=BF, cache_enabled=False):
                t, _ = bb.prepare_tokens_with_masks(x)
            t = t.float().contiguous()
        for i in range(nb):
            if i == first_joint:
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = block16(t, W[i], sin, cos, t.shape[1] - 1024, heads, 0.125 if i <= 1 else 1.0)
        return fh.predict_tail(t, grid)

    model_fn._keepalive = (W, pe, fh)
    return model_fn


def make_variant(name, model, variant):
    parts = variant.split("+")
    assert parts[0] == "r12", variant
    flags = set(parts[1:])
    assert flags <= {"nopatch", "head16"}, flags
    return make_model_fn_r12(name, model, flags), r11_model.Pipeline11
