"""Round 13 whole-model MXFP8 stack  (model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits)).

  r13_mx8a = r12 (round-11 patch-embed GEMM, dw_ln, tail, post_a2, r12 fp16 attention kernel) with
             * all four backbone GEMM kinds of every layer on cuBLASLt MXFP8 (`torch._scaled_mm`, E4M3 + UE8M0 1x32, fp32 accumulate), bf16 output,
               weights quantised once at build time;
             * LN -> MX8 fused (`ln_mx8`), attention output -> MX8 (`quant_rows`), fc1 output -> GELU -> MX8 (`gelu_quant`);
             * the epilogues the cuBLASLt GEMM does not have as separate passes (`epi_rope`: bias + RoPE + fp16 [3,B,N,H,64] layout for the r12 attention,
               `epi_res`: bias + gamma + fp32 residual);
             * the two head ConvT GEMMs on the same MXFP8 path (input quantised by `quant_rows`, output bf16 -> dw_ln unchanged).
  Variant strings:  "r13_mx8a[+nohead]"     step 1 (cuBLASLt)         [+nohead keeps the bf16 head of r12]
                    "r13_mx8b"              step 1 + residual add fused into the next LayerNorm (res_ln_mx8)
                    "r13_mx8[+lt=qkv.fc2][+lthead][+nohead]"   step 2: own mma.sync block-scaled kernels (impl/mgx_core.cu) with fused epilogues; lt= keeps kinds on cuBLASLt
Accuracy gates are OFF for this round (user decision 2026-10-05): only finiteness and the mask-logit cosine vs r5_v1 are reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import r11_common as c            # noqa: F401  (sys.path for round5 / 7 / 9 / 11, read-only imports)
import head11
import patch_embed as pe_mod
import r11_model
from round5_gemm_sm120.impl import ops as r5
import stack12                    # round-12 attention extension / helpers (read-only import)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import mx13  # noqa: E402

F16, BF = torch.float16, torch.bfloat16


@torch.no_grad()
def prepare_block_weights8(block) -> dict:
    w = r5.prepare_block_weights(block, ln_fold=False)
    out = {k: w[k] for k in ("g1", "g2", "ln1_w", "ln1_b", "ln1_eps", "ln2_w", "ln2_b", "ln2_eps")}
    for k in ("bqkv", "bproj", "b1", "b2"):
        out[k] = w[k].float().contiguous()
    for k in ("Wqkv", "Wproj", "W1", "W2"):
        out[k] = mx13.MXW(w[k])
    return out


def block8a(x, w, sin, cos, prefix, heads, attn_pre):
    """12 kernels: ln_mx8 | qkv GEMM | epi_rope | attention | quant | proj GEMM | epi_res | ln_mx8 | fc1 GEMM | gelu_quant | fc2 GEMM | epi_res."""
    E, at = mx13.ext(), stack12.attn_ext()
    B, N, d = x.shape
    M = B * N
    a = mx13.act_buf(M, d)
    E.ln_mx8(x, w["ln1_w"], w["ln1_b"], a.q, a.sf, w["ln1_eps"])
    y = mx13.mm(a, w["Wqkv"])
    qkv = E.epi_rope(y, w["bqkv"], sin, cos, N, prefix, d, heads)
    o = at.attn16_fwd(qkv, 4, attn_pre)                       # [B,N,d] fp16
    E.quant_rows(o, a.q, a.sf)
    y = mx13.mm(a, w["Wproj"])
    x = E.epi_res(y, x, w["g1"], w["bproj"])
    E.ln_mx8(x, w["ln2_w"], w["ln2_b"], a.q, a.sf, w["ln2_eps"])
    y = mx13.mm(a, w["W1"])
    f = mx13.act_buf(M, w["W1"].N)
    E.gelu_quant(y, w["b1"], f.q, f.sf)
    y = mx13.mm(f, w["W2"])
    return E.epi_res(y, x, w["g2"], w["b2"])


def block8b(x, pend, w, sin, cos, prefix, heads, attn_pre):
    """block8a with the residual add of the PREVIOUS block's fc2 (pend = (y, gamma, bias) or None) fused into this block's LN1 (res_ln_mx8) and proj's residual
    fused into LN2 (res_ln_mx8): 10 kernels per block instead of 12.  Returns (x_after_proj_residual, pend of this block's fc2)."""
    E, at = mx13.ext(), stack12.attn_ext()
    B, N, d = x.shape
    M = B * N
    a = mx13.act_buf(M, d)
    if pend is None:
        E.ln_mx8(x, w["ln1_w"], w["ln1_b"], a.q, a.sf, w["ln1_eps"])
    else:
        x = E.res_ln_mx8(pend[0], x, pend[1], pend[2], w["ln1_w"], w["ln1_b"], a.q, a.sf, w["ln1_eps"])
    y = mx13.mm(a, w["Wqkv"])
    qkv = E.epi_rope(y, w["bqkv"], sin, cos, N, prefix, d, heads)
    o = at.attn16_fwd(qkv, 4, attn_pre)
    E.quant_rows(o, a.q, a.sf)
    y = mx13.mm(a, w["Wproj"])
    x = E.res_ln_mx8(y, x, w["g1"], w["bproj"], w["ln2_w"], w["ln2_b"], a.q, a.sf, w["ln2_eps"])
    y = mx13.mm(a, w["W1"])
    f = mx13.act_buf(M, w["W1"].N)
    E.gelu_quant(y, w["b1"], f.q, f.sf)
    y = mx13.mm(f, w["W2"])
    return x, (y, w["g2"], w["b2"])


def block8(x, w, sin, cos, prefix, heads, attn_pre, lt=frozenset()):
    """Step 2: 7 kernels per block on the own mgx GEMMs with fused epilogues:
       ln_mx8 | qkv GEMM (+bias+RoPE+fp16 layout) | attention | quant_rows | proj GEMM (+bias+gamma+fp32 residual) | ln_mx8 | fc1 GEMM (+bias+GELU+MX8 quantize-out) | fc2 GEMM (+bias+gamma+residual)
       `lt` = set of GEMM kinds that stay on cuBLASLt + standalone passes (per-kind best of step 1 / step 2)."""
    import mgx
    E, at = mx13.ext(), stack12.attn_ext()
    B, N, d = x.shape
    M = B * N
    a = mx13.act_buf(M, d)
    E.ln_mx8(x, w["ln1_w"], w["ln1_b"], a.q, a.sf, w["ln1_eps"])
    if "qkv" in lt:
        qkv = E.epi_rope(mx13.mm(a, w["Wqkv"]), w["bqkv"], sin, cos, N, prefix, d, heads)
    else:
        qkv = mgx.qkv_rope(a, w["Wqkv"], w["bqkv"], sin, cos, N, prefix, heads)
    o = at.attn16_fwd(qkv, 4, attn_pre)
    E.quant_rows(o, a.q, a.sf)
    if "proj" in lt:
        x = E.epi_res(mx13.mm(a, w["Wproj"]), x, w["g1"], w["bproj"])
    else:
        x = mgx.gemm_res(a, w["Wproj"], w["bproj"], w["g1"], x, kind="proj")
    E.ln_mx8(x, w["ln2_w"], w["ln2_b"], a.q, a.sf, w["ln2_eps"])
    f = mx13.act_buf(M, w["W1"].N)
    if "fc1" in lt:
        E.gelu_quant(mx13.mm(a, w["W1"]), w["b1"], f.q, f.sf)
    else:
        mgx.gemm_gelu_q(a, w["W1"], w["b1"], f)
    if "fc2" in lt:
        return E.epi_res(mx13.mm(f, w["W2"]), x, w["g2"], w["b2"])
    return mgx.gemm_res(f, w["W2"], w["b2"], w["g2"], x, kind="fc2")


class FastHead13(head11.FastHead11):
    """round-11 head with the two ConvT GEMMs on MXFP8 (bf16 output -> dw_ln unchanged)."""

    def __init__(self, model, own=False):
        super().__init__(model)
        self.own = own
        for st in self.stages:
            st["w8"] = mx13.MXW(st["wr"].t().contiguous())            # [4C, C] K-major

    def upscale_nhwc(self, x_tok, B, Hs, Ws):
        E = mx13.ext()
        z = x_tok
        for i, st in enumerate(self.stages):
            a = mx13.act_buf(z.shape[0], z.shape[1])
            E.quant_rows(z, a.q, a.sf)
            if self.own:
                import mgx
                g = mgx.gemm_plain(a, st["w8"], kind="ct1" if i == 0 else "ct2")
            else:
                g = mx13.mm(a, st["w8"])
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=BF, device=x_tok.device)
            self.ext.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out


def make_model_fn_r13(name, model, flags, block_fn, fused=False, own_head=False):
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    W = [prepare_block_weights8(b) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    sin, cos = sin.float().contiguous(), cos.float().contiguous()
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks
    pe = pe_mod.PatchEmbedGemm(bb, **r11_model.PCFG[0])
    fh = head11.FastHead11(model) if "nohead" in flags else FastHead13(model, own=own_head)

    def model_fn(x):
        t = pe(x)
        pend = None
        for i in range(nb):
            if i == first_joint:
                if pend is not None:
                    t, pend = mx13.ext().epi_res(pend[0], t, pend[1], pend[2]), None
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            if fused:
                t, pend = block_fn(t, pend, W[i], sin, cos, t.shape[1] - 1024, heads, 0.125 if i <= 1 else 1.0)
            else:
                t = block_fn(t, W[i], sin, cos, t.shape[1] - 1024, heads, 0.125 if i <= 1 else 1.0)
        if pend is not None:
            t = mx13.ext().epi_res(pend[0], t, pend[1], pend[2])
        return fh.predict_tail(t, grid)

    model_fn._keepalive = (W, pe, fh)
    return model_fn


def make_variant(name, model, variant):
    """r13_mx8a / r13_mx8b: step 1 (cuBLASLt);  r13_mx8: step 2 (own mgx kernels);  flags: nohead | lt=<kind.kind> (kinds that stay on cuBLASLt) | lthead (ConvT on cuBLASLt)"""
    parts = variant.split("+")
    flags = set(parts[1:])
    lt = frozenset(k for f in flags if f.startswith("lt=") for k in f[3:].split("."))
    flags = {f for f in flags if not f.startswith("lt=")}
    assert flags <= {"nohead", "lthead"}, flags
    assert lt <= {"qkv", "proj", "fc1", "fc2"}, lt
    if parts[0] == "r13_mx8a":
        return make_model_fn_r13(name, model, flags, block8a), r11_model.Pipeline11
    if parts[0] == "r13_mx8b":      # step 1 + residual-add fused into the next LayerNorm
        return make_model_fn_r13(name, model, flags, block8b, fused=True), r11_model.Pipeline11
    if parts[0] in ("r13_mx8", "r13_mx8best"):       # step 2: own block-scaled GEMMs with fused epilogues;  r13_mx8best = per-kind best: qkv stays on cuBLASLt + epi_rope
        if parts[0] == "r13_mx8best":
            lt = lt | {"qkv"}
        fn = lambda *a: block8(*a, lt=lt)  # noqa: E731
        return make_model_fn_r13(name, model, flags, fn, own_head="lthead" not in flags), r11_model.Pipeline11
    raise ValueError(variant)
