"""Round 11 model_fn variants stacked on round-5 r5_v1  (model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits)).

Variant string: "r11[+flag...]"; default flags (no '+'): all bf16-equivalent items.
  patch   patch-embed as implicit-im2col GEMM writing fp32 x0 (impl/patch_embed.cu)       replaces cuDNN conv + cat + casts
  head    round-11 dw_ln kernel (impl/head_ext11.cu)                                         replaces round-7 dw_ln
  tail    fused bb.norm + cast + class_head + pre-cast mask_head weights                   replaces autocast _predict glue
  post    post_a2 kernel (impl/postfuse_ext11.cu), via a VideoPipeline subclass            replaces round-7 post_a
  skip<eps>  query skipping in post_a2 with threshold eps (e.g. skip1e-3); implies post
  attn=<name>  attention implementation swap (Part 2): flash (default) | tl | ...
  none    nothing (r5_v1 re-implemented, for A/A checks)
Examples: "r11" = patch+head+tail+post ; "r11+patch" = only patch ; "r11+patch+head+tail+skip1e-3".
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import r11_common as c
import head11
import patch_embed as pe_mod
import postfuse_ext11_py as p11
from round5_gemm_sm120.impl import ops as r5
import head as hd7

ALL = ("patch", "head", "tail", "post")
PCFG = [dict(), dict(BM=64, BN=64, BK=64, ST=3, WM=2, WN=2, MINB=2), dict(BM=64, BN=128, BK=32, ST=3, WM=2, WN=2, MINB=2),
        dict(BM=128, BN=128, BK=32, ST=3, WM=2, WN=4, MINB=2)]


def parse(variant: str):
    parts = variant.split("+")[1:]
    flags = {"attn": "flash", "eps": 0.0, "pcfg": 0}
    on = set()
    items_given = False
    for p in parts:
        if p.startswith("attn="):
            flags["attn"] = p[5:]
        elif p.startswith("jit"):
            flags["jit"] = int(p[3:])       # sensitivity probe: ulp-level jitter of ~0.016% of the x0 elements (seeded)
        elif p.startswith("pcfg"):
            flags["pcfg"] = int(p[4:])      # alternative patch-embed tile configs = different fp32 accumulation order (sensitivity probes)
        elif p.startswith("skip"):
            flags["eps"] = float(p[4:])
            on.add("post")
        elif p == "none":
            items_given = True
        else:
            assert p in ALL, p
            on.add(p)
            items_given = True
    if not items_given:
        on |= set(ALL)
    return on, flags


def attn_flash(qkv_phys, B, N, d, layer=None):
    """qkv_phys: physical [3,B,N,H,64] -> o [B,N,d]  (torch flash SDPA, the r5_v1 path)."""
    q = qkv_phys.transpose(2, 3)                 # logical [3,B,H,N,64] view
    o = F.scaled_dot_product_attention(q[0], q[1], q[2])
    return o.transpose(1, 2).reshape(B, N, d)


ATTN_IMPLS = {"flash": attn_flash}


def block(x, w, sin, cos, prefix, heads, attn_fn, layer=None):
    """r5 fused_block with a pluggable attention (same 7 kernels for attn=flash)."""
    B, N, d = x.shape
    _, h = r5.add_layernorm(x, None, w["ln1_w"], w["ln1_b"], w["ln1_eps"])
    qkv = r5.qkv_rope(h, w["Wqkv"], w["bqkv"], sin, cos, prefix, heads)
    o = attn_fn(qkv.transpose(2, 3), B, N, d, layer=layer)
    x = r5.gemm_bias_residual(o, w["Wproj"], w["bproj"], x, w["g1"])
    _, h = r5.add_layernorm(x, None, w["ln2_w"], w["ln2_b"], w["ln2_eps"])
    a = r5.gemm_bias_gelu(h, w["W1"], w["b1"])
    return r5.gemm_bias_residual(a, w["W2"], w["b2"], x, w["g2"])


def make_model_fn_r11(name, model, on, flags):
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    W = [r5.prepare_block_weights(b) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks
    use_patch, use_tail = "patch" in on, "tail" in on
    pe = pe_mod.PatchEmbedGemm(bb, **PCFG[flags.get("pcfg", 0)]) if use_patch else None
    if use_tail or "head" in on:
        fh = head11.FastHead11(model)
    else:
        fh = None
    if not (use_tail or "head" in on):
        hd7.enable(model)
    elif not use_tail:
        # head kernel swap only: keep the round-7 predict glue but with the new dw_ln kernel
        orig = hd7.FastHead(model)

        def predict(x, grid_size):
            m = model
            q = x[:, : m.num_queries, :]
            class_logits = m.class_head(q)
            xt = x[:, m.num_queries + 1 + bb.n_storage_tokens:, :]
            Bx, N, C = xt.shape
            xt = xt.to(torch.bfloat16).reshape(Bx * N, C)
            u = fh.upscale_nhwc(xt, Bx, grid_size[0], grid_size[1])
            mq = m.mask_head(q).to(torch.bfloat16)
            ml = torch.bmm(mq, u.view(Bx, -1, C).transpose(1, 2))
            return ml.view(Bx, mq.shape[1], u.shape[1], u.shape[2]), class_logits
        model._predict = predict
    jm = None
    if flags.get("jit"):
        g = torch.Generator(device="cpu").manual_seed(1000 + flags["jit"])
        r = torch.rand(2, 1 + bb.n_storage_tokens + grid[0] * grid[1], bb.embed_dim, generator=g)
        sgn = torch.where(torch.rand_like(r) < 0.5, -1.0, 1.0)
        jm = (1.0 + (r < 1.7e-4).float() * sgn * 2.0 ** -8).cuda()
    if callable(flags["attn"]):
        attn_fn = flags["attn"]
    elif flags["attn"] in ATTN_IMPLS:
        attn_fn = ATTN_IMPLS[flags["attn"]]
    else:
        import attn_cands
        attn_fn = attn_cands.get_full(flags["attn"])

    def model_fn(x):
        if use_patch:
            t = pe(x)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                t, _ = bb.prepare_tokens_with_masks(x)
            t = t.float().contiguous()
        if jm is not None:
            t = t * jm
        for i in range(nb):
            if i == first_joint:
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = block(t, W[i], sin, cos, t.shape[1] - 1024, heads, attn_fn, i)
        if use_tail:
            return fh.predict_tail(t, grid)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            ml, cl = model._predict(bb.norm(t), grid_size=grid)
        return ml, cl

    model_fn._keepalive = (W, pe, fh)
    return model_fn


class Pipeline11(c.vi.VideoPipeline):
    """VideoPipeline with the round-11 post_a2 kernel (2 crops, exact x4 upsample); falls back to round-7 post_a otherwise."""
    eps = 0.0
    use_post_a2 = True

    def _post_launch(self, s, ml, cl):
        h = ml.shape[2]
        if self.use_post_a2 and len(self.starts) == 2 and self.crop == 4 * h and ml.shape[3] == h and h % 32 == 0:
            p11.ext().post_a2_launch(ml.contiguous(), cl.contiguous(), self.lr, self.starts, self.crop, float(self.eps))
        else:
            c.pf.ext().post_a_launch(ml.contiguous(), cl.contiguous(), self.lr, self.starts, self.crop)
        c.pf.ext().post_b_launch(self.lr, self.mask_g[s], self.stats_g[s], None, self.class_ids[0],
                                 self.class_ids[1], self.head_ch, c.vi.HEAD_CLASS_ID)


def make_variant(name, model, variant):
    on, flags = parse(variant)
    fn = make_model_fn_r11(name, model, on, flags)
    if "post" in on:
        pcls = type("Pipeline11_eps", (Pipeline11,), {"eps": flags["eps"]})
    else:
        pcls = c.vi.VideoPipeline
    return fn, pcls
