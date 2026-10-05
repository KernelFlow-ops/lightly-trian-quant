"""Round 5 model_fn implementations for the Round-9 pipeline (model_fn(static bf16 [2,3,512,512]) -> (mask_logits_lr, class_logits)).

  r3     : round3 fused_block (reference: what round9's `fused` variant runs), for a same-session A/B
  r5_v1  : round5 tuned GEMM tiles + K1 LayerNorm kernels          (7 kernels / block)
  r5_v2  : round5 LayerNorm folded into the GEMMs (+ stream-K/split-K, optional PDL)  (5 kernels / block)
All capture-safe (no host sync, static shapes); fp32 residual stream; fused Round-7 head (hd.enable)."""
import sys
from pathlib import Path

import torch

ROOT = Path("/root/lightly-train/infer-opt")
for p in (ROOT, ROOT / "round7_head_postprocess" / "impl", ROOT / "common", ROOT / "round5_gemm_sm120"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import head as hd  # noqa: E402
from round5_gemm_sm120.impl import ops as r5  # noqa: E402
from r3load import load_r3  # noqa: E402

IMPLS = ("r3", "r5_v1", "r5_v2", "r5_v2_nopdl")


def make_model_fn_r5(name: str, model, impl: str = "r5_v2"):
    hd.enable(model)
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    if impl == "r3":
        ops = load_r3()
        W = [ops.prepare_block_weights(b) for b in bb.blocks]
    else:
        ops = r5
        W = [r5.prepare_block_weights(b) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks

    def model_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            t, _ = bb.prepare_tokens_with_masks(x)
        t = t.float().contiguous()
        if impl in ("r3", "r5_v1"):
            for i in range(nb):
                if i == first_joint:
                    t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
                t = ops.fused_block(t, W[i], sin, cos, t.shape[1] - 1024, heads, False)
        else:
            pdl = impl != "r5_v2_nopdl"
            st = r5.v2_enter(t)
            for i in range(nb):
                if i == first_joint:
                    st = r5.v2_enter(st.x, queries)
                st = r5.fused_block_v2(st, W[i], sin, cos, st.x.shape[1] - 1024, heads, pdl=pdl)
            t = st.x
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            ml, cl = model._predict(bb.norm(t), grid_size=grid)
        return ml, cl

    model_fn._keepalive = (W,)
    return model_fn
