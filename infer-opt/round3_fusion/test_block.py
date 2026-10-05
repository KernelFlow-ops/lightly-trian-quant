"""Level-B: fused_block vs the original block math on REAL vitl-best.pth / vitb-best.pth weights.

Block inputs are captured from a real-frame fp32 forward (norm1 pre-hooks), so the activations are realistic
(incl. the joint layers with 100 query tokens prepended).  All variants are compared against the fp32 eager block.
   eager_bf16   : model.to(bf16) block, no autocast (what "bf16-pure" does)
   eager_ac     : fp32 weights + autocast bf16 (main baseline: fp32 residual stream)
   fused_bf16res: fused_block, bf16 residual stream
   fused_fp32res: fused_block, fp32 residual stream (autocast-like)
usage: python test_block.py [vitl16|vitb16] [layers comma list]
"""
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import copy
import json

import model_factory as mf
import torch
from bench import write_results
from check import tensor_diff

from impl import ops

name = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else ([0, 12, 21, 22, 23] if name == "vitl16" else [0, 9, 10])
model = mf.load_model(name, torch.float32)
model_bf = copy.deepcopy(model).to(torch.bfloat16)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
nb = len(bb.blocks)
frames = mf.load_frames()

cap = {}
hooks = []
for i in set(layers) | {l + 1 for l in layers if l + 1 < nb}:
    hooks.append(bb.blocks[i].norm1.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0].detach().clone())))
hooks.append(bb.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__(nb, a[0].detach().clone())))

out = {}
sin, cos = bb.rope_embed(H=32, W=32)
for fi in (0, 3):
    cap.clear()
    with torch.inference_mode():
        batch, meta = mf.base_prep(model, frames[fi])
        model.forward_backend(batch)
    for i in layers:
        x = cap[i]
        B, N, d = x.shape
        prefix = N - 1024
        blk, blk_bf = bb.blocks[i], model_bf.backbone.blocks[i]
        w = ops.prepare_block_weights(blk)
        wf = ops.prepare_block_weights(blk, fold_gamma=True)

        def eager(m, blk_, xin, rope):
            xin = xin.clone()
            blk_ = blk_
            h = blk_.norm1(xin)
            xin = xin + blk_.ls1(m._attn(blk_.attn, h, rope=rope, mask=None))
            return xin + blk_.ls2(blk_.mlp(blk_.norm2(xin)))

        with torch.inference_mode():
            ref = eager(model, blk, x, (sin, cos))  # fp32 eager (also equals captured next-block input)
            chain_ref = cap[i + 1]
            e_bf = eager(model_bf, blk_bf, x.bfloat16(), (sin, cos))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e_ac = eager(model, blk, x, (sin, cos))
            f_bf = ops.fused_block(x.bfloat16().contiguous(), w, sin, cos, prefix, H)
            f_32 = ops.fused_block(x.contiguous(), w, sin, cos, prefix, H)
            f_32f = ops.fused_block(x.contiguous(), wf, sin, cos, prefix, H)
        rn = lambda o: (o.float() - ref).norm().item() / ref.norm().item()  # noqa: E731
        row = {"N": N, "ref_is_chain": tensor_diff(ref, chain_ref)["max_abs"], "ref_absmax": ref.abs().max().item()}
        for k, o in (("eager_bf16", e_bf), ("eager_ac", e_ac), ("fused_bf16res", f_bf), ("fused_fp32res", f_32), ("fused_fp32res_foldgamma", f_32f)):
            df = tensor_diff(o, ref)
            row[k] = {"max_abs": df["max_abs"], "cos": df["cos"], "rel_l2": rn(o), "nan": df["nan"]}
        # fused vs eager of the same precision
        row["fused_bf16res_vs_eager_bf16"] = tensor_diff(f_bf, e_bf)
        row["fused_fp32res_vs_eager_ac"] = tensor_diff(f_32, e_ac)
        out[f"{name}_frame{fi}_layer{i}"] = row
        print(f"{name} frame{fi} layer{i} N={N} |ref|max={row['ref_absmax']:.1f}")
        for k in ("eager_bf16", "eager_ac", "fused_bf16res", "fused_fp32res", "fused_fp32res_foldgamma"):
            r = row[k]
            print(f"   {k:23s} vs fp32: max_abs {r['max_abs']:.4f} cos {r['cos']:.7f} rel_l2 {r['rel_l2']:.2e}")
write_results("/root/lightly-train/infer-opt/round3_fusion/results.json", {"test_block": out})
