"""Real-weight layer test: round3 fused_block (v1) vs round5 fused_block (tuned GEMMs, LN kernels) vs round5 fused_block_v2
(LayerNorm folded into the GEMMs), all with fp32 residual, against the fp32 eager block.  Inputs are captured from a real-frame fp32 forward.
usage: python test_block_v2.py [vitl16|vitb16] [layers]"""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import model_factory as mf
from bench import write_results
from check import tensor_diff
import ops as r5
from r3load import load_r3

r3 = load_r3()
name = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else ([0, 5, 12, 20, 21, 22, 23] if name == "vitl16" else [0, 6, 9, 10, 11])
model = mf.load_model(name, torch.float32)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
nb = len(bb.blocks)
frames = mf.load_frames()
cap = {}
for i in set(layers) | {l + 1 for l in layers if l + 1 < nb}:
    bb.blocks[i].norm1.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0].detach().clone()))
sin, cos = bb.rope_embed(H=32, W=32)
out = {}
for fi in (0, 3):
    cap.clear()
    with torch.inference_mode():
        batch, meta = mf.base_prep(model, frames[fi])
        model.forward_backend(batch)
    for i in layers:
        x = cap[i]
        B, N, d = x.shape
        prefix = N - 1024
        blk = bb.blocks[i]
        w3 = r3.prepare_block_weights(blk)
        w5 = r5.prepare_block_weights(blk)

        def eager(xin):
            xin = xin.clone()
            h = blk.norm1(xin)
            xin = xin + blk.ls1(model._attn(blk.attn, h, rope=(sin, cos), mask=None))
            return xin + blk.ls2(blk.mlp(blk.norm2(xin)))

        with torch.inference_mode():
            ref = eager(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e_ac = eager(x)
            f3 = r3.fused_block(x.contiguous(), w3, sin, cos, prefix, H)
            f5 = r5.fused_block(x.contiguous(), w5, sin, cos, prefix, H)
            st = r5.v2_enter(x.contiguous())
            f5b = r5.fused_block_v2(st, w5, sin, cos, prefix, H, pdl=False).x
            f5c = r5.fused_block_v2(st, w5, sin, cos, prefix, H, pdl=True).x
            r5.ATTN = "tl"
            f5d = r5.fused_block_v2(st, w5, sin, cos, prefix, H, pdl=True).x
            r5.ATTN = "flash"
        rn = lambda o: (o.float() - ref).norm().item() / ref.norm().item()  # noqa: E731
        row = {"N": N, "ref_absmax": ref.abs().max().item()}
        for k, o in (("eager_autocast", e_ac), ("r3_fused_v1", f3), ("r5_v1_tuned", f5), ("r5_v2_nopdl", f5b), ("r5_v2_pdl", f5c), ("r5_v2_tlattn", f5d)):
            df = tensor_diff(o, ref)
            row[k] = {"max_abs": df["max_abs"], "cos": df["cos"], "rel_l2": rn(o), "nan": df["nan"]}
        row["v2_pdl_vs_nopdl_identical"] = bool(torch.equal(f5b, f5c))
        out[f"{name}_frame{fi}_layer{i}"] = row
        print(f"{name} frame{fi} layer{i} N={N} |ref|max={row['ref_absmax']:.1f}  pdl==nopdl bitwise: {row['v2_pdl_vs_nopdl_identical']}")
        for k in ("eager_autocast", "r3_fused_v1", "r5_v1_tuned", "r5_v2_nopdl", "r5_v2_pdl", "r5_v2_tlattn"):
            r = row[k]
            print(f"   {k:16s} vs fp32: max_abs {r['max_abs']:.4f} cos {r['cos']:.8f} rel_l2 {r['rel_l2']:.2e}")
write_results("/root/lightly-train/infer-opt/round5_gemm_sm120/results.json", {"test_block_v2": out})
