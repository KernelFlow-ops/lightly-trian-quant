"""Whole-model accuracy gate with the fused block substituted for every transformer block.

Uses common/model_factory (real video frames, fp32 eager reference, mf.gate: pooled head IoU >= 0.95 and
per-frame confidence rel diff <= 5%).  Variants:
   eager_autocast : baseline (bf16 autocast, fp32 residual stream)  -- sanity, expected conf diff ~1%
   fused_fp32res  : fused_block, fp32 residual stream (autocast-like)
   fused_bf16res  : fused_block, bf16 residual stream
usage: python test_model_fused.py [vitl16|vitb16]
"""
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import model_factory as mf
import torch
import torch.nn.functional as F
from bench import write_results

from impl import ops

name = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
model = mf.load_model(name, torch.float32)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
nb = len(bb.blocks)
W = [ops.prepare_block_weights(b) for b in bb.blocks]
WF = [ops.prepare_block_weights(b, fold_gamma=True) for b in bb.blocks]
frames = mf.load_frames()


def fused_backend(x_img, res_dtype, Wl=None, rope_bf16=False):
    Wl = Wl or W
    Hh, Ww = x_img.shape[-2:]
    grid = (Hh // bb.patch_size, Ww // bb.patch_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x, image_size = bb.prepare_tokens_with_masks(x_img)
    x = x.to(res_dtype).contiguous()
    sin, cos = bb.rope_embed(H=image_size[0], W=image_size[1])
    for i in range(nb):
        if i == nb - model.num_joint_blocks:
            q = model.queries.weight[None].expand(x.shape[0], -1, -1).to(res_dtype)
            x = torch.cat((q, x), dim=1).contiguous()
        x = ops.fused_block(x, Wl[i], sin, cos, x.shape[1] - 1024, H, rope_bf16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mask_logits, class_logits = model._predict(bb.norm(x), grid_size=grid)
    return F.interpolate(mask_logits, (Hh, Ww), mode="bilinear"), class_logits


def run(variant):
    accs = {}
    for fi, fr in enumerate(frames):
        with torch.inference_mode():
            batch, meta = mf.base_prep(model, fr)
            if variant == "eager_autocast":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model.forward_backend(batch)
            elif variant == "fused_fp32res_foldgamma":
                raw = fused_backend(batch, torch.float32, WF)
            elif variant == "fused_fp32res_ropebf16":
                raw = fused_backend(batch, torch.float32, None, True)
            else:
                raw = fused_backend(batch, torch.float32 if variant == "fused_fp32res" else torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mf.base_post_logits(model, raw, meta).float()
            mask, _, conf = mf.host_finalize(model, logits.cpu().numpy())
        accs[f"frame{fi}"] = mf.compare_to_ref(name, fi, logits, mask, conf)
    g = mf.gate(accs)
    return accs, g


res = {}
for v in ("eager_autocast", "fused_fp32res", "fused_fp32res_ropebf16", "fused_fp32res_foldgamma", "fused_bf16res"):
    accs, g = run(v)
    res[v] = {"gate": g, "frames": accs}
    print(f"{name} {v:15s} pooled_iou {g['pooled_head_iou']:.4f}  min_frame_iou {g['min_frame_head_iou']:.4f}  "
          f"max_conf_rel {g['max_conf_rel_diff']*100:.2f}%  pass={g['pass']}  "
          f"logits_cos_min {min(a['logits_cos'] for a in accs.values()):.6f}")
write_results("/root/lightly-train/infer-opt/round3_fusion/results.json", {"model_gate": {name: res}})

# ---- model-stage latency (batch=2 crops, 512x512), fused blocks vs eager baseline
if "--time" in sys.argv:
    from bench import time_fn
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        batch, meta = mf.base_prep(model, frames[0])

    def f_eager():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_backend(batch)

    def f_fused():
        return fused_backend(batch, torch.float32)

    def nk(fn):
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        ns = [e.name for e in p.events() if e.device_type == DeviceType.CUDA]
        return sum(1 for n in ns if "emcpy" not in n and "emset" not in n)

    out = {}
    for k, fn in (("eager_autocast", f_eager), ("fused_fp32res", f_fused)):
        with torch.inference_mode():
            fn()
        t = time_fn(fn, warmup=20, iters=60, repeats=3)
        out[k] = {"gpu_ms": t["gpu_ms_p50"], "wall_ms": t["wall_ms_p50"], "kernels": nk(fn)}
        print(f"{name} model stage {k:15s} gpu {t['gpu_ms_p50']:.2f} ms  wall {t['wall_ms_p50']:.2f} ms  kernels {out[k]['kernels']}")
    write_results("/root/lightly-train/infer-opt/round3_fusion/results.json", {"model_stage_latency": {name: out}})
