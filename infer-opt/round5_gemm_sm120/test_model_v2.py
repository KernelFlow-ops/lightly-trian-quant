"""Whole-model accuracy gate (mf.gate: pooled head IoU >= 0.95, per-frame conf rel err <= 5%) with every block replaced by
round3 fused_block / round5 fused_block / round5 fused_block_v2 (fp32 residual), 8 real video frames, + model-stage timing.
usage: python test_model_v2.py [vitl16|vitb16] [--time]"""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import torch.nn.functional as F
import model_factory as mf
from bench import write_results
import ops as r5
from r3load import load_r3

r3 = load_r3()
name = next((a for a in sys.argv[1:] if not a.startswith("--")), "vitl16")
model = mf.load_model(name, torch.float32)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
nb = len(bb.blocks)
W3 = [r3.prepare_block_weights(b) for b in bb.blocks]
W5 = [r5.prepare_block_weights(b) for b in bb.blocks]
frames = mf.load_frames()
first_joint = nb - model.num_joint_blocks
QUERIES = model.queries.weight.detach().float()


def backend(x_img, variant):
    Hh, Ww = x_img.shape[-2:]
    grid = (Hh // bb.patch_size, Ww // bb.patch_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x, image_size = bb.prepare_tokens_with_masks(x_img)
    x = x.float().contiguous()
    sin, cos = bb.rope_embed(H=image_size[0], W=image_size[1])
    if variant in ("r3_v1", "r5_v1"):
        ops, Wl = (r3, W3) if variant == "r3_v1" else (r5, W5)
        for i in range(nb):
            if i == first_joint:
                q = model.queries.weight[None].expand(x.shape[0], -1, -1).float()
                x = torch.cat((q, x), dim=1).contiguous()
            x = ops.fused_block(x, Wl[i], sin, cos, x.shape[1] - 1024, H, False)
    else:
        st = r5.v2_enter(x)
        for i in range(nb):
            if i == first_joint:
                st = r5.v2_enter(st.x, QUERIES)
            st = r5.fused_block_v2(st, W5[i], sin, cos, st.x.shape[1] - 1024, H, pdl=(variant != "r5_v2_nopdl"))
        x = st.x
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
            else:
                raw = backend(batch, variant)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mf.base_post_logits(model, raw, meta).float()
            mask, _, conf = mf.host_finalize(model, logits.cpu().numpy())
        accs[f"frame{fi}"] = mf.compare_to_ref(name, fi, logits, mask, conf)
    return accs, mf.gate(accs)


VARS = ("eager_autocast", "r3_v1", "r5_v1", "r5_v2_nopdl", "r5_v2")
res = {}
for v in VARS:
    accs, g = run(v)
    res[v] = {"gate": g, "frames": accs}
    print(f"{name} {v:15s} pooled_iou {g['pooled_head_iou']:.4f}  min_frame_iou {g['min_frame_head_iou']:.4f}  "
          f"max_conf_rel {g['max_conf_rel_diff']*100:.2f}%  pass={g['pass']}  logits_cos_min {min(a['logits_cos'] for a in accs.values()):.6f}", flush=True)
    print("     per-frame conf rel err:", " ".join("%.2f%%" % (100 * a["confidence_rel_diff"]) for a in accs.values()), flush=True)
write_results("/root/lightly-train/infer-opt/round5_gemm_sm120/results.json", {"model_gate": {name: res}})

if "--time" in sys.argv:
    from bench import time_fn
    from kt import kt
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        batch, meta = mf.base_prep(model, frames[0])
        xb = batch.bfloat16().contiguous() if False else batch

    def mk(v):
        if v == "eager_autocast":
            def f():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model.forward_backend(batch)
            return f
        return lambda: backend(batch, v)

    def nk(fn):
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        return sum(1 for e in p.events() if e.device_type == DeviceType.CUDA and "emcpy" not in e.name and "emset" not in e.name)

    out = {}
    for v in VARS:
        fn = mk(v)
        with torch.inference_mode():
            fn()
        t = time_fn(fn, warmup=20, iters=60, repeats=3)
        # CUDA-graph replay of the whole model stage (GPU busy time, no launch gaps): the Round-9 pipeline runs it this way
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                fn()
        from bench import gpu_lock
        with gpu_lock():
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(30):
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
        ts.sort()
        out[v] = {"gpu_ms_eager": t["gpu_ms_p50"], "wall_ms_eager": t["wall_ms_p50"], "kernels": nk(fn), "graph_ms_p50": ts[len(ts) // 2], "graph_ms_min": ts[0]}
        print(f"{name} model stage {v:15s} eager gpu {t['gpu_ms_p50']:.2f} ms wall {t['wall_ms_p50']:.2f} ms  kernels {out[v]['kernels']}  graph p50 {out[v]['graph_ms_p50']:.3f} ms (min {out[v]['graph_ms_min']:.3f})", flush=True)
    write_results("/root/lightly-train/infer-opt/round5_gemm_sm120/results.json", {"model_stage_latency": {name: out}})
