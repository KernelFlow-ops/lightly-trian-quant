"""All 126 frames of the real video vs fp32 eager: pooled head IoU and per-frame confidence rel. diff distribution for
eager bf16-autocast / round3 fused / round5 v1 / round5 v2.  usage: python test_heldout_v2.py [vitl16|vitb16]"""
import sys, json
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import cv2, numpy as np, torch
import torch.nn.functional as F
import model_factory as mf
from bench import write_results
import ops as r5
from r3load import load_r3

r3 = load_r3()
name = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
model = mf.load_model(name, torch.float32)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
nb = len(bb.blocks)
W3 = [r3.prepare_block_weights(b) for b in bb.blocks]
W5 = [r5.prepare_block_weights(b) for b in bb.blocks]
first_joint = nb - model.num_joint_blocks
Q = model.queries.weight.detach().float()


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
                x = torch.cat((model.queries.weight[None].expand(x.shape[0], -1, -1).float(), x), dim=1).contiguous()
            x = ops.fused_block(x, Wl[i], sin, cos, x.shape[1] - 1024, H, False)
    else:
        st = r5.v2_enter(x)
        for i in range(nb):
            if i == first_joint:
                st = r5.v2_enter(st.x, Q)
            st = r5.fused_block_v2(st, W5[i], sin, cos, st.x.shape[1] - 1024, H)
        x = st.x
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ml, cl = model._predict(bb.norm(x), grid_size=grid)
    return F.interpolate(ml, (Hh, Ww), mode="bilinear"), cl


def run_frame(fr, variant):
    with torch.inference_mode():
        batch, meta = mf.base_prep(model, fr)
        if variant == "fp32":
            raw = model.forward_backend(batch)
            logits = mf.base_post_logits(model, raw, meta).float()
        else:
            if variant == "eager_autocast":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model.forward_backend(batch)
            else:
                raw = backend(batch, variant)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mf.base_post_logits(model, raw, meta).float()
        mask, _, conf = mf.host_finalize(model, logits.cpu().numpy())
    return mask, conf


VARS = ("eager_autocast", "r3_v1", "r5_v1", "r5_v2")
stats = {v: dict(inter=0, union=0, conf=[], iou=[]) for v in VARS}
cap = cv2.VideoCapture(mf.VIDEO_PATH)
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    rm, rc = run_frame(fr, "fp32")
    for v in VARS:
        m, c = run_frame(fr, v)
        x, y = rm == 1, m == 1
        i_, u_ = int((x & y).sum()), int((x | y).sum())
        stats[v]["inter"] += i_; stats[v]["union"] += u_
        stats[v]["conf"].append(abs(c - rc) / max(abs(rc), 1e-6)); stats[v]["iou"].append(i_ / u_ if u_ else 1.0)
    n += 1
    if n % 20 == 0:
        print("frames", n, flush=True)
res = {}
for v in VARS:
    cr = np.array(stats[v]["conf"])
    res[v] = {"frames": n, "pooled_iou": stats[v]["inter"] / stats[v]["union"], "min_frame_iou": float(min(stats[v]["iou"])),
              "conf_max": float(cr.max()), "conf_p99": float(np.percentile(cr, 99)), "conf_p90": float(np.percentile(cr, 90)),
              "conf_mean": float(cr.mean()), "frames_over_5pct": int((cr > 0.05).sum()), "frames_over_3pct": int((cr > 0.03).sum())}
    print(name, v, json.dumps(res[v]), flush=True)
write_results("/root/lightly-train/infer-opt/round5_gemm_sm120/results.json", {"heldout_126": {name: res}})
