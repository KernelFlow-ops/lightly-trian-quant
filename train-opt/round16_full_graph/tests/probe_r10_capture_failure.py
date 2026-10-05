"""Root cause of the round10 whole-step capture failure ("Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the
CPU tensor is pinned").  Captures fwd + r3 padded loss + bwd + r7 capturable opt_step of the P2 stack as ONE graph.

  python probe_r10_capture_failure.py vitb16 P2          -> fails inside round7_optimizer/impl/fused_optimizer.py:25
        scale = torch.as_tensor(max_norm, dtype=total.dtype, device=total.device)    # python float -> cuda tensor = H2D from pageable memory
  python probe_r10_capture_failure.py vitb16 P2 --fix    -> same step with the clip scale computed as (total+eps).reciprocal().mul(max_norm).clamp(max=1)
        (no host->device copy): the whole P2 step captures and replays.  round16 avoids r7's clip altogether (round15 flat optimizer)."""
import os, sys, time, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
ROOT = "/root/lightly-train/train-opt"
sys.path.insert(0, ROOT)
from common import bench
from common.config import P1R13_STACK, P1R13_PATCH_KWARGS, P2_STACK, P2_PATCH_KWARGS
from common.harness import RunCfg, apply_patches, build, preload_batches, set_phase, train_step
from common.fast_forward import forward_train_v2
from round10_integration.impl.full_graph import padded_batch

model = sys.argv[1] if len(sys.argv) > 1 else "vitb16"
which = sys.argv[2] if len(sys.argv) > 2 else "P2"
stack, kw = (P2_STACK, P2_PATCH_KWARGS) if which == "P2" else (P1R13_STACK, P1R13_PATCH_KWARGS)
with bench.gpu_lock():
    cfg = RunCfg(model=model, patches=tuple(stack), patch_kwargs=dict(kw), seed=0, num_workers=0)
    ctx = build(cfg); apply_patches(ctx)
    batches = preload_batches(ctx, k=2)
    b = batches[0]
    print("batch keys", list(b.keys()), "image", b["image"].shape, b["image"].dtype, b["image"].device)
    bm = b["binary_masks"]
    print("binary_masks type", type(bm), len(bm), {k: (v.shape, v.dtype, v.device) for k, v in bm[0].items()})
    print("counts", [[int(x["masks"].shape[0]) for x in bb["binary_masks"]] for bb in batches])
    ctx.meta["phase_override"] = True
    set_phase(ctx, "mid")
    for i in range(3):
        train_step(ctx, batches[i % 2], i)
    torch.cuda.synchronize()
    print("eager ok; meta keys", [k for k in ctx.meta.keys()])
    dev = next(ctx.train_model.parameters()).device
    sb = padded_batch(batches[0], dev, max_targets=4)
    ctx.meta["round7_graph_capture"] = True
    import round7_optimizer.impl.patch as r7p
    def fixed_clip(parameters, max_norm, eps=1e-6):
        params = [p for p in parameters if p.grad is not None]
        grads = [p.grad for p in params]
        norms = torch._foreach_norm(grads, 2.0)
        total = torch.linalg.vector_norm(torch.stack(tuple(norms)), ord=2)
        scale = (total + eps).reciprocal().mul(max_norm).clamp(max=1.0)
        torch._foreach_mul_(grads, scale)
        return total
    if "--fix" in sys.argv:
        r7p.fused_clip_grad_norm_ = fixed_clip
    def body():
        masks, classes = forward_train_v2(ctx, sb["image"], return_logits_per_layer=True)
        from contextlib import nullcontext
        loss, _ = ctx.ops.loss_all_layers(ctx, masks, classes, sb["binary_masks"])
        ctx.fabric.backward(loss)
        ctx.ops.opt_step(ctx)
        return loss
    # warmup in eager with the same body (autocast through fabric)
    def body_ac():
        with ctx.fabric.autocast():
            return body()
    for _ in range(2):
        body_ac()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    t0 = time.time()
    try:
        with torch.cuda.graph(g):
            out = body_ac()
        torch.cuda.synchronize()
        print("CAPTURE OK in", time.time() - t0, "s")
        t1 = time.time()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        print("replay ms/step", (time.time() - t1) / 5 * 1e3, "loss", float(out))
    except Exception:
        traceback.print_exc()
