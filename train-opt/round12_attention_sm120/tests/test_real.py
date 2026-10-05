"""Gate A on REAL activations: capture (qkv, rope, dO) of selected blocks during one ViT-B/L training step of the TB1 stack, then compare the fused
kernels and the TB1 attention path with an fp64 reference on exactly those tensors."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
import round11_bf16_mixed.run as r11run  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round12_attention_sm120.impl import reference as ref  # noqa: E402
from round12_attention_sm120.impl.fused import make_attn_core as make_fused  # noqa: E402


def rel(a, b):
    return float((a.detach().double() - b).norm() / b.norm())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vitb16")
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 3, 8])
    p.add_argument("--phase", default="late")
    a = p.parse_args()
    cfg = r11run.make_cfg(a.model, "r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16", phase=a.phase)
    ctx = build(cfg)
    apply_patches(ctx)
    batches = preload_batches(ctx, k=1)
    ctx.meta["phase_override"] = True
    set_phase(ctx, a.phase)
    ctx.rng = RngTape(mode="off")
    base = ctx.ops.attn_core
    store = {}
    counter = {"i": 0}

    def capture(c, qkv, rope, mask, h):
        i = counter["i"]
        counter["i"] += 1
        o = base(c, qkv, rope, mask, h)
        if i in a.blocks and mask is None:
            store[i] = {"qkv": qkv.detach().clone(), "rope": rope, "h": h, "o_tb1": o.detach().clone()}
            def grab(g, i=i):
                store[i]["do"] = g.detach().clone()
            o.register_hook(grab)
        return o

    ctx.ops.attn_core = capture
    train_step(ctx, batches[0], 0)
    for i, d in store.items():
        torch.save({k: (v.cpu() if torch.is_tensor(v) else (tuple(t.cpu() for t in v) if isinstance(v, tuple) else v)) for k, v in d.items()}, f'/root/autodl-tmp/train-opt-tmp/round12_attention_sm120/real_{a.model}_block{i}.pt')
    fused = make_fused()

    class C:
        pass

    for i in sorted(store):
        s = store[i]
        qkv, rope, h, do = s["qkv"], s["rope"], s["h"], s["do"]
        B, N, c3 = qkv.shape
        prefix = N - rope[0].shape[0]
        xd = qkv.reshape(B, N, 3, h, 64).double().requires_grad_()
        o_ref, _ = ref.ref_attention(xd, rope, None, prefix, torch.float64, round_bf16=True)
        (g_ref,) = torch.autograd.grad(o_ref, xd, do.double().view(B, N, h * 64))
        res = {}
        for name, fn in (("fused", fused), ("TB1", base)):
            q = qkv.clone().requires_grad_()
            o = fn(C(), q, rope, None, h)
            (g,) = torch.autograd.grad(o, q, do)
            g = g.view(B, N, 3, h, 64)
            res[name] = {"O": rel(o, o_ref), "dq": rel(g[:, :, 0], g_ref[:, :, 0]), "dk": rel(g[:, :, 1], g_ref[:, :, 1]), "dv": rel(g[:, :, 2], g_ref[:, :, 2]),
                         "absmax_qkv": float(qkv.abs().max()), "absmax_do": float(do.abs().max())}
        print(f"block {i}: |qkv|max {res['fused']['absmax_qkv']:.1f} |dO|max {res['fused']['absmax_do']:.2e}", flush=True)
        for name, r in res.items():
            print(f"   {name:6s} O {r['O']:.2e} dq {r['dq']:.2e} dk {r['dk']:.2e} dv {r['dv']:.2e}", flush=True)


if __name__ == "__main__":
    main()
