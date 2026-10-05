"""R22 gate A: real-kernel FP4 backbone (``impl/fp4_block22.py``) tests.  One process, dual GPU lock.

  a  GEMM layer: ``G4.dgrad`` / ``dgrad_dgelu`` / ``wgrad`` on fp4q operands (real shapes, M tails, gaussian + spiky data) vs the dequantized fp32 product, rel L2 <= 2e-3
  b  RN block vs round20b's SIMULATED block (same policy, amax=exact, RN): out / dx / dW cosine >= 0.99999  (plain + joint block, normal + spiky data)
  c  SR block (real Philox kernel vs sim torch generator): statistical consistency of the per-seed gradient cosines against the FINAL block
  d  empty / nofirst policies: block 0 with nofirst is FINAL's block op for op; state machine (roll once per opt_step) + delayed init guard

    python tests/test_block22.py --model vitb16 [--only a,b,c,d]   ->  profile/test_block22_<model>.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/quack_cache"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

from common import bench  # noqa: E402
from round20_fp4_probe.impl import fp4_ref as R  # noqa: E402
from round21_fp4_quant_head.impl import fp4q  # noqa: E402
from round22_fp4_backbone.impl import fp4_gemm22 as G4  # noqa: E402

dev = "cuda"
RES: dict = {}


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def cos(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    if float(a.norm()) == 0.0 and float(b.norm()) == 0.0:
        return 1.0
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def rec(name, **kw):
    RES[name] = kw
    print(name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in kw.items()}), flush=True)


def spiky(x, g):
    """scale-8 outlier channels + a few massive tokens (the shape of the ViT-L block-0 massive activation)."""
    x = x.clone().float()
    nch = max(2, x.shape[-1] // 128)
    ch = torch.randperm(x.shape[-1], device=x.device, generator=g)[:nch]
    x[..., ch] *= 8.0
    tok = torch.randperm(x.shape[0], device=x.device, generator=g)[:4]
    x[tok] *= 300.0
    return x


def gelu_d(z):
    return 0.5 * (1 + torch.erf(z / 2 ** 0.5)) + z * torch.exp(-0.5 * z * z) / (2 * 3.141592653589793) ** 0.5


def deq_row(t):
    return R.dequant_copy(t.q, t.sf, t.ts, "nv4", t.rows, t.cols)


def deq_wt(t):  # W^T [K, N]
    return R.dequant_copy(t.qt, t.sft, t.tst, "nv4", t.cols, t.rows)


def deq_col(t):  # [C, Kt]
    return R.dequant_copy(t.qt, t.sft, t.tst, "nv4", t.cols, t.kt)


# ------------------------------------------------------------------------------------------------ a
def test_a():
    g = torch.Generator(device=dev).manual_seed(1)
    cases = {"vitl16": (1024, (7203, 7903)), "vitb16": (768, (14406, 15806))}
    n_fail = 0
    for model, (d, Ms) in cases.items():
        shapes = {"qkv": (d, 3 * d), "proj": (d, d), "fc1": (d, 4 * d), "fc2": (4 * d, d)}
        for M in Ms:
            for name, (K, N) in shapes.items():
                for kind in ("gauss", "spiky"):
                    state = fp4q.FP4State(3, device=dev)
                    x = torch.randn(M, K, device=dev, generator=g)
                    dy = torch.randn(M, N, device=dev, generator=g) * 1e-3
                    W = torch.randn(N, K, device=dev, generator=g) * 0.05
                    z = torch.randn(M, K, device=dev, generator=g).bfloat16()
                    if kind == "spiky":
                        x, dy = spiky(x, g), spiky(dy, g)
                    x, dy = x.bfloat16(), dy.bfloat16()
                    xq = fp4q.quant_act(x, "nv4", row=False, col=True, amax="exact", state=state, site=0)
                    dq = fp4q.quant_act(dy, "nv4", row=True, col=True, sr_row=True, sr_col=True, sr_offset=1, amax="exact", state=state, site=1)
                    wq = fp4q.quant_weight(W, "nv4", amax="exact", state=state, site=2)
                    Dh, Wt = deq_row(dq).float(), deq_wt(wq).float()
                    out = {}
                    acc = Dh @ Wt.t()
                    if name == "fc2":
                        dz, db = G4.dgrad_dgelu(dq, wq, z)
                        ref = acc * gelu_d(z.float())
                        out["dgrad_dgelu"], out["dgrad_dgelu_db"] = rel(dz.float(), ref), rel(db, ref.sum(0))
                    else:
                        out["dgrad"] = rel(G4.dgrad(dq, wq).float(), acc)
                    dw = G4.wgrad(dq, xq)
                    out["wgrad"] = rel(dw, deq_col(dq).float() @ deq_col(xq).float().t())
                    if name == "proj":
                        out["wgrad_nosplit"] = rel(G4.wgrad(dq, xq, split_k=False), deq_col(dq).float() @ deq_col(xq).float().t())
                    ok = all(v <= 2e-3 for v in out.values())
                    n_fail += (not ok)
                    rec(f"a_{model}_M{M}_{name}_{kind}", ok=ok, **out)
                    del x, dy, W, z, xq, dq, wq, Dh, Wt, acc, dw
                torch.cuda.empty_cache()
    # M tails (7203 / 14406 covered above); small M
    for M in (1029, 33):
        K, N = 1024, 4096
        state = fp4q.FP4State(3, device=dev)
        x = torch.randn(M, K, device=dev, generator=g).bfloat16()
        dy = (torch.randn(M, N, device=dev, generator=g) * 1e-3).bfloat16()
        W = torch.randn(N, K, device=dev, generator=g) * 0.05
        xq = fp4q.quant_act(x, "nv4", row=False, col=True, amax="exact", state=state, site=0)
        dq = fp4q.quant_act(dy, "nv4", row=True, col=True, sr_row=True, sr_col=True, sr_offset=1, amax="exact", state=state, site=1)
        wq = fp4q.quant_weight(W, "nv4", amax="exact", state=state, site=2)
        out = {"dgrad": rel(G4.dgrad(dq, wq).float(), deq_row(dq).float() @ deq_wt(wq).float().t()),
               "wgrad": rel(G4.wgrad(dq, xq), deq_col(dq).float() @ deq_col(xq).float().t())}
        ok = all(v <= 2e-3 for v in out.values())
        n_fail += (not ok)
        rec(f"a_tail_M{M}", ok=ok, **out)
    RES["a_failures"] = n_fail
    print("a failures:", n_fail, flush=True)


# ------------------------------------------------------------------------------------------------ model-level helpers (adapted from round20 tests/test_fp4_block.py)
def build_ctx(model):
    from common.config import BEST_STACK_V2, BEST_STACK_V2_KWARGS
    from common.harness import RunCfg, apply_patches, build
    from round19_final.impl import presets
    presets.register(ROUND)
    stack = BEST_STACK_V2 + ("round20_fp4_probe", "round22_fp4_backbone")
    kw = {**BEST_STACK_V2_KWARGS, "round20_fp4_probe": {"variant": ""}, "round22_fp4_backbone": {"variant": ""}}
    cfg = RunCfg(model=model, batch_size=None, phase="mid", patches=stack, patch_kwargs=kw, seed=0, num_workers=0)
    ctx = build(cfg)
    apply_patches(ctx)
    return ctx


def block_inputs(ctx, joint=False, seed=0, kind="normal"):
    model = ctx.train_model.model
    B = ctx.meta["batch_size"]
    g = torch.Generator(device=dev).manual_seed(seed)
    img = torch.randn(B, 3, 512, 512, device=dev, generator=g)
    with torch.no_grad(), ctx.fabric.autocast():
        x, size = model.backbone.prepare_tokens_with_masks(img)
        rope = ctx.ops.rope(ctx, 0, size[0], size[1])
        if joint:
            x = torch.cat((model.queries.weight[None].expand(x.shape[0], -1, -1), x), dim=1)
    x = x.float().contiguous()
    if kind == "spiky":  # outlier channels (x8) on every token + massive cls / register tokens (x300 on 4 channels), the shape of ViT-L block-0 activations
        gg = torch.Generator(device=dev).manual_seed(seed + 100)
        d = x.shape[-1]
        ch = torch.randperm(d, device=dev, generator=gg)[:max(2, d // 128)]
        x[..., ch] *= 8.0
        ch2 = torch.randperm(d, device=dev, generator=gg)[:4]
        x[:, : (105 if joint else 5), :][..., ch2] *= 300.0
    return x, rope


def run_block(ctx, fn, i, x, rope, wout):
    blk = ctx.model.backbone.blocks[i]
    for p in blk.parameters():
        p.grad = None
    xi = x.clone().requires_grad_(True)
    with ctx.fabric.autocast():
        out = fn(ctx, i, blk, xi, rope, None)
    (out.float() * wout).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in blk.named_parameters() if p.grad is not None}
    for p in blk.parameters():
        p.grad = None
    return out.detach().float(), xi.grad.detach().clone(), grads


def cmp_runs(a, b):
    gr = {n: cos(a[2][n], b[2][n]) for n in a[2]}
    worst = min(gr.items(), key=lambda kv: kv[1])
    return {"out_cos": cos(a[0], b[0]), "dx_cos": cos(a[1], b[1]), "dW_cos_min": worst[1], "dW_cos_min_name": worst[0],
            "bitwise": bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and all(torch.equal(a[2][n], b[2][n]) for n in a[2])), "out_rel": rel(a[0], b[0]), "dx_rel": rel(a[1], b[1]),
            "dW_rel_max": max(rel(a[2][n], b[2][n]) for n in a[2] if float(b[2][n].norm()) > 0)}


def blocks_of(ctx):
    blk22 = ctx.ops.block
    return blk22, blk22.previous, ctx.meta["round22_fp4_backbone"], ctx.meta["round20_fp4_probe"]


def pol_pair(base: str, extra20: str = "", extra22: str = ""):
    """(round20 sim policy, round22 real policy) for the same backbone policy ``base`` (round22 ``+``-joined grammar; both parsers accept it)."""
    return base + extra20, base + extra22


# ------------------------------------------------------------------------------------------------ b
def test_b(ctx, model):
    blk22, blk20, h22, h20 = blocks_of(ctx)
    nb = len(ctx.model.backbone.blocks)
    pols = {"rn": "all.d=nv4+rn+all.w=nv4+rn+amax=exact", "rn_rht": "all.d=nv4+rn+all.w=nv4+rn+rht+amax=exact"}
    cases = [(False, 1, "normal"), (True, nb - 1, "normal"), (False, 1, "spiky"), (True, nb - 1, "spiky"), (False, 0, "spiky")]
    n_fail = 0
    for pname, pol in pols.items():
        for joint, i, kind in cases:
            x, rope = block_inputs(ctx, joint=joint, kind=kind)
            wout = torch.randn(x.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
            h20.set_policy(pol)
            h20.begin_step()
            a = run_block(ctx, blk20, i, x, rope, wout)
            h22.set_policy(pol)
            h22.begin_step()
            b = run_block(ctx, blk22, i, x, rope, wout)
            c = cmp_runs(a, b)
            ok = c["out_cos"] >= 0.99999 and c["dx_cos"] >= 0.99999 and c["dW_cos_min"] >= 0.99999
            n_fail += (not ok)
            rec(f"b_{model}_{pname}_blk{i}_{'joint' if joint else 'plain'}_{kind}", ok=ok, **c)
            del a, b, x, wout
            torch.cuda.empty_cache()
    RES[f"b_failures_{model}"] = n_fail
    h20.set_policy("")
    h22.set_policy("")


# ------------------------------------------------------------------------------------------------ c
def test_c(ctx, model, nseed=8):
    """SR: per-seed cosine of (dx, dW) vs the FINAL block; real Philox kernel vs sim torch generator must agree in distribution."""
    blk22, blk20, h22, h20 = blocks_of(ctx)
    nb = len(ctx.model.backbone.blocks)
    pol = "all.d=nv4+sr+all.w=nv4+sr"
    n_fail = 0
    for joint, i in ((False, 1), (True, nb - 1)):
        x, rope = block_inputs(ctx, joint=joint)
        wout = torch.randn(x.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
        h20.set_policy("")
        h20.begin_step()
        ref = run_block(ctx, blk20, i, x, rope, wout)
        stats = {"real": [], "sim": []}
        acc = {"real": None, "sim": None}
        for k in range(nseed):
            for which, blk, h in (("sim", blk20, h20), ("real", blk22, h22)):
                h.set_policy(pol + "+amax=exact")
                if which == "sim":
                    h.sim.reset(seed=1000 + k, amax="exact")
                else:
                    h.state.seed.fill_(1000 + k)
                h.begin_step()
                r = run_block(ctx, blk, i, x, rope, wout)
                c = cmp_runs(ref, r)
                stats[which].append((c["dx_cos"], c["dW_cos_min"], statistics.fmean(cos(ref[2][n], r[2][n]) for n in ref[2] if float(ref[2][n].norm()) > 0)))
                gsum = r[2]
                if acc[which] is None:
                    acc[which] = {n: v.double().clone() for n, v in gsum.items()}
                else:
                    for n, v in gsum.items():
                        acc[which][n] += v.double()
        out = {}
        for which in ("real", "sim"):
            out[f"{which}_dx_cos_mean"] = statistics.fmean(s[0] for s in stats[which])
            out[f"{which}_dx_cos_std"] = statistics.pstdev(s[0] for s in stats[which])
            out[f"{which}_dW_cos_min_mean"] = statistics.fmean(s[1] for s in stats[which])
            out[f"{which}_dW_cos_mean_over_params"] = statistics.fmean(s[2] for s in stats[which])
            m = {n: v / nseed for n, v in acc[which].items()}
            out[f"{which}_dW_avg_cos_vs_final_min"] = min(cos(ref[2][n], m[n]) for n in ref[2] if float(ref[2][n].norm()) > 0)
        d_dx = abs(out["real_dx_cos_mean"] - out["sim_dx_cos_mean"])
        d_dw = abs(out["real_dW_cos_mean_over_params"] - out["sim_dW_cos_mean_over_params"])
        ok = d_dx <= 0.003 and d_dw <= 0.003
        n_fail += (not ok)
        rec(f"c_{model}_blk{i}_{'joint' if joint else 'plain'}_SR_n{nseed}", ok=ok, d_dx_cos_mean=d_dx, d_dW_cos_mean=d_dw, **out)
        del ref, x, wout, acc
    RES[f"c_failures_{model}"] = n_fail
    h20.set_policy("")
    h22.set_policy("")


# ------------------------------------------------------------------------------------------------ d
def test_d(ctx, model):
    """nofirst block 0 == FINAL's block; the delayed init guard; state roll through the opt_step wrapper."""
    blk22, blk20, h22, h20 = blocks_of(ctx)
    x, rope = block_inputs(ctx, joint=False, kind="normal")
    wout = torch.randn(x.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
    h20.set_policy("")
    h20.begin_step()
    a = run_block(ctx, blk20, 0, x, rope, wout)
    h22.set_policy("all.d=nv4+sr+all.w=nv4+sr+nofirst")
    h22.begin_step()
    b = run_block(ctx, blk22, 0, x, rope, wout)
    c = cmp_runs(a, b)
    rec(f"d_{model}_nofirst_block0_vs_FINAL", **{k: v for k, v in c.items()})
    assert c["out_cos"] >= 0.99999 and c["dW_cos_min"] >= 0.99999, c
    # empty policy == FINAL for a plain block
    h22.set_policy("")
    h22.begin_step()
    b = run_block(ctx, blk22, 1, x, rope, wout)
    a = run_block(ctx, blk20, 1, x, rope, wout)
    rec(f"d_{model}_empty_policy_vs_FINAL", **cmp_runs(a, b))
    # guard: a first delayed call inside a CUDA graph capture must raise (emulated: torch.cuda.is_current_stream_capturing -> True)
    h22.set_policy("all.d=nv4+sr+all.w=nv4+sr")
    orig_cap = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    raised = False
    try:
        h22.begin_step()
    except RuntimeError as e:
        raised = "first call of delayed-amax site" in str(e)
    finally:
        torch.cuda.is_current_stream_capturing = orig_cap
    rec(f"d_{model}_capture_guard", raised=raised)
    assert raised
    # state roll via the opt_step wrapper (the real wrapper object, with a stub inner step)
    wrapper = ctx.ops.opt_step
    assert hasattr(wrapper, "inner") and hasattr(wrapper, "handle"), "round22 opt_step wrapper missing"
    h22.set_policy("all.d=nv4+sr+all.w=nv4+sr")
    s0 = int(h22.state.step.item())
    calls = {"n": 0}
    real_inner = wrapper.inner

    def fake(c):
        calls["n"] += 1
        return torch.zeros((), device=dev)

    wrapper.inner = fake
    try:
        for _ in range(3):
            wrapper(ctx)
    finally:
        wrapper.inner = real_inner
    s1 = int(h22.state.step.item())
    rec(f"d_{model}_roll_once_per_step", steps_before=s0, steps_after=s1, wrapper_calls=calls["n"], ok=(s1 - s0 == 3 == calls["n"]))
    assert s1 - s0 == 3
    h22.set_policy("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b,c,d")
    ap.add_argument("--model", default="vitb16")
    args = ap.parse_args()
    only = set(args.only.split(","))
    out = ROUND / "profile" / f"test_block22_{args.model}.json"
    if out.exists():
        RES.update(json.loads(out.read_text()))
    with bench.gpu_lock():
        from round16_full_graph import run as r16
        r16.wait_exclusive()
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        if "a" in only:
            test_a()
        if only & {"b", "c", "d"}:
            ctx = build_ctx(args.model)
            print(f"ctx built {time.time() - t0:.0f}s", flush=True)
            if "b" in only:
                test_b(ctx, args.model)
            if "c" in only:
                test_c(ctx, args.model)
            if "d" in only:
                test_d(ctx, args.model)
        print(f"done in {time.time() - t0:.0f}s", flush=True)
    out.write_text(json.dumps(RES, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
