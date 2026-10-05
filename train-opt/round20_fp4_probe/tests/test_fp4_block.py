"""R20b tests of the simulated FP4 block / head (``impl/fp4_block.py``, ``fp4_head.py``, ``patch.py``).  One process, dual GPU lock.

  a  policy parser
  b  fp4_sim vs fp4_ref (bit-exact for amax=exact, RN and explicit-u SR); delayed amax semantics; default torch generators untouched
  c  real QuACK FP4 GEMM dispatch (nv4 / mx4, ViT-B plain-block shapes) vs the dequantized fp64 product: fwd bias / res / gelu, dgrad, dgrad+dgelu, wgrad (+ prefix)
  d  prefix reproduces R20a's block-0 massive-activation numbers on the captured ViT-L tensors (cap_vitl16.pt)
  e  model level (ViT-B, FINAL stack + round20 patch):  policy "" block / head == FINAL's (r17 block, r14 head): output + input grads + parameter grads
  f  nv4 policy sanity of a plain block and a joint block against the fp32 default block (cosine of output, dx, dW)

    python tests/test_fp4_block.py [--only a,b,...]   ->  profile/test_fp4_block.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache_b"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

from common import bench  # noqa: E402
from round20_fp4_probe.impl import fp4_block as FB  # noqa: E402
from round20_fp4_probe.impl import fp4_gemm as G4  # noqa: E402
from round20_fp4_probe.impl import fp4_policy as P  # noqa: E402
from round20_fp4_probe.impl import fp4_ref as R  # noqa: E402
from round20_fp4_probe.impl import fp4_sim as S  # noqa: E402

dev = "cuda"
RES: dict = {}


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def cos(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    if float(a.norm()) == 0.0 and float(b.norm()) == 0.0:
        return 1.0  # e.g. attn.qkv.bias: exactly zero gradient in every stack (masked k-bias)
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def rec(name, **kw):
    RES[name] = kw
    print(name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in kw.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ a
def test_a():
    p = P.parse("head.f=mx8,head.dw=nv4+sr,all.w=nv4+sr+rht,prefix,amax=exact")
    assert p.prefix and p.amax == "exact"
    h, q, f2 = p.spec("head"), p.spec("qkv", 3), p.spec("fc2", 3)
    assert (h.ff, h.fd, h.fw, h.sr_d, h.sr_w, h.rht_w) == ("mx8", "nv4", "nv4", True, True, False)
    assert (q.ff, q.fd, q.fw, q.rht_w, q.prefix) == ("bf16", "mx8", "nv4", True, True)
    assert (f2.ff, f2.fd, f2.fw) == ("mx8", "mx8", "nv4")
    p = P.parse("all.d=nv4+rn,all.w=mx4+rht+w1d,nofirst,nolast,nojoint")
    assert p.spec("fc1", 0).is_final and p.spec("fc1", 23, 24).is_final and p.spec("fc1", 21, 24).is_final and not p.spec("fc1", 5, 24).is_final
    s = p.spec("fc1", 5, 24)
    assert (s.fd, s.sr_d, s.fw, s.rht_w, s.w2d_d) == ("nv4", False, "mx4", True, True)
    assert P.parse("").spec("qkv").fmts == ("bf16", "mx8", "mx8") and P.parse("").spec("head").fmts == ("bf16",) * 3
    for bad in ("sr", "foo.f=nv4", "all.x=nv4", "all.f=fp8", "all.f=nv4+bogus"):
        try:
            P.parse(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    rec("a_policy_parser", ok=True)


# ------------------------------------------------------------------------------------------------ b
def test_b():
    g = torch.Generator(device=dev).manual_seed(0)
    cpu0, cuda0 = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    sim = S.Sim(dev, seed=5, amax="exact")
    out = {}
    for fmt in ("nv4", "mx4"):
        x = (torch.randn(257, 512, device=dev, generator=g) * torch.exp(torch.randn(257, 512, device=dev, generator=g))).bfloat16()
        ref = R.quantize_matrix(x, fmt)
        got = S.q_matrix(sim, "k", x, fmt)
        out[f"{fmt}_rn"] = all(torch.equal(a, b) for a, b in zip(ref[:2], got[:2])) and (ref[2] is None or torch.equal(ref[2], got[2]))
        u = torch.rand(257, 512, device=dev, generator=g)
        ref = R.quantize_matrix(x, fmt, signs=sim.signs, sr=True, u=u)
        got = S.q_matrix(sim, "k", x, fmt, rht=True, sr=True, u=u)
        out[f"{fmt}_rht_sr"] = all(torch.equal(a, b) for a, b in zip(ref[:2], got[:2])) and (ref[2] is None or torch.equal(ref[2], got[2]))
        t_ref = R.quantize(x, fmt, rht_col=True, rht_row=False, sr=False)
        t_got = S.q_col(sim, "k", x, fmt, rht=True)
        out[f"{fmt}_col_rht"] = torch.equal(t_ref.qt, t_got.qt) and torch.equal(t_ref.sft, t_got.sft)
        w = (torch.randn(256, 512, device=dev, generator=g) * 0.05)
        for two_d in (True, False):
            wr = R.quantize_weight(w, fmt, two_d=two_d)
            wg = S.q_weight(sim, "w", w, fmt, two_d=two_d)
            out[f"{fmt}_w{'2d' if two_d else '1d'}"] = all(torch.equal(getattr(wr, n), getattr(wg, n)) for n in ("q", "sf", "qt", "sft"))
    # delayed amax: first call exact, then 2 x the previous amax, the current amax is recorded
    sd = S.Sim(dev, seed=5, amax="delayed")
    x1, x2 = torch.randn(128, 256, device=dev, generator=g), torch.randn(128, 256, device=dev, generator=g) * 3
    t1 = S.q_matrix(sd, "s", x1, "nv4")[2]
    t2 = S.q_matrix(sd, "s", x2, "nv4")[2]
    a1, a2 = float(x1.abs().max()), float(x2.abs().max())
    out["delayed_first_is_exact"] = abs(float(t1) - a1 / 2688.0) <= 1e-9 * a1
    out["delayed_second_uses_2x_prev"] = abs(float(t2) - 2 * a1 / 2688.0) <= 1e-9 * a1
    out["default_rng_untouched"] = bool(torch.equal(cpu0, torch.random.get_rng_state()) and torch.equal(cuda0, torch.cuda.get_rng_state()))
    rec("b_sim_vs_ref", **out)
    assert all(out.values()), out


# ------------------------------------------------------------------------------------------------ c
def deq_row(t):
    return R.dequant_copy(t.q, t.sf, t.ts, t.fmt, t.q.shape[0], t.cols)


def deq_wcol(t):  # W^T [K, N]
    return R.dequant_copy(t.qt, t.sft, t.tst, t.fmt, t.qt.shape[0], t.qt.shape[1] * 2)


def deq_col(t):  # [C, Rpad]
    return R.dequant_col(t)


def gelu(z):
    return torch.nn.functional.gelu(z)


def dgelu(z):
    return 0.5 * (1 + torch.erf(z / 2 ** 0.5)) + z * torch.exp(-0.5 * z * z) / (2 * 3.141592653589793) ** 0.5


def test_c():
    from round20_fp4_probe.impl import fp4_gemm  # noqa: F401
    g = torch.Generator(device=dev).manual_seed(1)
    B, ntok, d = 14, 1029, 768
    M = B * ntok
    shapes = {"qkv": (d, 3 * d), "proj": (d, d), "fc1": (d, 4 * d), "fc2": (4 * d, d)}
    for fmt in ("nv4", "mx4"):
        sim = S.Sim(dev, seed=3, amax="exact")
        for name, (K, N) in shapes.items():
            spec = P.LinSpec(name, fmt, fmt, fmt, sr_d=True, sr_w=True)
            lq = FB.LayerQ(sim, f"t.{name}", spec)
            x = torch.randn(M, K, device=dev, generator=g).bfloat16()
            dy = (torch.randn(M, N, device=dev, generator=g) * 1e-3).bfloat16()
            W = torch.randn(N, K, device=dev, generator=g) * 0.05
            bias = torch.randn(N, device=dev, generator=g) * 0.1
            gamma = torch.rand(N, device=dev, generator=g) * 0.2 + 0.05
            resid = torch.randn(M, N, device=dev, generator=g)
            z = torch.randn(M, K, device=dev, generator=g).bfloat16()
            wq = lq.prep_weight(W)
            xq = lq.quant_x(x, ntok)
            Xh, Wh = deq_row(xq.xf).double(), deq_row(wq.wf).double()
            acc = Xh @ Wh.t()
            out = {}
            if name == "qkv":
                y = lq.fwd_bias(xq, wq, bias, None)
                out["fwd_bias"] = rel(y.float(), (acc + bias.double()).float())
            elif name == "fc1":
                zz, h = lq.fwd_gelu(xq, wq, bias, None)
                zr = acc + bias.double()
                out["fwd_gelu_z"], out["fwd_gelu_h"] = rel(zz.float(), zr.float()), rel(h.float(), gelu(zr).float())
            else:
                o, y = lq.fwd_res(xq, wq, bias, gamma, resid, None)
                yr = acc + bias.double()
                out["fwd_res_y"], out["fwd_res_out"] = rel(y.float(), yr.float()), rel(o, (resid.double() + gamma.double() * yr).float())
            dq = lq.quant_dy(dy, ntok)
            Dh, Wt = deq_row(dq.dr).double(), deq_wcol(wq.wd).double()
            dxr = Dh @ Wt.t()
            if name == "fc2":
                dz, db = lq.dgrad_dgelu(dq, wq, None, z)
                ref = dxr * dgelu(z.double())
                out["dgrad_dgelu"], out["dgrad_dgelu_db"] = rel(dz.float(), ref.float()), rel(db, ref.sum(0).float())
            else:
                out["dgrad"] = rel(lq.dgrad(dq, wq, None).float(), dxr.float())
            dw = lq.wgrad(dq, xq)
            out["wgrad"] = rel(dw, (deq_col(dq.dc).double() @ deq_col(xq.xw).double().t()).float())
            # prefix: the FP4 contraction over the patch tokens + exact bf16 prefix GEMM
            lqp = FB.LayerQ(sim, f"t.{name}.p", P.LinSpec(name, fmt, fmt, fmt, sr_d=True, sr_w=True, prefix=True))
            xqp, dqp = lqp.quant_x(x, ntok), lqp.quant_dy(dy, ntok)
            dwp = lqp.wgrad(dqp, xqp)
            pos = torch.arange(M, device=dev) % ntok
            keep = pos >= ntok - 1024
            ref = deq_col(dqp.dc).double() @ deq_col(xqp.xw).double().t() + dy[~keep].double().t() @ x[~keep].double()
            out["wgrad_prefix"] = rel(dwp, ref.float())
            out["wgrad_prefix_vs_exact"] = rel(dwp, dy.double().t() @ x.double())
            out["wgrad_noprefix_vs_exact"] = rel(dw, dy.double().t() @ x.double())
            rec(f"c_gemm_{fmt}_{name}", **out)
            for k, v in out.items():
                if "vs_exact" in k:
                    continue
                lim = 1e-5 if k in ("wgrad", "wgrad_prefix", "fwd_res_out") else 3e-3
                assert v <= lim, (fmt, name, k, v)
            del x, dy, W, resid, z, xq, wq, dq, xqp, dqp
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ d
def test_d():
    path = "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/cap_vitl16.pt"
    store = torch.load(path)
    r20a = json.load(open(ROUND / "profile" / "debug_prefix_tokens.json"))
    sim = S.Sim(dev, seed=9, amax="exact")
    for key in ("block0.fc1", "block0.fc2", "block0.proj", "block0.qkv", "block22.fc2"):
        if key not in store:
            continue
        X, dY = store[key]["X"].to(dev), store[key]["dY"].to(dev)
        ntok = 1129 if key.startswith("block22") else 1029
        exact = dY.float().t() @ X.float()
        out = {}
        for fmt in ("nv4",):
            for rht in (False, True):
                for pref in (False, True):
                    spec = P.LinSpec("x", fmt, fmt, fmt, sr_d=False, sr_w=False, rht_w=rht, prefix=pref)
                    lq = FB.LayerQ(sim, "d", spec)
                    xq, dq = lq.quant_x(X.contiguous(), ntok), lq.quant_dy(dY.contiguous(), ntok)
                    dw = lq.wgrad(dq, xq)
                    out[f"{fmt}{'_rht' if rht else ''}{'_prefix' if pref else '_all'}"] = rel(dw, exact)
        ref = r20a.get(key, {})
        out["r20a_nv4_all"], out["r20a_nv4_prefix"] = ref.get("nv4_all_tokens"), ref.get("nv4_prefix_excluded_exact_side_gemm")
        out["r20a_nv4_rht_all"], out["r20a_nv4_rht_prefix"] = ref.get("nv4_rht_all_tokens"), ref.get("nv4_rht_prefix_excluded_exact_side_gemm")
        rec(f"d_prefix_{key}", **out)
        del X, dY, exact
    del store


# ------------------------------------------------------------------------------------------------ e / f  (model level)
def build_ctx(model="vitb16", variant=""):
    from common.config import BEST_STACK_V2, BEST_STACK_V2_KWARGS
    from common.harness import RunCfg, apply_patches, build
    from round19_final.impl import presets
    presets.register(ROUND)
    stack = BEST_STACK_V2 + ("round20_fp4_probe",)
    kw = {**BEST_STACK_V2_KWARGS, "round20_fp4_probe": {"variant": variant}}
    cfg = RunCfg(model=model, batch_size=None, phase="mid", patches=stack, patch_kwargs=kw, seed=0, num_workers=0)
    ctx = build(cfg)
    prev = {}
    # keep FINAL's block / predict: apply the FINAL patches first, snapshot, then ours
    cfg0 = RunCfg(model=model, batch_size=None, phase="mid", patches=BEST_STACK_V2, patch_kwargs=BEST_STACK_V2_KWARGS, seed=0, num_workers=0)
    ctx.cfg = cfg0
    apply_patches(ctx)
    prev["block"], prev["predict"] = ctx.ops.block, ctx.ops.predict
    ctx.cfg = cfg
    from common.harness import _state_shapes
    before = _state_shapes(ctx.train_model)
    import importlib
    mod = importlib.import_module("round20_fp4_probe.impl.patch")
    mod.apply(ctx, variant=variant)
    assert before == _state_shapes(ctx.train_model)
    ctx.meta["_prev"] = prev
    return ctx


def block_inputs(ctx, B=None, joint=False, seed=0):
    model = ctx.train_model.model
    B = B or ctx.meta["batch_size"]
    g = torch.Generator(device=dev).manual_seed(seed)
    img = torch.randn(B, 3, 512, 512, device=dev, generator=g)
    with torch.no_grad(), ctx.fabric.autocast():
        x, size = model.backbone.prepare_tokens_with_masks(img)
        rope = ctx.ops.rope(ctx, 0, size[0], size[1])
        if joint:
            x = torch.cat((model.queries.weight[None].expand(x.shape[0], -1, -1), x), dim=1)
    return x.float().contiguous(), rope


def run_block(ctx, fn, i, x, rope, wout, autocast=True):
    blk = ctx.model.backbone.blocks[i]
    for p in blk.parameters():
        p.grad = None
    xi = x.clone().requires_grad_(True)
    from contextlib import nullcontext
    with (ctx.fabric.autocast() if autocast else nullcontext()):
        out = fn(ctx, i, blk, xi, rope, None)
    (out.float() * wout).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in blk.named_parameters() if p.grad is not None}
    for p in blk.parameters():
        p.grad = None
    return out.detach().float(), xi.grad.detach().clone(), grads


def cmp_runs(a, b):
    o, dx, gr = (cos(a[0], b[0]), cos(a[1], b[1]), {n: cos(a[2][n], b[2][n]) for n in a[2]})
    worst = min(gr.items(), key=lambda kv: kv[1])
    return {"out_cos": o, "dx_cos": dx, "dW_cos_min": worst[1], "dW_cos_min_name": worst[0], "bitwise": bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            and all(torch.equal(a[2][n], b[2][n]) for n in a[2])), "out_rel": rel(a[0], b[0]), "dx_rel": rel(a[1], b[1])}


def test_e(model):
    ctx = build_ctx(model, "")
    handle = ctx.meta["round20_fp4_probe"]
    prev = ctx.meta["_prev"]
    for joint, i in ((False, 1), (True, len(ctx.model.backbone.blocks) - 1)):
        x, rope = block_inputs(ctx, joint=joint)
        wout = torch.randn(x.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
        prev["block"].mx8_weights.refresh()
        handle.begin_step()
        a = run_block(ctx, prev["block"], i, x, rope, wout)
        b = run_block(ctx, ctx.ops.block, i, x, rope, wout)
        rec(f"e_block_empty_vs_r17_{model}_{'joint' if joint else 'plain'}", **cmp_runs(a, b))
        assert cmp_runs(a, b)["out_cos"] >= 0.99999 and cmp_runs(a, b)["dW_cos_min"] >= 0.99999
    # head: r14 predict vs ours (policy "")
    mdl = ctx.train_model.model
    B = ctx.meta["batch_size"]
    nq = mdl.num_queries
    N = nq + 1 + mdl.backbone.n_storage_tokens + 1024
    g = torch.Generator(device=dev).manual_seed(11)
    xh = (torch.randn(B, N, mdl.backbone.embed_dim, device=dev, generator=g)).contiguous()
    names = {id(p): n for n, p in ctx.model.named_parameters()}
    hp = [p for n, p in ctx.model.named_parameters() if n.startswith("upscale") or n.startswith("mask_head") or n.startswith("class_head")]

    def run_head(pred):
        for p in hp:
            p.grad = None
        xi = xh.clone().requires_grad_(True)
        with ctx.fabric.autocast():
            ml, cl = pred(ctx, xi, (32, 32))
        wm = torch.randn(ml.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(3))
        (ml.float() * wm).sum().backward(retain_graph=True)
        (cl.float() * 0.01).sum().backward()
        gr = {names[id(p)]: p.grad.detach().clone() for p in hp if p.grad is not None}
        for p in hp:
            p.grad = None
        return ml.detach().float(), xi.grad.detach().clone(), gr, cl.detach().float()

    a, b = run_head(prev["predict"]), run_head(ctx.ops.predict)
    gc_ = {n: cos(a[2][n], b[2][n]) for n in a[2]}
    rec(f"e_head_empty_vs_r14_{model}", mask_cos=cos(a[0], b[0]), dx_cos=cos(a[1], b[1]), param_cos_min=min(gc_.values()), cls_cos=cos(a[3], b[3]),
        bitwise=bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and all(torch.equal(a[2][n], b[2][n]) for n in a[2])))
    assert cos(a[0], b[0]) >= 0.99999 and min(gc_.values()) >= 0.99999
    # head FP4 sanity (policy switch at run time)
    for pol in ("head.dw=nv4+sr", "head.f=mx8,head.dw=nv4+sr+rht", "head.dw=mx8"):
        handle.set_policy(pol)
        c = run_head(ctx.ops.predict)
        gcc = {n: cos(a[2][n], c[2][n]) for n in a[2]}
        rec(f"e_head_policy_{model}_{pol}", mask_cos=cos(a[0], c[0]), dx_cos=cos(a[1], c[1]), param_cos_min=min(gcc.values()), param_cos_min_name=min(gcc.items(), key=lambda kv: kv[1])[0])
    handle.set_policy("")
    return ctx


def test_f(ctx, model):
    handle = ctx.meta["round20_fp4_probe"]
    from common.ops import default_ops
    ref_block = default_ops().block
    pols = ["all.d=nv4+sr,all.w=nv4+sr+rht", "all.d=nv4+sr,all.w=nv4+sr+rht,prefix", "all.f=nv4"]
    for joint, i in ((False, 0), (False, 1), (True, len(ctx.model.backbone.blocks) - 1)):
        x, rope = block_inputs(ctx, joint=joint)
        wout = torch.randn(x.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
        handle.set_policy("")
        handle.begin_step()
        try:
            handle.set_policy("")
            a = run_block(ctx, ref_block, i, x, rope, wout, autocast=True)  # R0's default torch block under bf16 autocast (an fp32 block without autocast has no SDPA kernel here)
            refname = "r0_default_block_bf16_autocast"
        except Exception as e:  # noqa: BLE001
            print("fp32 default block failed:", type(e).__name__, str(e)[:160], flush=True)
            handle.set_policy("all.fdw=bf16")
            a = run_block(ctx, ctx.ops.block, i, x, rope, wout)
            refname = "bf16_policy_block"
        handle.set_policy("")
        handle.begin_step()
        base = run_block(ctx, ctx.ops.block, i, x, rope, wout)
        rec(f"f_{model}_blk{i}_{'joint' if joint else 'plain'}_FINAL_vs_{refname}", **{k: v for k, v in cmp_runs(a, base).items() if k != "bitwise"})
        for pol in pols:
            handle.set_policy(pol)
            handle.begin_step()
            b = run_block(ctx, ctx.ops.block, i, x, rope, wout)
            handle.set_policy("")
            rec(f"f_{model}_blk{i}_{'joint' if joint else 'plain'}_{pol}_vs_{refname}", **{k: v for k, v in cmp_runs(a, b).items() if k != "bitwise"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b,c,d,e,f")
    ap.add_argument("--model", default="vitb16")
    args = ap.parse_args()
    only = set(args.only.split(","))
    out = ROUND / "profile" / "test_fp4_block.json"
    if out.exists():
        RES.update(json.loads(out.read_text()))
    with bench.gpu_lock():
        from round16_full_graph import run as r16
        r16.wait_exclusive()
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        if "a" in only:
            test_a()
        if "b" in only:
            test_b()
        if "c" in only:
            test_c()
        if "d" in only:
            test_d()
        ctx = None
        if only & {"e", "f"}:
            ctx = test_e(args.model) if "e" in only else build_ctx(args.model, "")
            if "f" in only:
                test_f(ctx, args.model)
        print(f"done in {time.time() - t0:.0f}s", flush=True)
    out.write_text(json.dumps(RES, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
