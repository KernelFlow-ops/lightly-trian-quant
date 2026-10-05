"""R20a part 4: per-GEMM relative L2 error (vs the fp32 product of the unquantized operands) of mx8 / nv4 / mx4 schemes on REAL captured tensors.

Capture: one real batch, eager bf16-mixed reference step (R0 modules + Fabric bf16-mixed autocast, no patches), forward-pre / full-backward hooks on
nn.Linear (qkv, proj, fc1, fc2 of blocks {0, 11, 22} for ViT-L and {0, 6, 10} for ViT-B; 22 / 10 are joint blocks) and on the head ConvTranspose2d (k2s2, viewed as
the GEMM [M,C] x [C,4C]); first and last predict call.  X = bf16 input, dY = bf16 grad-output, W = fp32 master weight.  (FINAL's fused blocks have no per-Linear
module boundary and its X / dY are these tensors up to the bf16 / MXFP8 noise of the pipeline, so the capture uses the reference modules.)

Schemes use fp4_ref (the bit-exact spec), the operand products are formed from the dequantized values in fp32.
  fwd   Y = X W^T      operands: X [M,K] along K, W [N,K] along K          (RHT along K on both)
  dgrad dX = dY W      operands: dY [M,N] along N, W^T [K,N] along N       (RHT along N on both; dY SR or RN)
  wgrad dW = dY^T X    operands: dY^T [N,Mp], X^T [K,Mp] along tokens      (RHT along tokens on both; dY SR or RN)
mx8 follows the r17 rule (E4M3, ue8m0 1x32 RCEIL; W / W^T 1D blocks along the contraction axis; token axis zero padded to 128).
"""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4_ref as R
from common.bench import gpu_lock

dev = "cuda"
SEL = {"vitl16": dict(blocks=(0, 11, 22), joint=(21, 22, 23)), "vitb16": dict(blocks=(0, 6, 10), joint=(9, 10, 11))}
LIN = {"qkv": r"attn\.qkv$", "proj": r"attn\.proj$", "fc1": r"mlp\.fc1$", "fc2": r"mlp\.fc2$"}


# ------------------------------------------------------------------------------------------------ capture
def capture(model_name, nbatch_seed=0):
    from common.harness import RunCfg, build, preload_batches, train_step

    cfg = RunCfg(model=model_name, precision="bf16-mixed", patches=(), seed=nbatch_seed)
    ctx = build(cfg)
    ctx.meta["enable_autocast"] = True
    batch = preload_batches(ctx, k=1)[0]
    store = {}
    handles = []
    sel = SEL[model_name]
    names = [n for n, m in ctx.model.named_modules() if isinstance(m, torch.nn.Linear)]
    print("linear names sample:", names[:6], len(names), flush=True)
    for name, mod in ctx.model.named_modules():
        mb = re.search(r"blocks\.(\d+)\.", name)
        if isinstance(mod, torch.nn.Linear) and mb and int(mb.group(1)) in sel["blocks"]:
            for lin, pat in LIN.items():
                if re.search(pat, name):
                    key = f"block{mb.group(1)}.{lin}"

                    def pre(m, inp, key=key):
                        x = inp[0].detach()
                        store.setdefault(key, {})["X"] = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu()
                        store[key]["W"] = m.weight.detach().float().cpu()

                    def bwd(m, gin, gout, key=key):
                        g = gout[0].detach()
                        store[key]["dY"] = g.reshape(-1, g.shape[-1]).to(torch.bfloat16).cpu()

                    handles.append(mod.register_forward_pre_hook(pre))
                    handles.append(mod.register_full_backward_hook(bwd))
        if isinstance(mod, torch.nn.ConvTranspose2d) and mod.kernel_size == (2, 2):
            cnt = {"n": 0}
            key0 = f"head.{name}"

            def fpre(m, inp, key0=key0, cnt=cnt):
                cnt["n"] += 1
                c = cnt["n"]
                if c in (1, 4):
                    x = inp[0].detach()  # [B,C,h,w]
                    B, C, h, w = x.shape
                    k = f"{key0}#call{c}"
                    store.setdefault(k, {})["X"] = x.permute(0, 2, 3, 1).reshape(B * h * w, C).to(torch.bfloat16).cpu()
                    store[k]["W"] = m.weight.detach().float().permute(2, 3, 1, 0).reshape(4 * C, C).cpu()  # Wm[(dy*2+dx)*C+co, ci]
                    store[k]["_dims"] = (B, C, h, w)
                    cnt["k"] = k
                else:
                    cnt["k"] = None

            def fbwd(m, gin, gout, key0=key0, cnt=cnt):
                # backward calls arrive in reverse order of the forward calls: call 4 first, call 1 last
                cnt.setdefault("b", 0)
                cnt["b"] += 1
                c = 5 - cnt["b"]
                k = f"{key0}#call{c}"
                if c in (1, 4) and k in store:
                    g = gout[0].detach()  # [B,C,2h,2w]
                    B, C, h, w = store[k]["_dims"]
                    store[k]["dY"] = g.reshape(B, C, h, 2, w, 2).permute(0, 2, 4, 3, 5, 1).reshape(B * h * w, 4 * C).to(torch.bfloat16).cpu()

            handles.append(mod.register_forward_pre_hook(fpre))
            handles.append(mod.register_full_backward_hook(fbwd))
    out = train_step(ctx, batch, 0)
    for h in handles:
        h.remove()
    info = {"loss": float(out.loss), "n_captured": len(store), "keys": sorted(store)}
    del ctx, batch
    gc.collect()
    torch.cuda.empty_cache()
    return store, info


# ------------------------------------------------------------------------------------------------ mx8 (r17 rule) fake quant
def mx8_deq(x: torch.Tensor) -> torch.Tensor:
    """E4M3 + ue8m0 1x32 RCEIL (QuACK to_mx rceil == r17 quantizer), dequantized fp32.  x [R, C], C % 32 == 0."""
    from quack.blockscaled.quantize import to_mx

    q, s = to_mx(x.float().contiguous(), 32, "rceil")
    sv = torch.exp2(s.view(torch.uint8).float() - 127.0)
    return q.float() * sv.repeat_interleave(32, 1)


def deq(q, sf, ts, fmt, rows, cols):
    return R.dequant_copy(q, sf, ts, fmt, rows, cols)


def operand_rows(x, fmt, *, rht=False, sr=False, gen=None, signs=None):
    """x [R, C] quantized along C -> dequantized fp32 [R, C] (RHT-rotated domain if rht)."""
    if fmt == "mx8":
        return mx8_deq(x)
    q, sf, ts = R.quantize_matrix(x, fmt, signs=signs if rht else None, sr=sr, gen=gen)
    return deq(q, sf, ts, fmt, x.shape[0], x.shape[1])


def weight_rows(w, fmt, *, two_d, rht, signs, along="K"):
    """W [N,K] along K ('K') or W^T along N ('N'); returns dequantized fp32 [N,K] / [K,N]."""
    if fmt == "mx8":
        return mx8_deq(w if along == "K" else w.t().contiguous())
    t = R.quantize_weight(w, fmt, two_d=two_d, rht_row=(rht and along == "K"), rht_col=(rht and along == "N"), signs=signs)
    if along == "K":
        return R.dequant_row(t)
    return R.dequant_col(t)  # [K, N] (rows = K, padded rows = none since N % 128 == 0)


def operand_tokens(x, fmt, *, rht=False, sr=False, gen=None, signs=None):
    """x [R, C] -> dequantized X^T [C, Rpad] (quantized along the zero padded token axis)."""
    Rr, C = x.shape
    Rp = R.pad128(Rr)
    xt = torch.nn.functional.pad(x.float(), (0, 0, 0, Rp - Rr)).t().contiguous()
    if fmt == "mx8":
        return mx8_deq(xt)
    q, sf, ts = R.quantize_matrix(xt, fmt, signs=signs if rht else None, sr=sr, gen=gen)
    return deq(q, sf, ts, fmt, C, Rp)


def rel(o, r):
    return float((o - r).norm() / r.norm().clamp_min(1e-30))


# scheme name -> kwargs.  fmt in {mx8, nv4, mx4}; sr: dY stochastic rounding; rht: Hadamard on the contraction axis; w2d: 2D weight scales (fwd / dgrad only)
SCHEMES = {
    "fwd": [("mx8", dict(fmt="mx8")),
            ("nv4_rn_w2d", dict(fmt="nv4", w2d=True)), ("nv4_rn_w1d", dict(fmt="nv4", w2d=False)), ("nv4_rht_w2d", dict(fmt="nv4", w2d=True, rht=True)),
            ("mx4_rn_w2d", dict(fmt="mx4", w2d=True)), ("mx4_rn_w1d", dict(fmt="mx4", w2d=False)), ("mx4_rht_w2d", dict(fmt="mx4", w2d=True, rht=True))],
    "dgrad": [("mx8", dict(fmt="mx8")),
              ("nv4_dYrn_w2d", dict(fmt="nv4", w2d=True)), ("nv4_dYsr_w2d", dict(fmt="nv4", w2d=True, sr=True)), ("nv4_dYsr_w1d", dict(fmt="nv4", w2d=False, sr=True)),
              ("nv4_rht_dYrn_w2d", dict(fmt="nv4", w2d=True, rht=True)), ("nv4_rht_dYsr_w2d", dict(fmt="nv4", w2d=True, rht=True, sr=True)),
              ("mx4_dYrn_w2d", dict(fmt="mx4", w2d=True)), ("mx4_dYsr_w2d", dict(fmt="mx4", w2d=True, sr=True)), ("mx4_dYsr_w1d", dict(fmt="mx4", w2d=False, sr=True)),
              ("mx4_rht_dYrn_w2d", dict(fmt="mx4", w2d=True, rht=True)), ("mx4_rht_dYsr_w2d", dict(fmt="mx4", w2d=True, rht=True, sr=True))],
    "wgrad": [("mx8", dict(fmt="mx8")),
              ("nv4_dYrn", dict(fmt="nv4")), ("nv4_dYsr", dict(fmt="nv4", sr=True)),
              ("nv4_rht_dYrn", dict(fmt="nv4", rht=True)), ("nv4_rht_dYsr", dict(fmt="nv4", rht=True, sr=True)),
              ("mx4_dYrn", dict(fmt="mx4")), ("mx4_dYsr", dict(fmt="mx4", sr=True)),
              ("mx4_rht_dYrn", dict(fmt="mx4", rht=True)), ("mx4_rht_dYsr", dict(fmt="mx4", rht=True, sr=True))],
}


def evaluate(X, dY, W, signs, seed=0):
    """X [M,K] bf16, dY [M,N] bf16, W [N,K] fp32 (all on GPU) -> {kind: {scheme: relL2}}."""
    Xf, dYf = X.float(), dY.float()
    ref = {"fwd": Xf @ W.t(), "dgrad": dYf @ W, "wgrad": dYf.t() @ Xf}
    out = {}
    for kind, schemes in SCHEMES.items():
        out[kind] = {}
        for name, p in schemes:
            fmt, sr, rht, w2d = p["fmt"], p.get("sr", False), p.get("rht", False), p.get("w2d", True)
            gen = R.make_gen(dev, 1000 + seed)
            A = B = None
            try:
                if kind == "fwd":
                    A = operand_rows(Xf, fmt, rht=rht, signs=signs)
                    B = weight_rows(W, fmt, two_d=w2d, rht=rht, signs=signs, along="K")
                    o = A @ B.t()
                elif kind == "dgrad":
                    A = operand_rows(dYf, fmt, rht=rht, sr=sr, gen=gen, signs=signs)
                    B = weight_rows(W, fmt, two_d=w2d, rht=rht, signs=signs, along="N")  # [K, N]
                    o = A @ B.t()
                else:
                    A = operand_tokens(dYf, fmt, rht=rht, sr=sr, gen=gen, signs=signs)  # [N, Mp]
                    B = operand_tokens(Xf, fmt, rht=rht, signs=signs)  # [K, Mp]
                    o = A @ B.t()
                out[kind][name] = rel(o, ref[kind])
            except Exception as e:  # noqa: BLE001
                out[kind][name] = f"err {type(e).__name__}: {str(e)[:120]}"
            del A, B
        torch.cuda.empty_cache()
    return out


def main():
    model = sys.argv[1]
    outp = ROUND / "profile" / f"fp4_error_table_{model}.json"
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"model": model, "layers": {}}
    with gpu_lock():
        store, info = capture(model)
        torch.set_float32_matmul_precision("highest")  # build() sets "high" (TF32); the fp32 references must be exact
        torch.backends.cuda.matmul.allow_tf32 = False
        res["capture"] = info
        if "--save" in sys.argv:
            torch.save({k: {a: b for a, b in v.items() if a != "_dims"} for k, v in store.items()}, f"/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/cap_{model}.pt")
        print("captured", info["n_captured"], "tensors; loss", info["loss"], flush=True)
        signs = R.rht_signs(device=dev)
        for key in sorted(store):
            d = store[key]
            if not all(k in d for k in ("X", "dY", "W")):
                res["layers"][key] = {"missing": [k for k in ("X", "dY", "W") if k not in d]}
                continue
            X, dY, W = d["X"].to(dev), d["dY"].to(dev), d["W"].to(dev)
            stats = {"M": X.shape[0], "K": X.shape[1], "N": dY.shape[1], "X_absmax": float(X.float().abs().max()), "X_rms": float(X.float().pow(2).mean().sqrt()),
                     "dY_absmax": float(dY.float().abs().max()), "dY_rms": float(dY.float().pow(2).mean().sqrt()), "W_absmax": float(W.abs().max())}
            r = evaluate(X, dY, W, signs)
            res["layers"][key] = {"stats": stats, "relL2": r}
            print(key, stats["M"], stats["N"], stats["K"], {k: {s: (round(v, 4) if isinstance(v, float) else v) for s, v in vv.items() if s in ("mx8", "nv4_rn_w2d", "nv4_dYrn_w2d", "nv4_dYrn", "nv4_rht_dYsr", "nv4_rht_dYsr_w2d")} for k, vv in r.items()}, flush=True)
            outp.write_text(json.dumps(res, indent=1))
            del X, dY, W
            torch.cuda.empty_cache()
    print("wrote", outp)


if __name__ == "__main__":
    main()
