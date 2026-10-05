"""Round 14 step 2 stack: r14_nv4 / r14_mx4 on the own block-scaled FP4 kernel (fp4g_core.cu) with fused producers / epilogues.

  per layer (9 kernels, attention is the r12 fp16 kernel):
    lnq        LN of the fp32 residual stream + FP4 quantisation                              (lnq14.cu)
    qkv GEMM   own kernel, EPI 2: bias + RoPE(fp32) + [3,B,N,H,64] fp16 layout -> attention  (no separate RoPE / layout pass)
    attention  round-12 attn16o (fp16 in / fp16 out), layers 0-1 with the 1/8 Q pre-scale
    quantx     attention output fp16 -> FP4                                                   (lnq14.cu)
    proj GEMM  own kernel, EPI 1: gamma * (acc + bias) + fp32 residual
    lnq        LN2 + FP4
    fc1 GEMM   own kernel, EPI 3: bias + GELU + FP4 quantize-out (writes fc2's operand + blocked scales directly)
    fc2 GEMM   own kernel, EPI 1
  head: quantx(z) -> own kernel EPI 0 (bf16 out) -> round-11 dw_ln, for both ConvT stages.
Per-tensor scales (nv4 only) are static: measured on the first (eager) call, headroom 2, never synced to the host.
GEMM tiles come from tuned14.json (written by the tile sweeps), default 128x128x64B / 4 stages.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

import r11_common as c            # sys.path for round5 / 7 / 9 / 11 (read-only imports)
import head11
import patch_embed as pe_mod
import r11_model
from round5_gemm_sm120.impl import ops as r5

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import ext14  # noqa: E402
import fp4gb  # noqa: E402
import fp4util as U  # noqa: E402

R = U.R
F16, BF = torch.float16, torch.bfloat16
DEFAULT_CFG = dict(bm=128, bn=128, bkb=64, st=4, wm=2, wn=4, minb=1)
_TUNED_PATH = Path(os.environ.get("R14_TUNED", str(_HERE / "tuned14.json")))
_TUNED = json.loads(_TUNED_PATH.read_text()) if _TUNED_PATH.exists() else {}


def cfg_for(fmt, kind, model):
    return dict(_TUNED.get(f"{model}|{fmt}|{kind}", _TUNED.get(f"{fmt}|{kind}", DEFAULT_CFG)))


def set_ts(ts_dev: torch.Tensor, x: torch.Tensor):
    """static per-tensor scale of an nv4 activation site: HEADROOM * amax / 2688 (device side, no host sync)."""
    amax = x.float().abs().amax()
    ts_dev.copy_(torch.where(amax > 0, amax * (U.HEADROOM / 2688.0), torch.ones_like(amax)).reshape(1))


class Lin:
    """Weight [N, K] quantised once along K (1D block scales, exact per-tensor scale) + the activation-scale product for the GEMM epilogue."""

    def __init__(self, W, fmt, dev):
        self.N, self.K = W.shape
        self.q, self.sf, ts = R.quantize_matrix(W.float().contiguous(), fmt)
        self.ts = ts
        self.tsp = None if ts is None else ts.clone()          # ts_act * ts_w (nv4), set on the calibration call

    def calibrate(self, ts_act):
        if self.ts is not None:
            self.tsp.copy_(ts_act * self.ts)


class Layer14b:
    def __init__(self, block, fmt, model_name):
        w = r5.prepare_block_weights(block, ln_fold=False)
        dev = w["Wqkv"].device
        self.fmt, self.fi, self.nv4 = fmt, 0 if fmt == "nv4" else 1, fmt == "nv4"
        self.ln1 = (w["ln1_w"], w["ln1_b"], w["ln1_eps"])
        self.ln2 = (w["ln2_w"], w["ln2_b"], w["ln2_eps"])
        self.qkv, self.proj, self.fc1, self.fc2 = (Lin(w[k], fmt, dev) for k in ("Wqkv", "Wproj", "W1", "W2"))
        f = lambda t: t.float().contiguous()  # noqa: E731
        self.bqkv, self.bproj, self.b1, self.b2, self.g1, self.g2 = f(w["bqkv"]), f(w["bproj"]), f(w["b1"]), f(w["b2"]), f(w["g1"]), f(w["g2"])
        mk = (lambda: torch.ones(1, device=dev)) if self.nv4 else (lambda: None)
        self.ts_ln1, self.ts_o, self.ts_ln2, self.ts_g = mk(), mk(), mk(), mk()
        kern = lambda kind, epi: fp4gb.get(dict(cfg_for(fmt, kind, model_name), fmt=self.fi, epi=epi))  # noqa: E731
        self.k_qkv, self.k_proj, self.k_fc1, self.k_fc1_dbg, self.k_fc2 = kern("qkv", 2), kern("proj", 1), kern("fc1", 3), kern("fc1", 4), kern("fc2", 1)


def block14b(x, L: Layer14b, sin, cos, prefix, heads, attn_pre, calib):
    B, N, d = x.shape
    M = B * N
    ff = L.fc1.N
    dev = x.device
    e, at, ln = ext14.lnq_ext(), ext14.attn_f16_ext(), ext14.ln_ext()
    fi, nv4, cal = L.fi, L.nv4, calib.on
    vec4 = 64 if nv4 else 128

    def act(K):
        return (torch.empty(M, K // 2, dtype=torch.uint8, device=dev), torch.empty((M + 127) // 128, K // vec4, 32, 4, 4, dtype=torch.uint8, device=dev))

    # ---- LN1 + quant -> qkv (bias + RoPE + layout)
    if cal and nv4:
        set_ts(L.ts_ln1, ln.ln_bf16(x, L.ln1[0], L.ln1[1], L.ln1[2]))
        L.qkv.calibrate(L.ts_ln1)
    q, sf = act(d)
    e.lnq(x, L.ln1[0], L.ln1[1], L.ln1[2], L.ts_ln1, q, sf, fi)
    qkv = torch.empty(3, B, N, heads, 64, dtype=F16, device=dev)
    L.k_qkv(q, L.qkv.q, sf, L.qkv.sf, M=M, N=3 * d, K=d, ts=L.qkv.tsp, bias=L.bqkv, C=qkv, sin=sin, cos=cos, Ntok=N, prefix=prefix, d=d, H=heads)
    o = at.attn16_fwd(qkv, 4, attn_pre)                                    # [B,N,d] fp16
    # ---- attention output quant -> proj (gamma + residual)
    if cal and nv4:
        set_ts(L.ts_o, o)
        L.proj.calibrate(L.ts_o)
    q, sf = act(d)
    e.quantx(o.view(M, d), L.ts_o, q, sf, fi)
    xn = torch.empty(M, d, dtype=torch.float32, device=dev)
    L.k_proj(q, L.proj.q, sf, L.proj.sf, M=M, N=d, K=d, ts=L.proj.tsp, bias=L.bproj, gamma=L.g1, R=x.view(M, d), C=xn)
    x = xn.view(B, N, d)
    # ---- LN2 + quant -> fc1 (GELU + quantize-out) -> fc2 (gamma + residual)
    if cal and nv4:
        set_ts(L.ts_ln2, ln.ln_bf16(x, L.ln2[0], L.ln2[1], L.ln2[2]))
        L.fc1.calibrate(L.ts_ln2)
    q, sf = act(d)
    e.lnq(x, L.ln2[0], L.ln2[1], L.ln2[2], L.ts_ln2, q, sf, fi)
    if cal and nv4:
        g = torch.empty(M, ff, dtype=BF, device=dev)
        L.k_fc1_dbg(q, L.fc1.q, sf, L.fc1.sf, M=M, N=ff, K=d, ts=L.fc1.tsp, bias=L.b1, C=g)
        set_ts(L.ts_g, g)
        L.fc2.calibrate(L.ts_g)
    qg, sfg = act(ff)
    L.k_fc1(q, L.fc1.q, sf, L.fc1.sf, M=M, N=ff, K=d, ts=L.fc1.tsp, bias=L.b1, Qo=qg, SFo=sfg, tso=L.ts_g)
    xn = torch.empty(M, d, dtype=torch.float32, device=dev)
    L.k_fc2(qg, L.fc2.q, sfg, L.fc2.sf, M=M, N=d, K=ff, ts=L.fc2.tsp, bias=L.b2, gamma=L.g2, R=x.view(M, d), C=xn)
    return xn.view(B, N, d)


class FastHead14b(head11.FastHead11):
    """round-11 head with both ConvT GEMMs on the own FP4 kernel (quantx of the bf16 input, plain GEMM with bf16 out -> dw_ln unchanged)."""

    def __init__(self, model, fmt, calib, model_name):
        super().__init__(model)
        dev = self.stages[0]["wr"].device
        self.fmt, self.fi, self.nv4, self.calib = fmt, 0 if fmt == "nv4" else 1, fmt == "nv4", calib
        self.lin = [Lin(st["wr"].float().t().contiguous(), fmt, dev) for st in self.stages]       # [4C, C]
        self.ts = [torch.ones(1, device=dev) if self.nv4 else None for _ in self.stages]
        self.k = [fp4gb.get(dict(cfg_for(fmt, "ct1" if i == 0 else "ct2", model_name), fmt=self.fi, epi=0)) for i in range(len(self.stages))]

    def upscale_nhwc(self, x_tok, B, Hs, Ws):
        z = x_tok
        e = ext14.lnq_ext()
        vec4 = 64 if self.nv4 else 128
        for i, st in enumerate(self.stages):
            M, C = z.shape
            lin = self.lin[i]
            if self.calib.on and self.nv4:
                set_ts(self.ts[i], z)
                lin.calibrate(self.ts[i])
            q = torch.empty(M, C // 2, dtype=torch.uint8, device=z.device)
            sf = torch.empty((M + 127) // 128, C // vec4, 32, 4, 4, dtype=torch.uint8, device=z.device)
            e.quantx(z, self.ts[i], q, sf, self.fi)
            g = torch.empty(M, 4 * C, dtype=BF, device=z.device)
            self.k[i](q, lin.q, sf, lin.sf, M=M, N=4 * C, K=C, ts=lin.tsp, C=g)
            out = torch.empty((B, 2 * Hs, 2 * Ws, self.C), dtype=BF, device=x_tok.device)
            self.ext.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, self.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        return out


def make_model_fn_b(name, model, fmt, flags):
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    calib = U.Calib()
    L = [Layer14b(b, fmt, name) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    sin, cos = sin.float().contiguous(), cos.float().contiguous()
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks
    pe = pe_mod.PatchEmbedGemm(bb, **r11_model.PCFG[0])
    fh = FastHead14b(model, fmt, calib, name)

    def model_fn(x):
        if calib.on and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("r14: the first call (static-scale calibration) must run eagerly, before CUDA graph capture")
        t = pe(x)
        for i in range(nb):
            if i == first_joint:
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = block14b(t, L[i], sin, cos, t.shape[1] - 1024, heads, 0.125 if i <= 1 else 1.0, calib)
        out = fh.predict_tail(t, grid)
        calib.on = False
        return out

    model_fn._keepalive = (L, pe, fh, calib)
    return model_fn
