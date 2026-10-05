"""Round 10 precision emulation of the r5_v1 backbone + fused head.

`EmuNet(name, model)` re-implements `round5_gemm_sm120/r5_model_impls.py::make_model_fn_r5(name, model, "r5_v1")` layer by layer (same R5 kernels and
epilogues: LN kernel, qkv+RoPE, flash SDPA, proj/fc2 gamma+fp32 residual, fc1 GELU, fused R7 head) but lets every GEMM (qkv / proj / fc1 / fc2 per layer,
head ConvT1 / ConvT2 / mask einsum) run in a different precision tier chosen by a `plan`:

    bf16   P0  the original path (R5 kernels, bf16 operands, fp32 accumulate)
    fp16   P1  the real `mg_f16acc` prototype kernel (fp16 operands, fp16 MMA accumulate promoted to fp32 every k-tile, R5 epilogues), power-of-two pre-scale
    mx8 / int8 / nv4 / mx4   P2/P3  X and W are quantised -> dequantised to bf16 (`quant_ref.qdq`; the values are exactly representable in bf16, except the
                              nv4 per-tensor scale / int8 scales which are rounded to bf16 -- error << quantisation error), then the SAME R5 kernel
                              runs, so only the accumulation order differs from a real block-scaled tensor-core GEMM.
Techniques per GEMM (Spec): SmoothQuant alpha folded into the LN affine and W (qkv, fc1) or into W_v / W_proj (proj), block Hadamard 16/32 along K on X and W,
prefix rows (cls + registers, 105 rows in joint layers) kept on the bf16 path, GPTQ rounding of W (gptq.py).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

ROOT = Path("/root/lightly-train/infer-opt")
HERE = Path(__file__).resolve().parent
for p in (HERE, ROOT, ROOT / "common", ROOT / "round7_head_postprocess" / "impl", ROOT / "round5_gemm_sm120"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import head as hd  # noqa: E402
import quant_ref as qr  # noqa: E402
from round5_gemm_sm120.impl import ops as r5  # noqa: E402

BF, F16 = torch.bfloat16, torch.float16
KINDS = ("qkv", "proj", "fc1", "fc2")
HEAD_KINDS = ("ct1", "ct2", "einsum")
FMTS = ("bf16", "fp16", "mx8", "int8", "nv4", "mx4")


@dataclass(frozen=True)
class Spec:
    fmt: str = "bf16"
    sq: float = 0.0       # SmoothQuant alpha (0 = off); qkv, fc1 (LN affine fold), proj (W_v fold)
    had: int = 0          # block Hadamard size along K (0 = off)
    pfx: bool = False     # prefix rows stay on the bf16 path
    gptq: bool = False    # GPTQ rounding of W (block scales fixed from the running weights)
    wfmt: str = ""        # weight format (default = fmt)
    pa: int = 0           # fp16 tier: X pre-scale 2^-pa
    pw: int = 0           # fp16 tier: W pre-scale 2^-pw
    seed: int = 0         # randomised block Hadamard: 0 = plain, k > 0 = +-1 sign vector from quant_ref.hadamard_signs(K, k) (noise-realisation robustness checks)

    def key(self):
        return (self.fmt, self.sq, self.had, self.pfx, self.gptq, self.wfmt or self.fmt, self.pa, self.pw, self.seed)

    @property
    def wf(self):
        return self.wfmt or self.fmt


P0 = Spec()


def layers_of(sel, nb: int, nj: int = 3) -> list[int]:
    if isinstance(sel, (list, tuple)):
        return list(sel)
    return {"all": list(range(nb)), "nofirst": list(range(1, nb)), "nojoint": list(range(nb - nj)),
            "nofirst_nojoint": list(range(1, nb - nj)), "joint": list(range(nb - nj, nb)), "first": [0], "last": [nb - 1]}[sel]


def make_plan(entries: list[dict], nb: int) -> dict:
    """entries: [{"kinds": [...], "layers": "all"|list, "spec": {Spec fields}}] -> {(layer, kind): Spec, ("head", kind): Spec}"""
    plan = {}
    for e in entries:
        spec = Spec(**e["spec"])
        for kind in e["kinds"]:
            if kind in HEAD_KINDS:
                plan[("head", kind)] = spec
            else:
                for i in layers_of(e.get("layers", "all"), nb):
                    plan[(i, kind)] = spec
    return plan


class Calib:
    """per (layer, kind) activation statistics of the P0 model on the calibration frames."""

    def __init__(self):
        self.cmax: dict = {}      # channel absmax over non-prefix rows [K]
        self.cmax_all: dict = {}  # over all rows
        self.tmax: dict = {}      # tensor absmax (all rows)
        self.hess: dict = {}      # X^T X (fp32) over non-prefix rows, for GPTQ
        self.n_hess: dict = {}

    def update(self, key, X2d: torch.Tensor, pmask: torch.Tensor, hess: bool):
        Xf = X2d.float()
        a = Xf.abs()
        cm = a[~pmask].amax(0)
        ca = a.amax(0)
        for d, v in ((self.cmax, cm), (self.cmax_all, ca)):
            d[key] = v if key not in d else torch.maximum(d[key], v)
        t = a.amax()
        self.tmax[key] = t if key not in self.tmax else torch.maximum(self.tmax[key], t)
        if hess:
            Xn = Xf[~pmask]
            h = Xn.t() @ Xn
            if key in self.hess:
                self.hess[key] += h
                self.n_hess[key] += Xn.shape[0]
            else:
                self.hess[key] = h
                self.n_hess[key] = Xn.shape[0]


def _nbytes(x):
    return sum(t.numel() * t.element_size() for t in x) if isinstance(x, tuple) else x.numel() * x.element_size()


class EmuNet:
    def __init__(self, name: str, model, calib: Optional[Calib] = None, wcache_gb: float = 12.0):
        self.name, self.model = name, model
        hd.enable(model)                       # R7 fused head (bf16), same as make_model_fn_r5
        self.bb = bb = model.backbone
        self.heads = bb.blocks[0].attn.num_heads
        self.nb = len(bb.blocks)
        self.nj = model.num_joint_blocks
        self.first_joint = self.nb - self.nj
        self.W5 = [r5.prepare_block_weights(b, ln_fold=False) for b in bb.blocks]
        self.grid = (512 // bb.patch_size, 512 // bb.patch_size)
        with torch.no_grad():
            self.sin, self.cos = bb.rope_embed(H=self.grid[0], W=self.grid[1])
        self.queries = model.queries.weight.detach().float()
        self.d = bb.blocks[0].attn.qkv.in_features
        self.calib = calib
        self.plan: dict = {}
        self.collect: Optional[Callable] = None
        self._lay: dict = {}
        self._wq: dict = {}
        self._wbytes = 0
        self._wlimit = int(wcache_gb * 2**30)
        self._pm: dict = {}
        self.gptq_fn = None            # callable(i, kind, Wm_f32, spec, ml) -> dequantised W fp32; set by gptq.py
        model._predict = self._head_predict
        self._fh = model._fast_head

    def set_plan(self, plan: dict):
        self.plan = plan
        self._lay.clear()

    # ------------------------------------------------------------------ SmoothQuant scales and per-layer effective weights
    def _sq_scale(self, i: int, kind: str, alpha: float, Wref: torch.Tensor) -> torch.Tensor:
        cm = self.calib.cmax[(i, kind)].clamp(min=1e-5)
        wm = Wref.float().abs().amax(0).clamp(min=1e-5)
        return (cm.pow(alpha) / wm.pow(1.0 - alpha)).clamp(1e-3, 1e3)

    def _wkey(self, i, kind, sp):
        extra = (sp["proj"].sq if sp["proj"].fmt != "bf16" else 0.0) if kind == "qkv" else 0
        return (i, kind, sp[kind].key(), extra)

    def _layer(self, i: int) -> dict:
        """small per-layer dict: LN affine (SmoothQuant folded), qkv bias, bf16 masters when modified, and the (cached) quantised weights."""
        if i in self._lay:
            return self._lay[i]
        plan = self.plan
        sp = {k: plan.get((i, k), P0) for k in KINDS}
        w = self.W5[i]
        d = self.d
        Wqkv, bqkv = w["Wqkv"].float(), w["bqkv"].float()
        Wp, W1, W2 = w["Wproj"].float(), w["W1"].float(), w["W2"].float()
        ln1_w, ln1_b, ln2_w, ln2_b = w["ln1_w"], w["ln1_b"], w["ln2_w"], w["ln2_b"]
        modified = {k: False for k in KINDS}
        if sp["qkv"].sq and sp["qkv"].fmt != "bf16":
            s = self._sq_scale(i, "qkv", sp["qkv"].sq, Wqkv)
            Wqkv = Wqkv * s[None, :]
            ln1_w, ln1_b = ln1_w / s, ln1_b / s
            modified["qkv"] = True
        if sp["fc1"].sq and sp["fc1"].fmt != "bf16":
            s = self._sq_scale(i, "fc1", sp["fc1"].sq, W1)
            W1 = W1 * s[None, :]
            ln2_w, ln2_b = ln2_w / s, ln2_b / s
            modified["fc1"] = True
        if sp["proj"].sq and sp["proj"].fmt != "bf16":
            s = self._sq_scale(i, "proj", sp["proj"].sq, Wp)      # per channel of the attention output o [H*64]
            Wp = Wp * s[None, :]
            Wqkv = Wqkv.clone()
            Wqkv[2 * d:] = Wqkv[2 * d:] / s[:, None]
            bqkv = bqkv.clone()
            bqkv[2 * d:] = bqkv[2 * d:] / s
            modified["proj"] = modified["qkv"] = True
        masters = {"qkv": Wqkv, "proj": Wp, "fc1": W1, "fc2": W2}
        out = dict(ln1_w=ln1_w.contiguous(), ln1_b=ln1_b.contiguous(), ln2_w=ln2_w.contiguous(), ln2_b=ln2_b.contiguous(),
                   bqkv=bqkv.to(BF).contiguous(), mbf={}, wq={})
        for k in KINDS:
            spec = sp[k]
            if spec.fmt == "bf16":
                if modified[k]:
                    out["mbf"][k] = masters[k].to(BF).contiguous()        # e.g. W_v rows rescaled by the proj SmoothQuant fold
                continue
            Wm = masters[k]
            if spec.had:
                Wm = qr.block_hadamard(Wm, spec.had, qr.hadamard_signs(Wm.shape[1], spec.seed, Wm.device) if spec.seed else None)
            if spec.pfx:
                out["mbf"][k] = Wm.to(BF).contiguous()                    # prefix rows: same rotated/smoothed master, no quantisation
            wk = self._wkey(i, k, sp)
            if wk not in self._wq:
                self._wq[wk] = self._quantize(i, k, Wm, spec, out)
                self._wbytes += _nbytes(self._wq[wk])
                while self._wbytes > self._wlimit and len(self._wq) > 1:
                    k0 = next(iter(self._wq))
                    if k0 == wk:
                        break
                    v = self._wq.pop(k0)
                    self._wbytes -= _nbytes(v)
            out["wq"][k] = self._wq[wk]
        self._lay[i] = out
        return out

    def _quantize(self, i: int, kind: str, Wm: torch.Tensor, spec: Spec, ml: dict):
        if spec.fmt == "int8k":            # real mg_i8 kernel path: int8 codes + per-output-channel scale
            import mg8
            return mg8.quant_rows(Wm)
        if spec.fmt == "fp16":
            return (Wm * 2.0 ** -spec.pw).to(F16).contiguous()
        if spec.gptq and self.gptq_fn is not None:
            Wf = self.gptq_fn(i, kind, Wm, spec, ml)
        else:
            Wf = qr.qdq(Wm, spec.wf, dim=-1)
        return Wf.to(BF).contiguous()

    def prefix_mask(self, B: int, N: int, dev) -> torch.Tensor:
        key = (B, N)
        if key not in self._pm:
            self._pm[key] = (torch.arange(B * N, device=dev) % N) < (N - 1024)
        return self._pm[key]

    # ------------------------------------------------------------------ one GEMM (activation transform + R5 / mg16 kernel + epilogue)
    def _kernel_bf16(self, kind, X, Wb, bias, resid, gamma, prefix):
        if kind == "qkv":
            return r5.qkv_rope(X, Wb, bias, self.sin, self.cos, prefix, self.heads).transpose(2, 3)   # physical [3,B,N,H,64]
        if kind == "fc1":
            return r5.gemm_bias_gelu(X, Wb, bias)
        return r5.gemm_bias_residual(X, Wb, bias, resid, gamma)

    def _kernel_int8(self, kind, X8, sx, W8, sw, bias, resid, gamma, prefix, B, N):
        import mg8
        if kind == "qkv":
            return mg8.qkv_rope8(X8, sx, W8, sw, bias, self.sin, self.cos, B, N, prefix, self.heads).transpose(2, 3)      # physical [3,B,N,H,64]
        if kind == "fc1":
            return mg8.gemm_bias_gelu8(X8, sx, W8, sw, bias).view(B, N, -1)
        return mg8.gemm_bias_residual8(X8, sx, W8, sw, bias, resid, gamma)

    def _kernel_fp16(self, kind, X16, W16, bias, resid, gamma, prefix, oscale):
        import mg16
        if kind == "qkv":
            return mg16.qkv_rope16(X16, W16, bias, self.sin, self.cos, prefix, self.heads, oscale=oscale, out_dtype=BF).transpose(2, 3)
        if kind == "fc1":
            return mg16.gemm_bias_gelu16(X16, W16, bias, oscale=oscale, out_dtype=F16)
        return mg16.gemm_bias_residual16(X16, W16, bias, resid, gamma, oscale=oscale)

    def gemm(self, i: int, kind: str, X: torch.Tensor, bias, resid=None, gamma=None):
        """X [B,N,K] (bf16 or fp16) -> output of the kind's epilogue (qkv: physical [3,B,N,H,64] bf16; fc1: [B,N,ff] bf16/fp16; proj/fc2: fp32 residual out)."""
        spec = self.plan.get((i, kind), P0)
        B, N, K = X.shape
        prefix = N - 1024
        if self.collect is not None:
            self.collect(i, kind, X)
        ml = self._layer(i)
        if spec.fmt == "bf16":
            Wb = ml["mbf"].get(kind)
            if Wb is None:
                Wb = self._orig_w(i, kind)
            return self._kernel_bf16(kind, X.to(BF), Wb, bias, resid, gamma, prefix)
        M = B * N
        Xf = X.reshape(M, K).float()
        if spec.had:
            Xf = qr.block_hadamard(Xf, spec.had, qr.hadamard_signs(K, spec.seed, Xf.device) if spec.seed else None)
        Wq = ml["wq"][kind]
        if spec.fmt == "int8k":
            import mg8
            X8, sx = mg8.quant_rows(Xf)
            out = self._kernel_int8(kind, X8, sx, Wq[0], Wq[1], bias, resid, gamma, prefix, B, N)
        elif spec.fmt == "fp16":
            X16 = (Xf * 2.0 ** -spec.pa).to(F16).view(B, N, K)
            out = self._kernel_fp16(kind, X16, Wq, bias, resid, gamma, prefix, 2.0 ** (spec.pa + spec.pw))
        else:
            Xq = qr.qdq(Xf, spec.fmt, dim=-1).to(BF).view(B, N, K)
            out = self._kernel_bf16(kind, Xq, Wq, bias, resid, gamma, prefix)
        if spec.pfx:
            outb = self._kernel_bf16(kind, Xf.to(BF).view(B, N, K), ml["mbf"][kind], bias, resid, gamma, prefix)
            out = self._merge(kind, out, outb, self.prefix_mask(B, N, X.device).view(B, N))
        return out

    def _orig_w(self, i, kind):
        w = self.W5[i]
        return {"qkv": w["Wqkv"], "proj": w["Wproj"], "fc1": w["W1"], "fc2": w["W2"]}[kind]

    @staticmethod
    def _merge(kind, out_q, out_b, pm):
        """rows flagged in pm (prefix tokens, [B,N]) come from the bf16 path."""
        if out_q.dtype != out_b.dtype:
            out_b = out_b.to(out_q.dtype)
        if kind == "qkv":
            return torch.where(pm[None, :, :, None, None], out_b, out_q)
        return torch.where(pm[:, :, None], out_b, out_q)

    # ------------------------------------------------------------------ block / model
    def block(self, i: int, x: torch.Tensor) -> torch.Tensor:
        w = self.W5[i]
        B, N, d = x.shape
        ml = self._layer(i)
        _, h = r5.add_layernorm(x, None, ml["ln1_w"], ml["ln1_b"], w["ln1_eps"])
        qkv = self.gemm(i, "qkv", h, ml["bqkv"])                       # physical [3,B,N,H,64]
        q = qkv.transpose(2, 3)
        if q.dtype != BF:
            q = q.to(BF)
        o = F.scaled_dot_product_attention(q[0], q[1], q[2])
        o = o.transpose(1, 2).reshape(B, N, d)
        x = self.gemm(i, "proj", o, w["bproj"], resid=x, gamma=w["g1"])
        _, h = r5.add_layernorm(x, None, ml["ln2_w"], ml["ln2_b"], w["ln2_eps"])
        a = self.gemm(i, "fc1", h, w["b1"])
        return self.gemm(i, "fc2", a, w["b2"], resid=x, gamma=w["g2"])

    def forward(self, x_img: torch.Tensor):
        bb = self.bb
        with torch.autocast("cuda", dtype=BF, cache_enabled=False):
            t, _ = bb.prepare_tokens_with_masks(x_img)
        t = t.float().contiguous()
        for i in range(self.nb):
            if i == self.first_joint:
                t = torch.cat((self.queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = self.block(i, t)
        with torch.autocast("cuda", dtype=BF, cache_enabled=False):
            return self.model._predict(bb.norm(t), grid_size=self.grid)

    # ------------------------------------------------------------------ head (copy of hd.FastHead.predict / upscale_nhwc with quantisation hooks)
    def _head_q(self, kind: str, X: torch.Tensor, W_kn: torch.Tensor):
        """X [R,K] bf16, W_kn [K,N] bf16 (torch.mm layout) -> quantised (X, W) per the head spec."""
        spec = self.plan.get(("head", kind), P0)
        if spec.fmt == "bf16":
            return X, W_kn
        ck = ("head", kind, spec.key())
        Xf = X.float()
        if spec.had:
            Xf = qr.block_hadamard(Xf, spec.had)
        if ck not in self._wq:
            Wt = W_kn.t().contiguous().float()
            if spec.had:
                Wt = qr.block_hadamard(Wt, spec.had)
            Wq = Wt.to(F16).to(BF) if spec.fmt == "fp16" else qr.qdq(Wt, spec.wf, dim=-1).to(BF)
            self._wq[ck] = Wq.t().contiguous()
        Wq = self._wq[ck]
        if spec.fmt == "fp16":
            return Xf.to(F16).to(BF), Wq
        return qr.qdq(Xf, spec.fmt, dim=-1).to(BF), Wq

    def _head_predict(self, x: torch.Tensor, grid_size):
        m, fh = self.model, self._fh
        q = x[:, : m.num_queries, :]
        class_logits = m.class_head(q)
        xt = x[:, m.num_queries + 1 + m.backbone.n_storage_tokens:, :]
        B, N, C = xt.shape
        Hs, Ws = grid_size
        z = xt.to(BF).reshape(B * N, C)
        e = hd.ext()
        for si, st in enumerate(fh.stages):
            hk = ("ct1", "ct2")[si]
            if self.plan.get(("head", hk), P0).fmt == "fp16":      # real fp16-accumulate prototype kernel (EPI 4: plain bf16 out)
                import mg16
                ck = ("head", hk, "fp16w")
                if ck not in self._wq:
                    self._wq[ck] = st["wr"].t().contiguous().to(F16)
                g = mg16.gemm_plain16(z.to(F16).contiguous(), self._wq[ck], out_dtype=BF, kind=hk)
            else:
                Xq, Wq = self._head_q(hk, z, st["wr"])
                g = torch.mm(Xq, Wq)
            out = torch.empty((B, 2 * Hs, 2 * Ws, fh.C), dtype=BF, device=x.device)
            e.dw_ln_launch(g, st["bias"], st["dww"], st["g"], st["b"], out, B, Hs, Ws, st["eps"])
            z = out.view(B * 4 * Hs * Ws, fh.C)
            Hs, Ws = 2 * Hs, 2 * Ws
        u = out
        mq = m.mask_head(q).to(BF)
        spec = self.plan.get(("head", "einsum"), P0)
        if spec.fmt == "bf16":
            ml = torch.bmm(mq, u.view(B, -1, fh.C).transpose(1, 2))
        elif spec.fmt == "fp16":
            import mg16
            uu = u.view(B, -1, fh.C).to(F16)
            ml = torch.stack([mg16.gemm_plain16(mq[b].to(F16).contiguous(), uu[b].contiguous(), out_dtype=BF, kind="einsum") for b in range(B)])
        else:
            a = mq.reshape(-1, fh.C).float()
            bm = u.view(B, -1, fh.C).reshape(-1, fh.C).float()
            if spec.had:
                a, bm = qr.block_hadamard(a, spec.had), qr.block_hadamard(bm, spec.had)
            aq = qr.qdq(a, spec.fmt, dim=-1).to(BF).view(B, -1, fh.C)
            bq_ = qr.qdq(bm, spec.wf, dim=-1).to(BF).view(B, -1, fh.C)
            ml = torch.bmm(aq, bq_.transpose(1, 2))
        return ml.view(B, mq.shape[1], u.shape[1], u.shape[2]), class_logits
