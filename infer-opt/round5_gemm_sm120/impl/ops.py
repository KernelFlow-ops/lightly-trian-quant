"""Round 5 fused transformer-block ops (bf16, fp32 accumulate).  Drop-in for round3_fusion/impl/ops.py plus fused_block_v2.

Same names / signatures as round3:
    add_layernorm(x, r, w, b)                       K1  CUDA C++ (unchanged kernel, own build dir)
    qkv_rope(h, W, b, sin, cos, prefix, heads)      K2  RoPE/layout epilogue GEMM
    gemm_bias_residual(h, W, b, r, gamma)           K3  bias + LayerScale + residual epilogue GEMM
    gemm_bias_gelu(h, W, b)                         K4  bias + GELU(erf) epilogue GEMM
    prepare_block_weights(block, fold_gamma=False)  weights (+ LayerNorm-folded copies for v2)
    fused_block(x, w, sin, cos, prefix, heads)      7-kernel block (LN kernels kept)
New:
    fused_block_v2(state, w, sin, cos, prefix, heads)   5-kernel block: LayerNorm folded into the consumer GEMM
                                                        (raw bf16 residual copy + row-stat partials written by the producer
                                                        epilogue), tuned tiles, optional PDL
    v2_enter(x, queries=None) -> state                  (x_fp32, xb_bf16, stats) for the first layer / the joint-layer entry

The GEMMs run either on the hand-written CUDA kernel (mg_core.cu, cubin per config, see mgb.py) or on TileLang (gemm_tl.py),
whichever the tuned table (tuned_cfg.json, written by make_tuned.py from the sweeps + interleaved A/B) says is faster for the shape.
All kernels launch on torch's current stream, allocate only their outputs, cache their compiled kernels per shape (CUDA-Graph safe).
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if __package__:
    from . import mgb
else:
    sys.path.insert(0, str(_HERE))
    import mgb  # type: ignore[no-redef]  # noqa: E402

_ln_mod = None
PREFETCH = False      # kernels prefetch the next kernels' weights into L2 (cold-weight DRAM latency is ~16us/layer otherwise)
PDL = True            # programmatic dependent launch between the GEMM kernels of fused_block_v2
ATTN = "flash"        # "flash" (torch SDPA) | "tl" (round4 TileLang plain attention)
BF = torch.bfloat16

_TUNED = _HERE / "tuned_cfg.json"
TUNED: dict = json.loads(_TUNED.read_text()) if _TUNED.exists() else {}

# fallbacks when a shape is not in tuned_cfg.json (found by the sweeps for ViT-L M=2058; see NOTES)
_EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"
_DEFAULT = {
    "fc1": dict(impl="mg", cfg=dict(epi=0, bm=96, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, extra=_EX + " -DSTG=1")),
    "fc2": dict(impl="mg", cfg=dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, splitk=2, extra=_EX)),
    "proj": dict(impl="mg", cfg=dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=1, wn=4, minb=3, extra=_EX)),
    "qkv": dict(impl="mg", cfg=dict(epi=2, bm=64, bn=128, bk=32, st=2, wm=2, wn=2, minb=3, extra=_EX + " -DSTG=1")),
}


def set_pdl(flag: bool):
    global PDL
    PDL = bool(flag)


def _pick(kind, N, K, M):
    """tuned entry for (kind, N, K, M): exact M, else the nearest tuned M of the same (kind, N, K), else the default."""
    key = f"{kind}|{N}|{K}|{M}"
    if key in TUNED:
        return TUNED[key]
    best = None
    for k, v in TUNED.items():
        kk, nn, kd, mm = k.split("|")
        if kk == kind and int(nn) == N and int(kd) == K:
            dm = abs(int(mm) - M)
            if best is None or dm < best[0]:
                best = (dm, v)
    return best[1] if best else _DEFAULT[kind]


def _with(cfg, **kw):
    c = dict(cfg)
    c.update(kw)
    if "extra" in c:
        # NFULL only valid when N % bn == 0: caller adds it explicitly via cfg; nothing to do here
        pass
    return c


def _nfull(cfg, N):
    """make sure -DNFULL=1 is only present when N % bn == 0 (tuned configs are shape specific but be safe)."""
    ex = cfg.get("extra", "")
    if "-DNFULL=1" in ex and N % cfg["bn"]:
        cfg = dict(cfg, extra=ex.replace("-DNFULL=1", ""))
    return cfg


# ----------------------------------------------------------------------------------------------------------- K1 (unchanged)
def _ln():
    global _ln_mod
    if _ln_mod is None:
        from torch.utils.cpp_extension import load

        build = "/root/autodl-tmp/infer-opt-tmp/gemm120/build_ln"
        os.makedirs(build, exist_ok=True)
        _ln_mod = load(name="r5_add_ln", sources=[str(_HERE / "ln_kernels.cu")], build_directory=build,
                       extra_cuda_cflags=["-arch=sm_120a", "-O3", "--use_fast_math"], verbose=False)
    return _ln_mod


def add_layernorm(x, r, w, b, eps: float = 1e-5):
    """returns (s, y): s = x + r (new residual; == x when r is None), y = LN(s) bf16."""
    return tuple(_ln().add_layernorm(x, r, w, b, eps))


# ----------------------------------------------------------------------------------------------------------- K2/K3/K4 (v1 API)
_ONES: dict = {}


def _tl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r5_gemm_tl", str(_HERE / "gemm_tl.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_TL = None


def tl():
    global _TL
    if _TL is None:
        _TL = _tl()
    return _TL


def _is_tl_cfg(cfg):
    return cfg is not None and "threads" in cfg


def gemm_bias_gelu(h, W, b, cfg=None):
    shp = h.shape
    h2 = h.reshape(-1, shp[-1])
    M, K = h2.shape
    N = W.shape[0]
    if _is_tl_cfg(cfg):
        return tl().gemm_bias_gelu_kernel(M, N, K, **cfg)(h2, W, b).view(*shp[:-1], N)
    ent = {"impl": "mg", "cfg": cfg} if cfg is not None else _pick("fc1", N, K, M)
    out = torch.empty(M, N, device=h.device, dtype=BF)
    if ent["impl"] == "tl":
        return tl().gemm_bias_gelu_kernel(M, N, K, **ent["cfg"])(h2, W, b).view(*shp[:-1], N)
    mgb.get(_nfull(ent["cfg"], N))(h2, W, b, C=out)
    return out.view(*shp[:-1], N)


def gemm_bias_residual(h, W, b, r, gamma=None, cfg=None):
    """out = r + gamma * (h @ W^T + b); r bf16 or fp32 (fp32 residual stream), output has r's dtype."""
    shp = h.shape
    h2 = h.reshape(-1, shp[-1])
    M, K = h2.shape
    N = W.shape[0]
    if gamma is None:
        key = (N, W.device)
        gamma = _ONES.get(key)
        if gamma is None:
            gamma = _ONES[key] = torch.ones(N, device=W.device, dtype=torch.float32)
    r2 = r.reshape(-1, N)
    rf32 = r.dtype == torch.float32
    if _is_tl_cfg(cfg):
        c = dict(cfg, res_dtype="float32" if rf32 else "bfloat16")
        return tl().gemm_bias_residual_kernel(M, N, K, **c)(h2, W, b, gamma, r2).view(*shp[:-1], N)
    kind = "fc2" if K > N else "proj"
    ent = {"impl": "mg", "cfg": cfg} if cfg is not None else _pick(kind, N, K, M)
    if ent["impl"] == "tl":
        c = dict(ent["cfg"], res_dtype="float32" if rf32 else "bfloat16")
        return tl().gemm_bias_residual_kernel(M, N, K, **c)(h2, W, b, gamma, r2).view(*shp[:-1], N)
    out = torch.empty_like(r2)
    mgb.get(_nfull(dict(ent["cfg"], rf32=1 if rf32 else 0), N))(h2, W, b, gamma, r2, out)
    return out.view(*shp[:-1], N)


def qkv_rope(h, W, b, sin, cos, prefix: int, num_heads: int, cfg=None, round_bf16: bool = False, layout: str = "bnhd"):
    """h [B,N,d] -> logical out [3,B,H,N,64] (out[0]=q, out[1]=k roped for tokens >= prefix, out[2]=v).
    layout="bnhd" (default): physical [3,B,N,H,64], returned as a permuted view (SDPA-friendly, no copies)."""
    B, N, d = h.shape
    if round_bf16 or layout != "bnhd" or _is_tl_cfg(cfg):
        c = cfg if _is_tl_cfg(cfg) else {}
        k = tl().qkv_rope_kernel(B, N, d, num_heads, prefix, round_bf16=round_bf16, layout=layout, **c)
        out = k(h.reshape(B * N, d), W, b, sin, cos)
        return out.transpose(2, 3) if layout == "bnhd" else out
    ent = {"impl": "mg", "cfg": cfg} if cfg is not None else _pick("qkv", 3 * d, d, B * N)
    if ent["impl"] == "tl":
        k = tl().qkv_rope_kernel(B, N, d, num_heads, prefix, layout="bnhd", **ent["cfg"])
        return k(h.reshape(B * N, d), W, b, sin, cos).transpose(2, 3)
    out = torch.empty(3, B, N, num_heads, 64, device=h.device, dtype=BF)
    mgb.get(_nfull(ent["cfg"], 3 * d))(h.reshape(B * N, d), W, b, C=out, sin=sin, cos=cos, Ntok=N, prefix=prefix, d=d, H=num_heads)
    return out.transpose(2, 3)


# ----------------------------------------------------------------------------------------------------------- weights
@torch.no_grad()
def _fold_ln(W, ln_w, ln_b, bias):
    """LayerNorm folded into the following linear (see mg_core.cu): Wf = bf16(ln_w * W), cs = sum_k Wf, bias32 = bias + W @ ln_b."""
    Wf = (W.float() * ln_w[None, :]).to(BF).contiguous()
    cs = Wf.float().sum(1).contiguous()
    b32 = (bias.float() + W.float() @ ln_b).contiguous()
    return Wf, cs, b32


@torch.no_grad()
def prepare_block_weights(block, fold_gamma: bool = False, ln_fold: bool = True) -> dict:
    """SelfAttentionBlock -> bf16 contiguous weights, K-bias masked.  fold_gamma as in round3.
    ln_fold=True additionally stores the LayerNorm-folded copies used by fused_block_v2 (W' = ln_w * W in bf16, column sums of W',
    fp32 folded bias) -- costs one extra bf16 copy of Wqkv and W1."""
    qkv = block.attn.qkv
    bq = qkv.bias
    if hasattr(qkv, "bias_mask"):
        bq = bq * qkv.bias_mask.to(bq.dtype)
    g1 = block.ls1.gamma.float()
    g2 = block.ls2.gamma.float()
    bf = lambda t: t.detach().to(BF).contiguous()  # noqa: E731
    f32 = lambda t: t.detach().float().contiguous()  # noqa: E731
    one = torch.ones_like(g1)
    fg1, fg2 = (g1, g2) if fold_gamma else (one, one)
    w = {
        "g1": (one if fold_gamma else g1).contiguous(), "g2": (one if fold_gamma else g2).contiguous(),
        "ln1_w": f32(block.norm1.weight), "ln1_b": f32(block.norm1.bias), "ln1_eps": block.norm1.eps,
        "Wqkv": bf(qkv.weight), "bqkv": bf(bq),
        "Wproj": bf(fg1[:, None] * block.attn.proj.weight.float()),
        "bproj": bf(fg1 * block.attn.proj.bias.float()),
        "ln2_w": f32(block.norm2.weight), "ln2_b": f32(block.norm2.bias), "ln2_eps": block.norm2.eps,
        "W1": bf(block.mlp.fc1.weight), "b1": bf(block.mlp.fc1.bias),
        "W2": bf(fg2[:, None] * block.mlp.fc2.weight.float()),
        "b2": bf(fg2 * block.mlp.fc2.bias.float()),
    }
    if ln_fold:
        w["Wqkv_f"], w["cs_qkv"], w["bqkv32"] = _fold_ln(qkv.weight.detach(), w["ln1_w"], w["ln1_b"], bq.detach().float())
        w["W1_f"], w["cs1"], w["b1_32"] = _fold_ln(block.mlp.fc1.weight.detach(), w["ln2_w"], w["ln2_b"], block.mlp.fc1.bias.detach().float())
    return w


# ----------------------------------------------------------------------------------------------------------- v1 block
def _attn(qkv, B, N, d, heads):
    if ATTN == "tl":
        return _tl_attn(qkv)
    q = qkv.transpose(2, 3)  # logical [3,B,H,N,64] view of the physical [3,B,N,H,64]
    o = F.scaled_dot_product_attention(q[0], q[1], q[2])
    return o.transpose(1, 2).reshape(B * N, d)  # free view: flash writes [B,N,H,64]


_R4 = None


def _tl_attn(qkv):
    global _R4
    if _R4 is None:
        import importlib.util
        p = "/root/lightly-train/infer-opt/round4_attention/impl"
        sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location("r4_ops", p + "/ops.py")
        _R4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_R4)
    three, B, N, H, D = qkv.shape
    return _R4.attention_3bnhd(qkv).reshape(B * N, H * D)


def fused_block(x, w: dict, sin, cos, prefix: int, num_heads: int, rope_round_bf16: bool = False):
    """x [B,N,d] -> x_out (same dtype).  7 kernels: LN1, qkv+rope, SDPA, proj+res, LN2, fc1+gelu, fc2+res (tuned tiles)."""
    B, N, d = x.shape
    _, h = add_layernorm(x, None, w["ln1_w"], w["ln1_b"], w["ln1_eps"])
    qkv = qkv_rope(h, w["Wqkv"], w["bqkv"], sin, cos, prefix, num_heads, round_bf16=rope_round_bf16)
    q = qkv
    o = F.scaled_dot_product_attention(q[0], q[1], q[2])
    o = o.transpose(1, 2).reshape(B, N, d)
    x = gemm_bias_residual(o, w["Wproj"], w["bproj"], x, w["g1"])
    _, h = add_layernorm(x, None, w["ln2_w"], w["ln2_b"], w["ln2_eps"])
    a = gemm_bias_gelu(h, w["W1"], w["b1"])
    return gemm_bias_residual(a, w["W2"], w["b2"], x, w["g2"])


# ----------------------------------------------------------------------------------------------------------- v2 block
class State:
    """residual stream + the raw-bf16 copy and row-stat partials consumed by the LN-fused GEMMs of the next kernel."""
    __slots__ = ("x", "xb", "stats", "np_", "pcols")

    def __init__(self, x, xb, stats, np_, pcols):
        self.x, self.xb, self.stats, self.np_, self.pcols = x, xb, stats, np_, pcols


def v2_enter(x, queries=None) -> State:
    """x fp32 [B,Nin,d] (+ optional query rows [Q,d] prepended per batch) -> State (one fused pass: fp32 copy, bf16 copy, row stats)."""
    x = x.contiguous()
    x_out, xb, stats = mgb.ln_prep(x, queries, write_x=False)
    return State(x if x_out is None else x_out, xb, stats, 1, x.shape[-1])


def _mg_kernel(kind, N, K, M, **over):
    # v1 sweep results do not measure v2's LN/stats epilogues; keep their tile choices independent.
    ent = TUNED.get(f"v2_{kind}|{N}|{K}|{M}", _DEFAULT[kind])
    if ent["impl"] != "mg":
        ent = _DEFAULT[kind]
    cfg = _nfull(dict(ent["cfg"]), N)
    cfg.update(over)
    return mgb.get(cfg), cfg


def fused_block_v2(st: State, w: dict, sin, cos, prefix: int, num_heads: int, pdl=None, next_w: dict = None, prefetch=None) -> State:
    """5 kernels: [LN1+qkv+rope] flash [proj+res(+stats)] [LN2+fc1+gelu] [fc2+res(+stats)]  (LN folded into the consumers)."""
    pdl = PDL if pdl is None else pdl
    prefetch = PREFETCH if prefetch is None else prefetch
    x = st.x
    B, N, d = x.shape
    M = B * N
    ff = w["W1"].shape[0]
    H = num_heads
    dev = x.device
    # --- qkv (LN1 folded)
    kq, cq = _mg_kernel("qkv", 3 * d, d, M, lnf=1)
    qkv = torch.empty(3, B, N, H, 64, device=dev, dtype=BF)
    kq(st.xb, w["Wqkv_f"], w["bqkv32"], C=qkv, cs=w["cs_qkv"], stats_in=st.stats, np_in=st.np_, pcols_in=st.pcols,
       sin=sin, cos=cos, Ntok=N, prefix=prefix, d=d, H=H, eps=w["ln1_eps"], pdl=pdl,
       prefetch=(w["Wproj"], w["W1_f"]) if prefetch else ())
    o = _attn(qkv, B, N, d, H)
    # --- proj + residual (+ x1 bf16 copy + row-stat partials)
    kp, cp = _mg_kernel("proj", d, d, M, stats=1, rf32=1)
    NP1 = -(-d // cp["bn"])
    x1 = torch.empty_like(x)
    xb1 = torch.empty(M, d, device=dev, dtype=BF)
    st1 = torch.empty(M, NP1, 2, device=dev, dtype=torch.float32)
    kp(o, w["Wproj"], w["bproj"], w["g1"], x, x1, Cb=xb1, stats_out=st1, pdl=pdl)
    # --- fc1 (LN2 folded) + GELU
    k1, c1 = _mg_kernel("fc1", ff, d, M, lnf=1)
    a = torch.empty(M, ff, device=dev, dtype=BF)
    k1(xb1, w["W1_f"], w["b1_32"], C=a, cs=w["cs1"], stats_in=st1, np_in=NP1, pcols_in=cp["bn"], eps=w["ln2_eps"], pdl=pdl,
       prefetch=(w["W2"],) if prefetch else ())
    # --- fc2 + residual (+ bf16 copy + stats for the next layer's LN1)
    k2, c2 = _mg_kernel("fc2", d, ff, M, stats=1, rf32=1)
    NP2 = -(-d // c2["bn"])
    x2 = torch.empty_like(x)
    xb2 = torch.empty(M, d, device=dev, dtype=BF)
    st2 = torch.empty(M, NP2, 2, device=dev, dtype=torch.float32)
    k2(a, w["W2"], w["b2"], w["g2"], x1, x2, Cb=xb2, stats_out=st2, pdl=pdl,
       prefetch=(next_w["Wqkv_f"],) if (prefetch and next_w is not None) else ())
    return State(x2, xb2, st2, NP2, c2["bn"])
