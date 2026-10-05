"""Round 11 attention candidates.  Every candidate takes the r5 qkv GEMM epilogue output qkv_phys [3,B,N,H,64] (bf16, RoPE applied)
and returns O [B,N,H*64] bf16 (the proj GEMM input).

  get_full(name)  -> fn(qkv, B, N, d) -> O          everything the candidate needs between the qkv GEMM and the proj GEMM
                                                      (mean/quant/convert/transposes included): usable as r11 `attn=<name>`
  get_split(name) -> (prep(qkv) -> state, kern(state) -> O)   prep = everything that production would fuse into the qkv epilogue
                                                      (quantization, K mean, dtype conversion), kern = the attention kernel alone
Candidates: flash, tl, sage2_fp8 (INT8 QK per-warp + FP8 PV, fp32+fp16 accum), sage2_fp8_f32 (fp32+fp32), sage2_pv16 (INT8 QK + FP16 PV,
f16 accum), sage2_pv16_f32 (fp16+fp32), sage3 (FP4), a16k0..a16k4 (round-11 FA2 prototype, kinds 0-4), spark_fp8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TMP = "/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0"
SAGE2 = TMP + "/sage2s"   # copy with all launches moved onto the current torch stream (stock Sage2 launches on the legacy default stream: unusable under CUDA graph capture / side streams)
SAGE3 = TMP + "/sage3"
_cache: dict = {}


def _sage2():
    if "sage2" not in _cache:
        if SAGE2 not in sys.path:
            sys.path.insert(0, SAGE2)
        import sageattention
        from sageattention import quant, sm80_compile, sm89_compile
        _cache["sage2"] = (sageattention, quant, sm80_compile, sm89_compile)
    return _cache["sage2"]


def _sage3():
    if "sage3" not in _cache:
        if SAGE3 not in sys.path:
            sys.path.insert(0, SAGE3)
        import sageattn3.api as api
        _cache["sage3"] = api
    return _cache["sage3"]


# ----------------------------------------------------------------------------------------------------------- flash / tilelang
def flash(qkv, B, N, d, layer=None):
    q = qkv.transpose(2, 3)
    o = F.scaled_dot_product_attention(q[0], q[1], q[2])
    return o.transpose(1, 2).reshape(B, N, d)


def tl(qkv, B, N, d, layer=None):
    if "r4" not in _cache:
        import importlib.util
        p = "/root/lightly-train/infer-opt/round4_attention/impl"
        sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location("r4_ops", p + "/ops.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _cache["r4"] = m
    return _cache["r4"].attention_3bnhd(qkv).reshape(B, N, d)


# ----------------------------------------------------------------------------------------------------------- SageAttention2
def _sage2_fp8_split(accum):
    sa, quant, sm80, sm89 = _sage2()
    kern_fn = {"fp32+fp16": sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
               "fp32+fp32": sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf}[accum]

    def prep(qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # NHD views [B,N,H,64]
        km = k.mean(dim=1, keepdim=True)
        q8, qs, k8, ks = quant.per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
        v8, vs, _ = quant.per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25 if accum == "fp32+fp16" else 448.0, smooth_v=False)
        return dict(q8=q8, qs=qs, k8=k8, ks=ks, v8=v8, vs=vs, o=torch.empty_like(q))

    def kern(st):
        kern_fn(st["q8"], st["k8"], st["v8"], st["o"], st["qs"], st["ks"], st["vs"], 0, 0, 2, 0.125, 0)
        o = st["o"]
        return o.reshape(o.shape[0], o.shape[1], -1)
    return prep, kern


def _sage2_pv16_split(accum):
    sa, quant, sm80, sm89 = _sage2()
    kern_fn = {"fp16": sm80.qk_int8_sv_f16_accum_f16_attn, "fp16+fp32": sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf}[accum]

    def prep(qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        km = k.mean(dim=1, keepdim=True)
        q8, qs, k8, ks = quant.per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
        return dict(q8=q8, qs=qs, k8=k8, ks=ks, v16=v.to(torch.float16), o=torch.empty_like(q))

    def kern(st):
        kern_fn(st["q8"], st["k8"], st["v16"], st["o"], st["qs"], st["ks"], 0, 0, 2, 0.125, 0)
        o = st["o"]
        return o.reshape(o.shape[0], o.shape[1], -1)
    return prep, kern


# ----------------------------------------------------------------------------------------------------------- SageAttention3 (FP4)
def _sage3_split():
    api = _sage3()
    import fp4attn_cuda  # noqa: F401  (imported by api)

    def prep(qkv):
        q = qkv[0].transpose(1, 2).contiguous()      # [B,H,N,D]; the fp4 pipeline mutates k in place -> private copies
        k = qkv[1].transpose(1, 2).contiguous()
        v = qkv[2].transpose(1, 2).contiguous()
        KL = k.size(2)
        q, k, v, delta_s = api.preprocess_qkv(q, k, v, True)
        ql = api.scale_and_quant_fp4(q)
        kl = api.scale_and_quant_fp4_permute(k)
        vl = api.scale_and_quant_fp4_transpose(v)
        return dict(ql=ql, kl=kl, vl=vl, ds=delta_s, KL=KL, QL=qkv.shape[2], bf16=(qkv.dtype == torch.bfloat16))

    def kern(st):
        o = api.blockscaled_fp4_attn(st["ql"], st["kl"], st["vl"], st["ds"], st["KL"], False, True, st["bf16"])
        o = o[0] if isinstance(o, (tuple, list)) else o
        o = o[:, :, :st["QL"]]                       # [B,H,N,D]
        return o.transpose(1, 2).reshape(o.shape[0], st["QL"], -1)
    return prep, kern


# ----------------------------------------------------------------------------------------------------------- round-11 FA2 prototype
def _a16_split(kind, pre=1.0, **cfg):
    import attn16
    e = attn16.ext(**cfg)

    def prep(qkv):
        return qkv if kind == 0 else qkv.half().contiguous()

    def kern(x):
        return e.attn16_fwd(x, kind, pre)
    return prep, kern


# ----------------------------------------------------------------------------------------------------------- Spark fp8 attention
def _spark_fp8_split(per_head=True):
    if "spark" not in _cache:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import spark_fp8
        _cache["spark"] = spark_fp8.load_spark_fp8()
    ext = _cache["spark"]

    def quant(x):                                   # x [B,H,S,D] -> e4m3 + descale ([B,H] or [1])
        xf = x.float()
        amax = xf.abs().amax(dim=(2, 3), keepdim=True) if per_head else xf.abs().amax()
        scale = (amax / 448.0).clamp(min=1e-30)
        x8 = (xf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return x8.contiguous(), (scale.reshape(x.shape[0], x.shape[1]) if per_head else scale.reshape(1)).contiguous()

    def prep(qkv):
        q8, qs = quant(qkv[0].transpose(1, 2))
        k8, ks = quant(qkv[1].transpose(1, 2))
        v8, vs = quant(qkv[2].transpose(1, 2))
        return dict(q8=q8, k8=k8, v8=v8, qs=qs, ks=ks, vs=vs)

    def kern(st):
        o = ext.attention_fp8(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"], per_head, -1)   # [B,H,N,D] bf16
        return o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1)
    return prep, kern


def get_split(name):
    if name == "spark_fp8":
        return _spark_fp8_split(True)
    if name == "flash":
        return (lambda qkv: qkv), (lambda qkv: flash(qkv, qkv.shape[1], qkv.shape[2], qkv.shape[3] * 64))
    if name == "tl":
        return (lambda qkv: qkv), (lambda qkv: tl(qkv, qkv.shape[1], qkv.shape[2], qkv.shape[3] * 64))
    if name == "sage2_fp8":
        return _sage2_fp8_split("fp32+fp16")
    if name == "sage2_fp8_f32":
        return _sage2_fp8_split("fp32+fp32")
    if name == "sage2_pv16":
        return _sage2_pv16_split("fp16")
    if name == "sage2_pv16_f32":
        return _sage2_pv16_split("fp16+fp32")
    if name == "sage3":
        return _sage3_split()
    if name.startswith("a16k"):
        base, _, cfg = name.partition("@")
        base, _, pre = base.partition("~")          # "a16k4~8": Q pre-scaled by 1/8 (power of two) for the f16-accumulated S
        kv = dict((p.split("=")[0], int(p.split("=")[1])) for p in cfg.split(";") if p)
        return _a16_split(int(base[4]), pre=(1.0 / int(pre)) if pre else 1.0, **kv)
    raise ValueError(name)


def get_full(name):
    if name == "flash":
        return flash
    if name == "tl":
        return tl
    if name.startswith("a16pol"):
        # production-style policy: S and PV in f16 everywhere; the early layers whose S range exceeds f16 get a power-of-two Q pre-scale
        p_lo, k_lo = get_split("a16k4~8" + name[6:])
        p_hi, k_hi = get_split("a16k4" + name[6:])

        def fn_pol(qkv, B, N, d, layer=None):
            return k_lo(p_lo(qkv)) if (layer is not None and layer <= 1) else k_hi(p_hi(qkv))
        return fn_pol
    if name.startswith("a16mix"):
        # layer 0: PV-f16 only (kind 3, S in f32: layer-0 S overflows f16); other layers: S and PV in f16 (kind 4)
        p3, k3 = get_split("a16k3" + name[6:])
        p4, k4 = get_split("a16k4" + name[6:])

        def fn_mix(qkv, B, N, d, layer=None):
            return k3(p3(qkv)) if layer == 0 else k4(p4(qkv))
        return fn_mix
    prep, kern = get_split(name)

    def fn(qkv, B, N, d, layer=None):
        return kern(prep(qkv))
    return fn
