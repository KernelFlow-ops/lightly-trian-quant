"""Round 10 quantizer reference: the single bit-exact specification for R12's gate A (PLAN_v2 section 2.3 / 4).

Adapted from train-opt round20_fp4_probe/impl/fp4_ref.py (read-only source; FP4 parts) and round17_mxfp8_sm120/impl/mx8_kernels.cu (MXFP8 scale rule),
plus INT8 and FP16-with-power-of-two-prescale, and the block Hadamard used as an outlier-spreading rotation.  Pure torch, any device.

Formats ("fmt")
  mx8   e4m3 data, ue8m0 scale per 32 values along the last dim (RCEIL):  byte = clamp(biased_exp(fl32(amax/448)) + (mantissa != 0), max 255),
        v = x * 2^-(byte-127) (exact) -> cvt.rn.satfinite e4m3.   All-zero block: byte 0, data 0.   Bit-identical to r17 `quant_dual` (tests/test_quant_ref.py)
  nv4   e2m1 data, e4m3 block scale per 16, fp32 per-tensor scale ts = amax/(6*448)      (see fp4_ref docstring; sf = e4m3_RN(min((amax_blk/6)/ts, 448)))
  mx4   e2m1 data, ue8m0 scale per 32, RCEIL: byte = clamp(biased_exp(amax_blk/6) + (mantissa != 0), max 254)
  int8  symmetric int8, one fp32 scale per row (token) for activations / per output channel for weights: s = amax/127, q = clamp(rint(x/s), -127, 127)
  fp16  x * 2^-p rounded to fp16 (p = integer power-of-two pre-scale; the GEMM epilogue multiplies by 2^(pa+pw))

API
  quantize_*(x [R,C]) -> (codes, scales...)    dequant_*(...) -> fp32      qdq(x, fmt, **kw) -> fp32 (dequantized values the tensor core multiplies)
  block_hadamard(x, b) : orthonormal Walsh-Hadamard over consecutive blocks of b (16/32/64) along the last dim (natural order, x 1/sqrt(b))
"""
from __future__ import annotations

import math
from typing import Optional

import torch

VEC = {"nv4": 16, "mx4": 32, "mx8": 32}
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GRID_CACHE: dict = {}
F8_MAX = 448.0


def _grid(device):
    k = str(device)
    if k not in _GRID_CACHE:
        _GRID_CACHE[k] = torch.tensor(E2M1_GRID, dtype=torch.float32, device=device)
    return _GRID_CACHE[k]


# ------------------------------------------------------------------------------------------------ e2m1 (from fp4_ref.py)
def e2m1_rn(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 code (uint8, bit 3 = sign), round-to-nearest-even, saturating at 6 (cvt.rn.satfinite semantics)."""
    a = v.abs()
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)).to(torch.uint8)
    return code | (torch.signbit(v).to(torch.uint8) << 3)


def e2m1_value(code: torch.Tensor) -> torch.Tensor:
    g = _grid(code.device)
    val = g[(code & 7).long()]
    return torch.where((code & 8) != 0, -val, val)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_nibbles(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b & 0xF, b >> 4], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


# ------------------------------------------------------------------------------------------------ block Hadamard
def fwht(x: torch.Tensor, b: int) -> torch.Tensor:
    """Natural-order fast Walsh-Hadamard over consecutive blocks of b along the last dim, butterfly h = 1, 2, 4, ... (unnormalised)."""
    sh = x.shape
    assert sh[-1] % b == 0 and (b & (b - 1)) == 0
    x = x.reshape(*sh[:-1], sh[-1] // b, b)
    h = 1
    while h < b:
        y = x.reshape(*x.shape[:-1], b // (2 * h), 2, h)
        a, c = y[..., 0, :], y[..., 1, :]
        x = torch.stack((a + c, a - c), dim=-2).reshape(*x.shape[:-1], b)
        h *= 2
    return x.reshape(sh)


def hadamard_signs(K: int, seed: int, device="cuda") -> torch.Tensor:
    """fixed +-1 vector [K] (fp32) for the randomised block Hadamard (seed 0 = all ones = the plain block Hadamard)."""
    if seed == 0:
        return torch.ones(K, device=device)
    g = torch.Generator(device="cpu").manual_seed(7919 * seed + K)
    return (torch.randint(0, 2, (K,), generator=g).float() * 2 - 1).to(device)


def block_hadamard(x: torch.Tensor, b: int, signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """orthonormal: <H a, H c> == <a, c>.  fp32 math.  signs [K] (+-1): randomised Hadamard H S x (same signs on activations and weights keep the product)."""
    xf = x.float()
    if signs is not None:
        xf = xf * signs
    return fwht(xf, b) * (1.0 / math.sqrt(b))


# ------------------------------------------------------------------------------------------------ MXFP8
def _rceil_byte(amax: torch.Tensor, denom: float, max_byte: int) -> torch.Tensor:
    """ue8m0 byte (int32) = e + (frac != 0) of fl32(amax / denom)."""
    r = (amax / denom).contiguous().view(torch.int32)
    e = (r >> 23) & 0xFF
    frac = ((r & 0x7FFFFF) != 0).to(torch.int32)
    return (e + frac).clamp(max=max_byte)


def mx8_quantize(x: torch.Tensor):
    """x [R, C] (C % 32 == 0) -> (q [R, C] float8_e4m3fn, byte [R, C/32] uint8)."""
    R, C = x.shape
    xf = x.float().reshape(R, C // 32, 32)
    amax = xf.abs().amax(-1)
    byte = _rceil_byte(amax, F8_MAX, 255)
    inv = torch.exp2((127 - torch.where(byte == 0, torch.ones_like(byte), byte)).float())   # kernel: inv_scale_of(max(s,1))
    v = (xf * inv[..., None]).clamp(-F8_MAX, F8_MAX)
    q = v.to(torch.float8_e4m3fn).reshape(R, C)
    return q, byte.to(torch.uint8)


def mx8_dequant(q: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
    R, C = q.shape
    s = torch.exp2(byte.float() - 127.0)
    return (q.float().reshape(R, C // 32, 32) * s[..., None]).reshape(R, C)


# ------------------------------------------------------------------------------------------------ NVFP4 / MXFP4
def _nv4_ts(xf: torch.Tensor) -> torch.Tensor:
    amax = xf.abs().amax()
    return torch.where(amax > 0, amax / (6.0 * 448.0), torch.ones_like(amax)).reshape(1).float()


def _nv4_sf(amax_blk: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    return (amax_blk / 6.0 / ts).clamp(max=448.0).to(torch.float8_e4m3fn)


def nv4_quantize(x: torch.Tensor, ts: Optional[torch.Tensor] = None):
    """x [R, C] (C % 16 == 0) -> (q [R, C/2] uint8 packed e2m1, sf [R, C/16] float8_e4m3fn, ts [1] fp32).  ts given = static calibrated scale."""
    R, C = x.shape
    xf = x.float().contiguous()
    ts = _nv4_ts(xf) if ts is None else ts.reshape(1).float()
    sf8 = _nv4_sf(xf.reshape(R, C // 16, 16).abs().amax(-1), ts)
    denom = (sf8.float() * ts).reshape(R, C // 16, 1)
    v = xf.reshape(R, C // 16, 16) / torch.where(denom > 0, denom, torch.ones_like(denom))
    v = torch.where(denom > 0, v, torch.zeros_like(v)).reshape(R, C)
    return pack_nibbles(e2m1_rn(v)), sf8, ts


def nv4_dequant(q, sf8, ts) -> torch.Tensor:
    R = q.shape[0]
    vals = e2m1_value(unpack_nibbles(q))
    return vals * (sf8.float() * ts).repeat_interleave(16, 1)


def mx4_quantize(x: torch.Tensor):
    """x [R, C] (C % 32 == 0) -> (q [R, C/2] uint8, byte [R, C/32] uint8)."""
    R, C = x.shape
    xf = x.float().contiguous()
    amax = xf.reshape(R, C // 32, 32).abs().amax(-1)
    byte = _rceil_byte(amax, 6.0, 254)
    inv = torch.exp2((127 - byte).float()).reshape(R, C // 32, 1)
    v = (xf.reshape(R, C // 32, 32) * inv).reshape(R, C)
    return pack_nibbles(e2m1_rn(v)), byte.to(torch.uint8)


def mx4_dequant(q, byte) -> torch.Tensor:
    vals = e2m1_value(unpack_nibbles(q))
    return vals * torch.exp2(byte.float() - 127.0).repeat_interleave(32, 1)


# ------------------------------------------------------------------------------------------------ INT8
def int8_quantize(x: torch.Tensor, dim: int = -1):
    """symmetric per-row (reduce over `dim`) int8: returns (q int8, scale fp32 with the reduced dim kept)."""
    xf = x.float()
    amax = xf.abs().amax(dim, keepdim=True)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(xf / s).clamp(-127, 127).to(torch.int8)       # torch.round = round half to even
    return q, s


def int8_dequant(q, s) -> torch.Tensor:
    return q.float() * s


def int8b_quantize(x: torch.Tensor, b: int):
    """block-scaled int8 (one fp32 scale amax/127 per b consecutive values of the last dim, no per-row reduction): returns (q int8 [R, C], scale fp32 [R, C/b])."""
    R, C = x.shape
    xf = x.float().reshape(R, C // b, b)
    amax = xf.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(xf / s).clamp(-127, 127).to(torch.int8)
    return q.reshape(R, C), s.squeeze(-1)


def int8b_dequant(q: torch.Tensor, s: torch.Tensor, b: int) -> torch.Tensor:
    R, C = q.shape
    return (q.float().reshape(R, C // b, b) * s[..., None]).reshape(R, C)


# ------------------------------------------------------------------------------------------------ FP16 with power-of-two pre-scale
def fp16_quantize(x: torch.Tensor, p: int = 0) -> torch.Tensor:
    """x * 2^-p rounded to fp16 (RN)."""
    return (x.float() * (2.0 ** -p)).to(torch.float16)


def fp16_dequant(h: torch.Tensor, p: int = 0) -> torch.Tensor:
    return h.float() * (2.0 ** p)


# ------------------------------------------------------------------------------------------------ one-call dequantised values
def qdq(x: torch.Tensor, fmt: str, *, dim: int = -1, ts: Optional[torch.Tensor] = None, p: int = 0) -> torch.Tensor:
    """fp32 values the tensor core multiplies for `x` quantised in `fmt` (blocks along the last dim; int8 along `dim`).
    nv4: the fp32 per-tensor scale ts is folded in (the real GEMM applies it in the epilogue; the product is identical up to fp32 rounding)."""
    if fmt == "bf16":
        return x.to(torch.bfloat16).float()
    if fmt == "fp16":
        return fp16_dequant(fp16_quantize(x, p), p)
    if fmt == "mx8":
        return mx8_dequant(*mx8_quantize(x))
    if fmt == "nv4":
        return nv4_dequant(*nv4_quantize(x, ts))
    if fmt == "mx4":
        return mx4_dequant(*mx4_quantize(x))
    if fmt == "int8":
        return int8_dequant(*int8_quantize(x, dim))
    if fmt.startswith("i8b"):          # i8b32 / i8b64 / i8b128: int8 with one fp32 scale per b values along K
        b = int(fmt[3:])
        return int8b_dequant(*int8b_quantize(x, b), b)
    raise ValueError(fmt)


# ------------------------------------------------------------------------------------------------ scale layout (from fp4_ref.py)
def to_blocked(sf2d: torch.Tensor) -> torch.Tensor:
    """uint8 [R, nb] -> [rm, rk, 32, 4, 4] (QuACK / cuBLASLt 128x4 blocked layout; rows padded to 128, nb to 4 with zeros)."""
    R, nb = sf2d.shape
    rm, rk = -(-R // 128), -(-nb // 4)
    pad = torch.zeros(rm * 128, rk * 4, dtype=torch.uint8, device=sf2d.device)
    pad[:R, :nb] = sf2d.view(torch.uint8)
    return pad.view(rm, 128, rk, 4).permute(0, 2, 1, 3).reshape(rm, rk, 4, 32, 4).transpose(2, 3).contiguous()


def from_blocked(sf: torch.Tensor, R: int, nb: int) -> torch.Tensor:
    rm, rk = sf.shape[:2]
    t = sf.view(torch.uint8).transpose(2, 3).reshape(rm, rk, 128, 4).permute(0, 2, 1, 3).reshape(rm * 128, rk * 4)
    return t[:R, :nb].contiguous()
