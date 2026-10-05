"""Pure-torch reference for the phase-3 FP4 formats (``FP4Tensor`` of CONTRACTS 11.2).  Bit-exact specification for the R21/R22 kernels (gate A).

Formats (``fmt``)
  nv4  e2m1 data, e4m3 block scale per 16 values, fp32 per-tensor scale ``ts = amax/(6*448)`` (amax == 0 -> ts = 1)
         s_blk = e4m3_RN(min((amax_blk / 6) / ts, 448))        (fp32 ops in exactly this order; same as QuACK ``to_nvfp4`` / Spark)
         v     = x / (e4m3(s_blk) * ts)  [fp32 division]  ->  e2m1 (RN-even, saturating at +-6, sign bit from signbit(v))
         a block whose scale is 0 (all zeros, or amax_blk/6/ts below the e4m3 subnormal range) is all-zero data with scale byte 0
         (QuACK ``to_nvfp4`` instead clamps the scale to >= 2^-6 and multiplies by a reciprocal: see selftest, a)
  mx4  e2m1 data, ue8m0 block scale per 32 values, no per-tensor scale
         byte = clamp(biased_exp(amax_blk/6) + (mantissa(amax_blk/6) != 0), max=254)     (RCEIL: scale = 2^ceil(log2(amax_blk/6)), no saturation)
         v    = x * 2^-(byte-127)  (exact)  ->  e2m1.   ALL-ZERO BLOCK CONVENTION: scale byte 0, data 0 (same as r17 padding rows).
         ``mx_round="floor"`` reproduces QuACK ``to_mxfp4`` / Spark (e = floor(log2 amax) - 2) for cross checks only.

Layouts
  q    [R, C/2] uint8, even element index in the low nibble, K-major (contraction axis contiguous)
  sf   [R/128, C/(4 vec), 32, 4, 4] uint8 (QuACK / cuBLASLt 128x4 blocked), view as float8_e4m3fn (nv4) / float8_e8m0fnu (mx4)
  qt   [C, Rpad/2], sft [C/128, Rpad/(4 vec), 32, 4, 4]: the transposed copy, quantized along the token axis; token rows R..Rpad-1 are 0
  ts / tst  fp32 [1] per-tensor scale of the row / transposed copy (nv4 only; separate because RHT changes the amax)

Options: RHT (16-point FWHT butterfly, x1/4, then x signs[j], along the quantization axis), SR (dedicated ``torch.Generator``, never the
default one; or an explicit uniform tensor ``u`` so a kernel with the same stream can be compared bit for bit), weights with 2D scale
(nv4 16x16 / mx4 32x32 tiles: W and W^T are element-wise identical) or 1D scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

VEC = {"nv4": 16, "mx4": 32}
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
RHT_SEED = 20251003
_GRID_CACHE: dict = {}


def pad128(n: int) -> int:
    return -(-n // 128) * 128


def make_gen(device="cuda", seed: int = 1234) -> torch.Generator:
    """Dedicated generator for SR noise (never touches the default stream)."""
    return torch.Generator(device=device).manual_seed(seed)


def rht_signs(seed: int = RHT_SEED, n: int = 16, device="cuda") -> torch.Tensor:
    """Fixed +-1 sign vector [n] (fp32), from a private CPU generator."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1).to(device)


# ------------------------------------------------------------------------------------------------ e2m1
def _grid(device):
    k = str(device)
    if k not in _GRID_CACHE:
        _GRID_CACHE[k] = torch.tensor(E2M1_GRID, dtype=torch.float32, device=device)
    return _GRID_CACHE[k]


def e2m1_rn(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 code (uint8, bit 3 = sign), round-to-nearest-even, saturating at 6 (cvt.rn.satfinite semantics)."""
    a = v.abs()
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)).to(torch.uint8)
    return code | (torch.signbit(v).to(torch.uint8) << 3)


def e2m1_sr(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding to the e2m1 grid: pick the upper neighbour iff u < (|v|-lo)/(hi-lo), u in [0,1); saturates at 6.  Unbiased in [-6,6]."""
    g = _grid(v.device)
    a = v.abs().clamp(max=6.0)
    idx = (torch.bucketize(a, g, right=True) - 1).clamp(0, 6)
    lo, hi = g[idx], g[idx + 1]
    p = (a - lo) / (hi - lo)
    code = (idx + (u < p).long()).to(torch.uint8)
    return code | (torch.signbit(v).to(torch.uint8) << 3)


def e2m1_value(code: torch.Tensor) -> torch.Tensor:
    g = _grid(code.device)
    val = g[(code & 7).long()]
    return torch.where((code & 8) != 0, -val, val)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[..., C] codes -> [..., C/2] bytes, even index in the low nibble."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_nibbles(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b & 0xF, b >> 4], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


# ------------------------------------------------------------------------------------------------ scale layout
def to_blocked(sf2d: torch.Tensor) -> torch.Tensor:
    """uint8 [R, nb] -> [rm, rk, 32, 4, 4] (QuACK/cuBLASLt blocked layout; rows padded to 128, nb to 4 with zeros)."""
    R, nb = sf2d.shape
    rm, rk = -(-R // 128), -(-nb // 4)
    pad = torch.zeros(rm * 128, rk * 4, dtype=torch.uint8, device=sf2d.device)
    pad[:R, :nb] = sf2d.view(torch.uint8)
    return pad.view(rm, 128, rk, 4).permute(0, 2, 1, 3).reshape(rm, rk, 4, 32, 4).transpose(2, 3).contiguous()


def from_blocked(sf: torch.Tensor, R: int, nb: int) -> torch.Tensor:
    rm, rk = sf.shape[:2]
    t = sf.view(torch.uint8).transpose(2, 3).reshape(rm, rk, 128, 4).permute(0, 2, 1, 3).reshape(rm * 128, rk * 4)
    return t[:R, :nb].contiguous()


# ------------------------------------------------------------------------------------------------ RHT
def fwht16(x: torch.Tensor) -> torch.Tensor:
    """Natural-order fast Walsh-Hadamard transform over the last dim (16), butterfly order h = 1, 2, 4, 8 (the spec for the kernels)."""
    sh = x.shape
    h = 1
    while h < 16:
        y = x.reshape(*sh[:-1], 16 // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(sh)
        h *= 2
    return x


def rht_apply(xf: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """16-point RHT along the last dim: FWHT, x 1/4, x signs[j].  Orthogonal: <rht(a), rht(b)> == <a, b> up to fp32 rounding."""
    R, C = xf.shape
    y = fwht16(xf.reshape(R, C // 16, 16)) * 0.25
    return (y * signs).reshape(R, C)


# ------------------------------------------------------------------------------------------------ block scales + codes
def _round_codes(v, sr, u):
    return e2m1_sr(v, u) if sr else e2m1_rn(v)


def _nv4_ts(xf):
    amax = xf.abs().amax()
    return torch.where(amax > 0, amax / (6.0 * 448.0), torch.ones_like(amax)).reshape(1).float()


def _nv4_sf(amax_blk, ts):
    """e4m3 block scale (float8_e4m3fn) from the block amax."""
    return (amax_blk / 6.0 / ts).clamp(max=448.0).to(torch.float8_e4m3fn)


def _mx4_byte(amax_blk, mx_round="rceil"):
    """ue8m0 byte (int32) of the block scale."""
    if mx_round == "rceil":
        r = (amax_blk / 6.0).view(torch.int32)
        e = (r >> 23) & 0xFF
        frac = ((r & 0x7FFFFF) != 0).to(torch.int32)
        return (e + frac).clamp(max=254)
    p = torch.frexp(amax_blk).exponent - 1 - 2
    return torch.where(amax_blk > 0, p + 127, torch.zeros_like(p)).clamp(0, 254).to(torch.int32)


def _codes_nv4(xf, sf8, ts, sr, u):
    R, C = xf.shape
    denom = (sf8.float() * ts).reshape(R, C // 16, 1)
    v = xf.reshape(R, C // 16, 16) / torch.where(denom > 0, denom, torch.ones_like(denom))
    v = torch.where(denom > 0, v, torch.zeros_like(v)).reshape(R, C)
    return _round_codes(v, sr, u)


def _codes_mx4(xf, byte, sr, u):
    R, C = xf.shape
    inv = torch.exp2((127 - byte).float()).reshape(R, C // 32, 1)
    return _round_codes((xf.reshape(R, C // 32, 32) * inv).reshape(R, C), sr, u)


def quantize_matrix(x: torch.Tensor, fmt: str, *, signs: Optional[torch.Tensor] = None, sr: bool = False, u: Optional[torch.Tensor] = None,
                    gen: Optional[torch.Generator] = None, mx_round: str = "rceil"):
    """Quantize a [R, C] matrix along C (C % vec == 0, R % 1 == 0; the caller pads rows).  Returns (q [R,C/2] u8, sf blocked u8, ts [1] fp32 or None)."""
    xf = x.float().contiguous()
    R, C = xf.shape
    vec = VEC[fmt]
    assert C % vec == 0, (C, vec)
    if signs is not None:
        xf = rht_apply(xf, signs)
    if sr and u is None:
        u = torch.rand(R, C, generator=gen, device=xf.device)
    if fmt == "nv4":
        ts = _nv4_ts(xf)
        sf8 = _nv4_sf(xf.reshape(R, C // 16, 16).abs().amax(-1), ts)
        codes = _codes_nv4(xf, sf8, ts, sr, u)
        return pack_nibbles(codes), to_blocked(sf8.view(torch.uint8)), ts
    amax = xf.reshape(R, C // 32, 32).abs().amax(-1)
    byte = _mx4_byte(amax, mx_round)
    codes = _codes_mx4(xf, byte, sr, u)
    return pack_nibbles(codes), to_blocked(byte.to(torch.uint8)), None


def quantize_matrix_2d(x: torch.Tensor, fmt: str):
    """Weight quantization with a 2D (vec x vec) block scale.  Returns (q [R,C/2], sf, ts, qt [C,R/2], sft, tst): row copy and the
    transposed copy (quantized along R); codes of qt are exactly the transposed codes of q and the scales are the transposed tile scales."""
    xf = x.float().contiguous()
    R, C = xf.shape
    v = VEC[fmt]
    assert R % v == 0 and C % v == 0
    amax_t = xf.reshape(R // v, v, C // v, v).abs().amax((1, 3))  # [R/v, C/v]
    if fmt == "nv4":
        ts = _nv4_ts(xf)
        sf8 = _nv4_sf(amax_t, ts)  # [R/v, C/v]
        sfr = sf8.repeat_interleave(v, 0)  # [R, C/v]
        sfc = sf8.t().contiguous().repeat_interleave(v, 0)  # [C, R/v]
        codes = _codes_nv4(xf, sfr, ts, False, None)
        codes_t = codes.t().contiguous()
        return (pack_nibbles(codes), to_blocked(sfr.view(torch.uint8)), ts, pack_nibbles(codes_t), to_blocked(sfc.view(torch.uint8)), ts)
    byte = _mx4_byte(amax_t)
    bytr = byte.repeat_interleave(v, 0)
    bytc = byte.t().contiguous().repeat_interleave(v, 0)
    codes = _codes_mx4(xf, bytr, False, None)
    return (pack_nibbles(codes), to_blocked(bytr.to(torch.uint8)), None, pack_nibbles(codes.t().contiguous()), to_blocked(bytc.to(torch.uint8)), None)


# ------------------------------------------------------------------------------------------------ container
@dataclass
class FP4Tensor:
    fmt: str
    rows: int
    cols: int
    rpad: int
    q: Optional[torch.Tensor] = None
    sf: Optional[torch.Tensor] = None
    ts: Optional[torch.Tensor] = None
    qt: Optional[torch.Tensor] = None
    sft: Optional[torch.Tensor] = None
    tst: Optional[torch.Tensor] = None
    rht: str = ""  # "", "row", "col", "row+col"
    sr: bool = False

    def sf_view(self, t):
        return t.view(torch.float8_e4m3fn if self.fmt == "nv4" else torch.float8_e8m0fnu)


def quantize(x: torch.Tensor, fmt: str, *, row: bool = True, col: bool = True, rht_row: bool = False, rht_col: bool = False, sr: bool = False,
             gen: Optional[torch.Generator] = None, u_row=None, u_col=None, signs=None, rpad: Optional[int] = None, mx_round="rceil") -> FP4Tensor:
    """Activation / gradient ``x [R, C]``: row copy (along C) and transposed copy (along the zero-padded token axis).
    SR noise: row copy draws first from ``gen``, then the transposed copy (uniform [0,1) of shape [R,C] / [C,Rpad])."""
    R, C = x.shape
    rp = pad128(R) if rpad is None else rpad
    if (rht_row or rht_col) and signs is None:
        signs = rht_signs(device=x.device)
    t = FP4Tensor(fmt, R, C, rp, rht="+".join(n for n, f in (("row", rht_row), ("col", rht_col)) if f), sr=sr)
    if row:
        t.q, t.sf, t.ts = quantize_matrix(x, fmt, signs=signs if rht_row else None, sr=sr, u=u_row, gen=gen, mx_round=mx_round)
    if col:
        xt = torch.nn.functional.pad(x.float(), (0, 0, 0, rp - R)).t().contiguous()  # [C, Rpad]
        t.qt, t.sft, t.tst = quantize_matrix(xt, fmt, signs=signs if rht_col else None, sr=sr, u=u_col, gen=gen, mx_round=mx_round)
    return t


def quantize_weight(w: torch.Tensor, fmt: str, *, two_d: bool = True, rht_row: bool = False, rht_col: bool = False, signs=None) -> FP4Tensor:
    """Weight ``w [N, K]`` (nn.Linear layout): ``q`` = W along K (fwd B operand), ``qt`` = W^T [K, N] along N (dgrad B operand).  RN only."""
    N, K = w.shape
    t = FP4Tensor(fmt, N, K, N, rht="+".join(n for n, f in (("row", rht_row), ("col", rht_col)) if f))
    if (rht_row or rht_col) and signs is None:
        signs = rht_signs(device=w.device)
    if two_d and not (rht_row or rht_col):
        t.q, t.sf, t.ts, t.qt, t.sft, t.tst = quantize_matrix_2d(w, fmt)
        return t
    if two_d:  # RHT breaks the W / W^T identity: each rotated copy gets its own 2D tiles
        a = rht_apply(w.float(), signs) if rht_row else w.float()
        bt = rht_apply(w.float().t().contiguous(), signs) if rht_col else w.float().t().contiguous()  # [K, N], rotated along N
        t.q, t.sf, t.ts = quantize_matrix_2d(a, fmt)[:3]
        t.qt, t.sft, t.tst = quantize_matrix_2d(bt, fmt)[:3]
        return t
    t.q, t.sf, t.ts = quantize_matrix(w, fmt, signs=signs if rht_row else None)
    t.qt, t.sft, t.tst = quantize_matrix(w.t().contiguous(), fmt, signs=signs if rht_col else None)
    return t


# ------------------------------------------------------------------------------------------------ dequant
def dequant_copy(q, sf, ts, fmt: str, R: int, C: int, apply_ts: bool = True) -> torch.Tensor:
    """fp32 [R, C] from (q [R,C/2], blocked sf) -- the exact values the tensor core multiplies (times ts if apply_ts)."""
    vec = VEC[fmt]
    vals = e2m1_value(unpack_nibbles(q))
    s2 = from_blocked(sf, R, C // vec)
    sv = s2.view(torch.float8_e4m3fn).float() if fmt == "nv4" else torch.exp2(s2.float() - 127.0)
    out = vals * sv.repeat_interleave(vec, 1)
    if apply_ts and ts is not None:
        out = out * ts
    return out


def dequant_row(t: FP4Tensor, apply_ts: bool = True):
    return dequant_copy(t.q, t.sf, t.ts, t.fmt, t.q.shape[0], t.cols, apply_ts)


def dequant_col(t: FP4Tensor, apply_ts: bool = True):
    """[C, Rpad] fp32 (token axis last, padded rows are 0)."""
    return dequant_copy(t.qt, t.sft, t.tst, t.fmt, t.qt.shape[0], t.qt.shape[1] * 2, apply_ts)


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 reference of the dequantized product ``a [M,K] @ b [N,K]^T`` returned as fp32 (the value an exact-product / fp32-accumulate GEMM approximates)."""
    return (a.double() @ b.double().t()).float()
