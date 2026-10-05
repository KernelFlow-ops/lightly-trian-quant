// round21 FP4 (NVFP4 / MXFP4) fused quantizers for the sm_120 training stack (CUDA C++, compile with -gencode arch=compute_120a,code=sm_120a).
//
//   quant_act    x[R, C] (bf16 / fp32, C % 64 == 0)  ->  ONE read of x, any combination of
//                  row copy   q  [R, C/2]  e2m1 (even index = low nibble), blocks along C, blocked scales sf   [rm, rk, 32, 4, 4]
//                  col copy   qt [C, Kt/2] e2m1, blocks along the token axis (Kt = Rpad, zero padded; or prefix mode Kt = B*P),
//                             blocked scales sft [ceil(C/128), Kt/(4 vec), 32, 4, 4]
//                  prefix rows (first npre tokens of each image) gathered as bf16 xp [B*npre, C]; the col copy then only covers the P = N-npre patch tokens
//                optional per copy: 16-point RHT (FWHT16, x1/4, x signs) along the quantisation axis, RN or SR (Philox / explicit uniforms).
//   quant_weight w[N, K] (fp32 / bf16)  ->  q [N, K/2] and qt [K, N/2] with a 2D (vec x vec) block scale (identical numerics of both copies), RN only.
//   amax_roll    per step: prev <- next, next <- 0, seed += 1, step += 1 (device only, graph capturable)
//
// Bit-exact reference: round20_fp4_probe/impl/fp4_ref.py (see tests/test_fp4q.py).  Formats (CONTRACTS 12.2):
//   nv4: ts = amax / (6*448) (amax == 0 -> 1); sf = e4m3_RN(min((amax_blk/6)/ts, 448)); v = x / (float(sf) * ts) (IEEE fp32 division); e2m1 RN-even / SR
//   mx4: byte = clamp(biased_exp(amax_blk/6) + (mant != 0), 254) (RCEIL); v = x * 2^-(byte-127)
//   `div_true` = 0 (default): "/ 6" and "/ 2688" are multiplications by fl32(1/6), fl32(1/2688), which is what fp4_ref does on CUDA tensors (torch scalar-divide
//   = reciprocal multiply); div_true = 1: IEEE fp32 division (= fp4_ref run on CPU tensors = the literal text of CONTRACTS 12.2).
//
// Geometry: CTA = 128 threads, tile = 64 tokens x 64 columns.  Loads: thread = 4 rows (rg + 16 p) x 8 consecutive columns (16 B bf16) -> the row copy runs straight from
// registers; the raw tile goes to smem (fp32, ld 68) for the col copy: thread = (column c = tid % 64, token half = tid / 64) = 32 tokens (2 nv4 groups or 1 mx4 block),
// writes 16 B of qt (half a 32 B sector; the other half comes from the sibling warp of the same CTA).
// Third-party: none copied (design hints from Spark fp4quant.cu / QuTLASS were only read).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu
#ifndef PHILOX_ROUNDS
#define PHILOX_ROUNDS 10
#endif
#ifndef ACT_MINB
#define ACT_MINB 1
#endif
constexpr int NPART = 16;  // sub-slots per (site, copy) of the spread amax publication: one 32 B sector each
constexpr float RCP6 = 1.0f / 6.0f;
constexpr float RCP2688 = 1.0f / 2688.0f;

// ------------------------------------------------------------------------------------------------ loads / conversions
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
  float4 a = *reinterpret_cast<const float4*>(p);
  float4 b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void store8_bf16(__nv_bfloat16* p, const float* v) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

__device__ __forceinline__ uint32_t f32_to_e4m3(float t) {  // RN-even, satfinite
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(t));
  return (uint32_t)(r & 0xffu);
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)b));
  return __half2float(__ushort_as_half((unsigned short)(h & 0xffffu)));
}
// 8 floats -> 8 packed e2m1 (v[0] in the low nibble of byte 0), RN-even, saturating at +-6
__device__ __forceinline__ uint32_t pack8_rn(const float* v) {
  uint32_t out;
  asm("{\n.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}
// stochastic rounding to the e2m1 grid (same definition as fp4_ref.e2m1_sr): upper neighbour iff u < (|v|-lo)/(hi-lo); returns code | sign<<3.
// Bit trick (exact): for 1 <= a <= 6 the lower neighbour is 2^e (1 + m1/2) -> code (bits >> 22) - 252 (2e + m1 + 2) and the fraction p is the low 22 mantissa bits / 2^22;
// for a < 1 the spacing is 0.5: b = 2a, code = (b >= 1), p = b - code.  (Both p are exact in fp32, the compare u < p is the same compare as the reference.)
__device__ __forceinline__ uint32_t e2m1_sr_code(float v, float u) {
  const uint32_t ab = min(__float_as_uint(v) & 0x7fffffffu, 0x40C00000u);  // |v| clamped to 6
  const float a = __uint_as_float(ab);
  const float p_hi = (float)(ab & 0x3FFFFFu) * 2.384185791015625e-07f;
  const float b = a + a;
  const bool ge = b >= 1.f;
  const float p_lo = ge ? b - 1.f : b;
  const bool hi = ab >= 0x3F800000u;
  const uint32_t code = (hi ? ((ab >> 22) - 252u) : (uint32_t)ge) + ((u < (hi ? p_hi : p_lo)) ? 1u : 0u);
  return code | ((__float_as_uint(v) >> 28) & 8u);
}

// ------------------------------------------------------------------------------------------------ Philox4x32-10 (counter based)
__device__ __forceinline__ uint4 philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
#pragma unroll
  for (int r = 0; r < PHILOX_ROUNDS; r++) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}
// runtime round count (generic kernel / reference stream helper; the fast kernels use the compile-time philox_t<NR>)
__device__ __forceinline__ uint4 philox4x32_rt(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1, int nr) {
#pragma unroll 1
  for (int r = 0; r < nr; r++) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}
// 16 bit uniform in (0, 1): u = (2h + 1) / 2^17 (h = 16 random bits): P(u < p) = p up to the 2^-17 granularity (negligible against the e2m1 grid)
__device__ __forceinline__ float u16(uint32_t w, int hi) { return (float)((((w >> (16 * hi)) & 0xffffu) << 1) | 1u) * 7.62939453125e-06f; }
// uniforms of the flat elements f .. f+7 (f % 8 == 0): explicit fp32 tensor, or ONE Philox call with counter (f/8, w1, 0, 0), key = device seed:
// element j = f % 8 takes the (j%2)-th 16 bit half of output word j/2
__device__ __forceinline__ void get_u8(float* u, const float* up, size_t f, uint32_t k0, uint32_t k1, uint32_t w1, int nr = 10) {
  if (up) {
    float4 a = *reinterpret_cast<const float4*>(up + f), b = *reinterpret_cast<const float4*>(up + f + 4);
    u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w; u[4] = b.x; u[5] = b.y; u[6] = b.z; u[7] = b.w;
  } else {
    const uint4 r = nr == 10 ? philox4x32_10((uint32_t)(f >> 3), w1, 0u, 0u, k0, k1) : philox4x32_rt((uint32_t)(f >> 3), w1, 0u, 0u, k0, k1, nr);
    u[0] = u16(r.x, 0); u[1] = u16(r.x, 1); u[2] = u16(r.y, 0); u[3] = u16(r.y, 1);
    u[4] = u16(r.z, 0); u[5] = u16(r.z, 1); u[6] = u16(r.w, 0); u[7] = u16(r.w, 1);
  }
}

// ------------------------------------------------------------------------------------------------ scale rules
__device__ __forceinline__ float ts_of(float amax, int div_true) {
  if (!(amax > 0.f)) return 1.f;
  return div_true ? __fdiv_rn(amax, 2688.f) : amax * RCP2688;
}
__device__ __forceinline__ uint32_t nv4_sf_byte(float amax_blk, float ts, int div_true) {
  float t = div_true ? __fdiv_rn(amax_blk, 6.f) : amax_blk * RCP6;
  t = __fdiv_rn(t, ts);
  t = fminf(t, 448.f);
  return f32_to_e4m3(t);
}
// y / d for a whole block sharing the denominator d, bit-identical to IEEE __fdiv_rn on the working range (verified by div_check, tests/test_fp4q.py):
// Markstein: r = RN(1/d) (once per block), q = RN(y r), res = fma(-d, q, y), q' = RN(q + res r).  Outside 2^-100 < d < 2^100 fall back to __fdiv_rn.
template <int N>
__device__ __forceinline__ void div_vec(const float* y, float* t, float d) {
  if (d > 7.8886090522101180e-31f && d < 1.2676506002282294e30f) {
    const float r = __frcp_rn(d);
#pragma unroll
    for (int j = 0; j < N; j++) {
      const float q = __fmul_rn(y[j], r);
      const float res = fmaf(-d, q, y[j]);
      const float tq = fmaf(res, r, q);
      t[j] = __uint_as_float((__float_as_uint(tq) & 0x7fffffffu) | (__float_as_uint(y[j]) & 0x80000000u));  // d > 0: sign(y / d) = sign(y), also for +-0
    }
  } else {
#pragma unroll
    for (int j = 0; j < N; j++) t[j] = __fdiv_rn(y[j], d);
  }
}
__device__ __forceinline__ uint32_t mx4_byte(float amax_blk, int div_true) {
  const float r = div_true ? __fdiv_rn(amax_blk, 6.f) : amax_blk * RCP6;
  const uint32_t b = __float_as_uint(r);
  const uint32_t s = ((b >> 23) & 0xffu) + ((b & 0x7fffffu) != 0u ? 1u : 0u);
  return s > 254u ? 254u : s;
}
__device__ __forceinline__ float mx4_inv(uint32_t byte) {  // 2^(127 - byte)
  const uint32_t f = 254u - byte;                           // byte in [0, 254]
  return __uint_as_float(f > 0u ? (f << 23) : 0x00400000u);
}

// ------------------------------------------------------------------------------------------------ RHT
__device__ __forceinline__ void fwht16(float (&x)[16]) {
#pragma unroll
  for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
      if ((i & h) == 0) {
        const float a = x[i], b = x[i + h];
        x[i] = __fadd_rn(a, b);
        x[i + h] = __fsub_rn(a, b);
      }
    }
  }
}
__device__ __forceinline__ float sgn(float v, uint32_t mask, int j) { return __uint_as_float(__float_as_uint(v) ^ (((mask >> j) & 1u) << 31)); }
// 16-point RHT of a row group held by two adjacent lanes (hi = lane parity: this lane holds elements 8*hi .. 8*hi+7)
__device__ __forceinline__ void rht_row8(float (&x)[8], int hi, uint32_t mask) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if ((i & h) == 0) {
        const float a = x[i], b = x[i + h];
        x[i] = __fadd_rn(a, b);
        x[i + h] = __fsub_rn(a, b);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float o = __shfl_xor_sync(FULL_MASK, x[i], 1);
    x[i] = hi ? __fsub_rn(o, x[i]) : __fadd_rn(x[i], o);
  }
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = sgn(__fmul_rn(x[i], 0.25f), mask, hi * 8 + i);
}

// ------------------------------------------------------------------------------------------------ amax atomics (values >= 0: float bits order == integer order)
__device__ __forceinline__ void amax_publish(unsigned* dst, unsigned bits) {
  if (bits > *reinterpret_cast<volatile unsigned*>(dst)) atomicMax(dst, bits);
}
__device__ __forceinline__ unsigned warp_max_u(unsigned a) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) a = max(a, __shfl_xor_sync(FULL_MASK, a, o));
  return a;
}

struct ActParams {
  const void* x;
  uint8_t *q, *sf, *qt, *sft;
  __nv_bfloat16* xp;
  const float *u_row, *u_col;
  const float* tsrc;      // ts_mode 0: [ts_row, ts_col];  1: [amax_row, amax_col] (ts = f(amax * amax_mul))
  float* ts_out;          // [2] or null
  unsigned* amax_next;    // [2] or null: atomicMax targets (bits of the current step's amax of the rotated values)
  unsigned* amax_parts;   // [2, NPART, 8] or null (fast kernels): the same amax, spread over NPART sectors per copy (slot = linear CTA id % NPART); folded into amax_next by the roll kernel
  const int64_t* seed;    // device Philox seed or null
  int R, C, Kt, rows_pad;
  int B, N, npre, P, Tp, PT;  // prefix mode iff npre > 0
  int ts_mode, div_true, rht_r, rht_c, sr_r, sr_c, amax_only, nr, srm;
  uint32_t sign_mask, sr_off;
  float amax_mul;
};

template <typename T, int FMT, bool ROW, bool COL>
__global__ void __launch_bounds__(128, ACT_MINB) act_kernel(const ActParams p) {
  constexpr int VEC = FMT == 0 ? 16 : 32;
  constexpr int LD = 68;
  __shared__ __align__(16) float sm[COL ? 64 * LD : 4];
  const int tid = threadIdx.x, lane = tid & 31;
  const int cb = blockIdx.x, ty = blockIdx.y;
  const T* X = reinterpret_cast<const T*>(p.x);

  // ---- tile -> source rows (CTA uniform)
  int kind = 0, row_base, nvalid, ctile = -1, xp_row0 = 0;
  if (p.npre == 0) {
    row_base = ty * 64;
    nvalid = p.R - row_base;
    ctile = ty;
  } else {
    const int ppt = p.P >> 6;
    if (ty < p.Tp) {
      const int b = ty / ppt;
      row_base = b * p.N + p.npre + (ty - b * ppt) * 64;
      nvalid = 64;
      ctile = ty;
    } else if (ty < p.Tp + p.B * p.PT) {
      const int t = ty - p.Tp, b = t / p.PT, j = t - b * p.PT;
      row_base = b * p.N + 64 * j;
      nvalid = p.npre - 64 * j;
      kind = 1;
      xp_row0 = b * p.npre + 64 * j;
    } else {
      row_base = p.R + 64 * (ty - p.Tp - p.B * p.PT);
      nvalid = 0;
      kind = 2;
    }
  }
  nvalid = max(0, min(64, nvalid));
  const bool want_row = ROW && (nvalid > 0 || ((p.npre == 0 || kind == 2) && row_base < p.rows_pad));
  const bool want_xp = (kind == 1) && p.xp != nullptr && nvalid > 0;
  const bool want_col = COL && ctile >= 0;
  if (!want_row && !want_col && !want_xp) return;
  const bool zero_pad_ok = (p.npre == 0) || kind == 2;

  // ---- per tensor scales
  float ts_r = 1.f, ts_c = 1.f;
  if constexpr (FMT == 0) {
    if (!p.amax_only) {
      if (p.ts_mode == 0) {
        ts_r = p.tsrc[0];
        ts_c = p.tsrc[1];
      } else {
        ts_r = ts_of(p.tsrc[0] * p.amax_mul, p.div_true);
        ts_c = ts_of(p.tsrc[1] * p.amax_mul, p.div_true);
      }
      if (cb == 0 && ty == 0 && tid == 0 && p.ts_out) {
        p.ts_out[0] = ts_r;
        p.ts_out[1] = ts_c;
      }
    }
  }
  uint32_t k0 = 0, k1 = 0;
  if (p.seed) {
    const int64_t s = *p.seed;
    k0 = (uint32_t)s;
    k1 = (uint32_t)((uint64_t)s >> 32);
  }

  // ---- load the 64 x 64 tile
  const int cg = tid & 7, rg = tid >> 3;
  const int col0 = cb * 64 + cg * 8;
  float v[4][8];
#pragma unroll
  for (int pp = 0; pp < 4; pp++) {
    const int i = pp * 16 + rg;
    if (i < nvalid) {
      load8(X + (size_t)(row_base + i) * p.C + col0, v[pp]);
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) v[pp][j] = 0.f;
    }
  }
  if (want_xp) {
#pragma unroll
    for (int pp = 0; pp < 4; pp++) {
      const int i = pp * 16 + rg;
      if (i < nvalid) store8_bf16(p.xp + (size_t)(xp_row0 + i) * p.C + col0, v[pp]);
    }
  }
  if (want_col) {
#pragma unroll
    for (int pp = 0; pp < 4; pp++) {
      float4* d = reinterpret_cast<float4*>(&sm[(pp * 16 + rg) * LD + cg * 8]);
      d[0] = make_float4(v[pp][0], v[pp][1], v[pp][2], v[pp][3]);
      d[1] = make_float4(v[pp][4], v[pp][5], v[pp][6], v[pp][7]);
    }
  }

  // ---- row copy (blocks along C), straight from registers
  if constexpr (ROW) {
    if (want_row) {
      unsigned rmax = 0u;
      const int rk = FMT == 0 ? (p.C >> 6) : ((p.C + 127) >> 7);
      const int ncb = p.C >> 6;
#pragma unroll
      for (int pp = 0; pp < 4; pp++) {
        const int i = pp * 16 + rg;
        const int row = row_base + i;
        float (&y)[8] = v[pp];
        if (p.rht_r) rht_row8(y, cg & 1, p.sign_mask);
        unsigned a = 0u;
#pragma unroll
        for (int j = 0; j < 8; j++) a = max(a, __float_as_uint(fabsf(y[j])));
        a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, 1));
        if constexpr (FMT == 1) a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, 2));
        rmax = max(rmax, a);
        if (p.amax_only) continue;
        const float af = __uint_as_float(a);
        uint32_t sb, w = 0u;
        float u[8];
        if (p.sr_r && i < nvalid) get_u8(&u[0], p.u_row, (size_t)row * p.C + col0, k0, k1, p.sr_off * 2u, p.nr);
        if constexpr (FMT == 0) {
          sb = nv4_sf_byte(af, ts_r, p.div_true);
          const float denom = e4m3_to_f32(sb) * ts_r;
          if (denom > 0.f) {
            float t8[8];
            div_vec<8>(y, t8, denom);
            if (!p.sr_r) {
              w = pack8_rn(t8);
            } else {
#pragma unroll
              for (int j = 0; j < 8; j++) w |= e2m1_sr_code(t8[j], u[j]) << (4 * j);
            }
          }
        } else {
          sb = mx4_byte(af, p.div_true);
          const float inv = mx4_inv(sb);
          float t8[8];
#pragma unroll
          for (int j = 0; j < 8; j++) t8[j] = y[j] * inv;
          if (!p.sr_r) {
            w = pack8_rn(t8);
          } else {
#pragma unroll
            for (int j = 0; j < 8; j++) w |= e2m1_sr_code(t8[j], u[j]) << (4 * j);
          }
        }
        if (i < nvalid) *reinterpret_cast<uint32_t*>(p.q + (size_t)row * (p.C >> 1) + (col0 >> 1)) = w;
        // scale bytes: gather the blocks of this 64-column tile in lane cg == 0
        if constexpr (FMT == 0) {
          const uint32_t s1 = __shfl_down_sync(FULL_MASK, sb, 2), s2 = __shfl_down_sync(FULL_MASK, sb, 4), s3 = __shfl_down_sync(FULL_MASK, sb, 6);
          if (cg == 0 && (i < nvalid || (zero_pad_ok && row < p.rows_pad))) {
            const uint32_t word = (sb & 0xffu) | ((s1 & 0xffu) << 8) | ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
            reinterpret_cast<uint32_t*>(p.sf)[(size_t)((row >> 7) * rk + cb) * 128 + (row & 31) * 4 + ((row >> 5) & 3)] = word;
          }
        } else {
          const uint32_t s1 = __shfl_down_sync(FULL_MASK, sb, 4);
          if (cg == 0 && (i < nvalid || (zero_pad_ok && row < p.rows_pad))) {
            uint8_t* base = p.sf + (size_t)((row >> 7) * rk + (cb >> 1)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4;
            *reinterpret_cast<uint16_t*>(base + (cb & 1) * 2) = (uint16_t)((sb & 0xffu) | ((s1 & 0xffu) << 8));
            if (cb == ncb - 1 && (cb & 1) == 0) *reinterpret_cast<uint16_t*>(base + 2) = 0;
          }
        }
      }
      if (p.amax_next && FMT == 0) {
        rmax = warp_max_u(rmax);
        if (lane == 0) amax_publish(p.amax_next + 0, rmax);
      }
    }
  }

  // ---- col copy (blocks along the token axis), from smem
  if constexpr (COL) {
    if (want_col) {
      __syncthreads();
      const int c = tid & 63, half = tid >> 6;
      const int col = cb * 64 + c;
      const int kt0 = ctile * 64 + half * 32;
      float xs[32];
#pragma unroll
      for (int k = 0; k < 32; k++) xs[k] = sm[(half * 32 + k) * LD + c];
      unsigned cmax = 0u;
      uint32_t w[4] = {0u, 0u, 0u, 0u};
      uint32_t sbytes = 0u;
      const size_t fcol = (size_t)col * p.Kt + kt0;
      const uint32_t wc = p.sr_off * 2u + 1u;
      if constexpr (FMT == 0) {
#pragma unroll
        for (int g = 0; g < 2; g++) {
          float y[16];
#pragma unroll
          for (int j = 0; j < 16; j++) y[j] = xs[g * 16 + j];
          if (p.rht_c) {
            fwht16(y);
#pragma unroll
            for (int j = 0; j < 16; j++) y[j] = sgn(__fmul_rn(y[j], 0.25f), p.sign_mask, j);
          }
          unsigned a = 0u;
#pragma unroll
          for (int j = 0; j < 16; j++) a = max(a, __float_as_uint(fabsf(y[j])));
          cmax = max(cmax, a);
          if (p.amax_only) continue;
          const uint32_t sb = nv4_sf_byte(__uint_as_float(a), ts_c, p.div_true);
          sbytes |= (sb & 0xffu) << (8 * g);
          const float denom = e4m3_to_f32(sb) * ts_c;
          uint32_t w0 = 0u, w1 = 0u;
          if (denom > 0.f) {
            float t16[16];
            div_vec<16>(y, t16, denom);
            if (!p.sr_c) {
              w0 = pack8_rn(&t16[0]);
              w1 = pack8_rn(&t16[8]);
            } else {
              float uu[16];
              get_u8(&uu[0], p.u_col, fcol + g * 16, k0, k1, wc, p.nr);
              get_u8(&uu[8], p.u_col, fcol + g * 16 + 8, k0, k1, wc, p.nr);
#pragma unroll
              for (int j = 0; j < 8; j++) {
                w0 |= e2m1_sr_code(t16[j], uu[j]) << (4 * j);
                w1 |= e2m1_sr_code(t16[8 + j], uu[8 + j]) << (4 * j);
              }
            }
          }
          w[2 * g] = w0;
          w[2 * g + 1] = w1;
        }
      }
      if (p.amax_only) {
        if (p.amax_next && FMT == 0) {
          cmax = warp_max_u(cmax);
          if (lane == 0) amax_publish(p.amax_next + 1, cmax);
        }
        return;
      }
      if constexpr (FMT == 1) {
        float y[32];
#pragma unroll
        for (int j = 0; j < 32; j++) y[j] = xs[j];
        if (p.rht_c) {
#pragma unroll
          for (int h2 = 0; h2 < 2; h2++) {
            float t0[16];
#pragma unroll
            for (int j = 0; j < 16; j++) t0[j] = y[h2 * 16 + j];
            fwht16(t0);
#pragma unroll
            for (int j = 0; j < 16; j++) y[h2 * 16 + j] = sgn(__fmul_rn(t0[j], 0.25f), p.sign_mask, j);
          }
        }
        unsigned a = 0u;
#pragma unroll
        for (int j = 0; j < 32; j++) a = max(a, __float_as_uint(fabsf(y[j])));
        const uint32_t sb = mx4_byte(__uint_as_float(a), p.div_true);
        sbytes = sb & 0xffu;
        const float inv = mx4_inv(sb);
#pragma unroll
        for (int g = 0; g < 4; g++) {
          float t8[8];
#pragma unroll
          for (int j = 0; j < 8; j++) t8[j] = y[g * 8 + j] * inv;
          if (!p.sr_c) {
            w[g] = pack8_rn(t8);
          } else {
            float uu[8];
            get_u8(uu, p.u_col, fcol + g * 8, k0, k1, wc, p.nr);
            uint32_t ww = 0u;
#pragma unroll
            for (int j = 0; j < 8; j++) ww |= e2m1_sr_code(t8[j], uu[j]) << (4 * j);
            w[g] = ww;
          }
        }
      }
      // qt row `col`, token columns kt0 .. kt0+31 -> 16 bytes
      *reinterpret_cast<uint4*>(p.qt + (size_t)col * (p.Kt >> 1) + (kt0 >> 1)) = make_uint4(w[0], w[1], w[2], w[3]);
      // scales of the transposed matrix: row `col`, k-blocks along tokens
      const int rkc = FMT == 0 ? (p.Kt >> 6) : (p.Kt >> 7);
      const int kb = kt0 / VEC;  // first k-block of this thread (global)
      uint8_t* base = p.sft + (size_t)((col >> 7) * rkc + (kb >> 2)) * 512 + (col & 31) * 16 + ((col >> 5) & 3) * 4 + (kb & 3);
      if constexpr (FMT == 0) *reinterpret_cast<uint16_t*>(base) = (uint16_t)sbytes;
      else *base = (uint8_t)sbytes;
      // rows >= C of the last 128-row atom (C % 128 == 64) must be zero bytes
      if ((p.C & 127) == 64 && cb == (p.C >> 6) - 1) {
        uint8_t* b2 = p.sft + (size_t)(((col + 64) >> 7) * rkc + (kb >> 2)) * 512 + ((col + 64) & 31) * 16 + (((col + 64) >> 5) & 3) * 4 + (kb & 3);
        if constexpr (FMT == 0) *reinterpret_cast<uint16_t*>(b2) = 0;
        else *b2 = 0;
      }
      if (p.amax_next && FMT == 0) {
        cmax = warp_max_u(cmax);
        if (lane == 0) amax_publish(p.amax_next + 1, cmax);
      }
    }
  }
}

// ================================================================================================ R24 fast act kernel (nv4, bf16 input, no RHT, no prefix, Philox / no SR)
// Same outputs, bit for bit, as act_kernel above on its domain (tests/test_fp4q2.py compares both, and fp4_ref) -- restructured for instruction count:
//   * compile-time ROW / COL / SR / Philox rounds (no dead RHT / SR / prefix / div_true code, no runtime branches in the element loops),
//   * thread = one 16-element block (row copy: 2 blocks of 16 columns; col copy: 2 columns x 16 tokens), so the block scale, 1/denom and the amax are computed once per block (R21a:
//     both lanes of a pair redundantly), |x| max on packed bf16 pairs (VIMNMX.U16x2), the nv4 block-scale division t / ts by Markstein with a hoisted RN(1/ts) (bit-identical to the
//     IEEE division for ts in [2^-60, 2^60]; out of that range a per-element __fdiv_rn path runs),
//   * the tile goes to smem as raw bf16 (16 B stores), the col copy reads 32-bit pairs of adjacent columns,
//   * SR: integer carry formulation of the same decision u < p (u = (2h+1)/2^17): for |v| >= 1 round-up iff m22 > 32 (2h+1), done as one add + shift; for |v| < 1 a directed-rounding
//     (FADD.RP) ceil compare -- exact, same Philox stream (counter (flat/8, w1, 0, 0)) as the generic kernel.
// Row scale bytes are stored per lane (1 B) instead of a shuffle-gathered word.
struct TsF {  // per-tensor scale, its RN reciprocal and the range flag for the Markstein block-scale division
  float ts, rts;
  bool ok;
};
__device__ __forceinline__ TsF make_ts(float ts) {
  TsF c;
  c.ts = ts;
  c.ok = (ts >= 8.673617379884035e-19f) && (ts <= 1.152921504606847e18f);  // [2^-60, 2^60]
  c.rts = c.ok ? __frcp_rn(ts) : 0.f;
  return c;
}
// e4m3_RN(min((amax/6) / ts, 448)) == nv4_sf_byte(amax, ts, div_true = 0)
__device__ __forceinline__ uint32_t sb_fast(float amax, const TsF& c) {
  const float t = amax * RCP6;
  float q;
#ifdef R24_ASSUME_OK
  {
#else
  if (c.ok) {
#endif
    q = __fmul_rn(t, c.rts);
    const float res = fmaf(-c.ts, q, t);
    q = fmaf(res, c.rts, q);
#ifndef R24_ASSUME_OK
  } else {
    q = __fdiv_rn(t, c.ts);
  }
#else
  }
#endif
  return f32_to_e4m3(fminf(q, 448.f));
}
__device__ __forceinline__ uint32_t amax16_bits(const uint32_t (&w)[8]) {  // max |bf16| bit pattern (15 bit) of 16 elements held as 8 words
  uint32_t m = w[0] & 0x7fff7fffu;
#pragma unroll
  for (int i = 1; i < 8; i++) m = __vmaxu2(m, w[i] & 0x7fff7fffu);
  return max(m & 0xffffu, m >> 16);
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

template <int NR>
__device__ __forceinline__ uint4 philox_t(uint32_t c0, uint32_t c1, uint32_t k0, uint32_t k1) {  // counter (c0, c1, 0, 0)
  uint32_t c2 = 0u, c3 = 0u;
#pragma unroll
  for (int r = 0; r < NR; r++) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}
// 3-bit magnitude code of SR(|v|) for the quotient bits qb (any sign) and the 16 random bits h of the element (u = (2h+1) / 2^17):  identical to e2m1_sr_code(v, u) & 7
__device__ __forceinline__ uint32_t sr_mag(uint32_t qb, uint32_t h) {
  const uint32_t ab = min(qb & 0x7fffffffu, 0x40C00000u);
  // |v| >= 1: carry of (m22 + 2^22 - 1 - 32 n) into bit 22  <=>  m22 > 32 n  <=>  u < p_hi;  code = ((ab + R) >> 22) - 252 = (((ab + R + 4 << 22) >> 22) & 7),  n = 2h + 1
  const uint32_t t = ab + (0x13FFFFFu - 32u) - (h << 6);
  const uint32_t mag_hi = (t >> 22) & 7u;
  // |v| < 1: grid {0, 0.5, 1}; u < p_lo  <=>  n < ceil(X), X = a 2^18 - ge 2^17 (all exact); ceil via FADD.RP with 2^23
  const float F = __uint_as_float(ab) * 262144.f;
  const bool ge = ab >= 0x3F000000u;
  const float X = ge ? F - 131072.f : F;
  const uint32_t xc = __float_as_uint(__fadd_ru(X, 8388608.f));
  const uint32_t mag_lo = (ge ? 1u : 0u) + ((0x4B000001u + (h << 1) < xc) ? 1u : 0u);
  return ab >= 0x3F800000u ? mag_hi : mag_lo;
}
__device__ __forceinline__ uint32_t copysign_bits(uint32_t mag_bits, uint32_t sign_src) {  // (mag & 0x7fffffff) | (sign_src & 0x80000000) in ONE LOP3
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(d) : "r"(mag_bits), "r"(sign_src), "r"(0x7fffffffu));
  return d;
}
// 16 values y[0..15] with common denominator d (r = RN(1/d)) -> 8 packed bytes (low nibble = element 0).  f0 / w1: Philox counter words of the first 8 elements (second group f0 + 1).
template <bool SR, int NR, int SRM>
__device__ __forceinline__ uint2 quant16(const float (&y)[16], float d, float r, bool ok, uint32_t f0, uint32_t w1, uint32_t k0, uint32_t k1) {
  float t[16];  // quotient y / d (RN); sign of y (also +-0) restored in the RN path, the SR path takes the sign from y itself
  if (ok) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const float q = __fmul_rn(y[j], r);
      const float res = fmaf(-d, q, y[j]);
      t[j] = fmaf(res, r, q);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; j++) t[j] = __fdiv_rn(y[j], d);
  }
  uint2 o;
  if constexpr (!SR) {
#pragma unroll
    for (int j = 0; j < 16; j++) t[j] = __uint_as_float(copysign_bits(__float_as_uint(t[j]), __float_as_uint(y[j])));  // d > 0: sign(y / d) = sign(y), also for +-0
    o.x = pack8_rn(&t[0]);
    o.y = pack8_rn(&t[8]);
  } else if constexpr (SRM == 1) {
    // "dither" SR: |v| >= 1 exact carry trick (the rounded value is (ab + R) with the low 22 mantissa bits cleared = an exact e2m1 grid value); |v| < 1 (grid spacing 0.5): RN(|v| + delta),
    // delta = d 2^-18, d = 2h - 65535 odd in [-65535, 65535] (|delta| < 0.25 never crosses into a cell of another spacing): unbiased up to the 2^-17 granularity, same variance as the
    // u < p rule.  The result is a float on / next to the grid, packed by the hardware RN conversion.  NOT bit-identical to the u < p rule for the same random bits.
    float fv[16];
#pragma unroll
    for (int g = 0; g < 2; g++) {
      const uint4 rr = philox_t<NR>(f0 + g, w1, k0, k1);
      const uint32_t rw[4] = {rr.x, rr.y, rr.z, rr.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
          const int j = g * 8 + i * 2 + h2;
          const uint32_t hb = h2 == 0 ? (rw[i] & 0xffffu) : (rw[i] >> 16);
          const uint32_t ab = min(__float_as_uint(t[j]) & 0x7fffffffu, 0x40C00000u);
          const uint32_t hi_bits = (ab + 0x3FFFDFu - (hb << 6)) & 0xFFC00000u;
          const float noisy = fmaf((float)((int)(2u * hb) - 65535), 3.814697265625e-06f, __uint_as_float(ab));  // 2^-18
          const uint32_t sel = ab >= 0x3F800000u ? hi_bits : __float_as_uint(noisy);
          fv[j] = __uint_as_float(copysign_bits(sel, __float_as_uint(y[j])));
        }
      }
    }
    o.x = pack8_rn(&fv[0]);
    o.y = pack8_rn(&fv[8]);
  } else {
    uint32_t wd[2] = {0u, 0u};
#pragma unroll
    for (int g = 0; g < 2; g++) {
      const uint4 rr = philox_t<NR>(f0 + g, w1, k0, k1);
      const uint32_t rw[4] = {rr.x, rr.y, rr.z, rr.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int j = g * 8 + i * 2 + h;
          const uint32_t hb = h == 0 ? (rw[i] & 0xffffu) : (rw[i] >> 16);
          const uint32_t mag = sr_mag(__float_as_uint(t[j]), hb);
          wd[g] |= (mag << (4 * (i * 2 + h))) | ((__float_as_uint(y[j]) >> (28 - 4 * (i * 2 + h))) & (8u << (4 * (i * 2 + h))));
        }
      }
    }
    o.x = wd[0];
    o.y = wd[1];
  }
  return o;
}

template <bool ROW, bool COL, bool SR, int NR, int SRM>
__global__ void __launch_bounds__(128, ACT_MINB) actf_kernel(const ActParams p) {
  constexpr int LDW = 36;  // smem row = 64 bf16 = 32 words + 4 pad (conflict free 16 B stores of rows 2 apart)
  __shared__ __align__(16) uint32_t sm[COL ? 64 * LDW : 4];
  __shared__ unsigned s_red[8];
  uint32_t rmax = 0u, cmax = 0u;  // amax (bf16 bit pattern) of this thread's row / col blocks
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cb = blockIdx.x, ty = blockIdx.y;
  const __nv_bfloat16* __restrict__ X = reinterpret_cast<const __nv_bfloat16*>(p.x);
  const int row_base = ty * 64;
  const int nvalid = max(0, min(64, p.R - row_base));
  const bool want_row = ROW && (nvalid > 0 || row_base < p.rows_pad);
  // ---- load FIRST: thread = (row rr (+32 p2), 16-column block blk) of the 64 x 64 tile.  Nothing the loads depend on may wait for another global load or follow a global store
  // (the per-tensor scale loads / the ts_out store used to sit in front of the tile loads: a serialized ~2 x L2-latency chain per CTA, which was the real bound of round21's kernels).
  const int blk = tid & 3, rr = tid >> 2;
  const int col0 = cb * 64 + blk * 16;
  uint32_t w[2][8];
#pragma unroll
  for (int p2 = 0; p2 < 2; p2++) {
    const int i = rr + 32 * p2;
    if (i < nvalid) {
      const uint4* src = reinterpret_cast<const uint4*>(X + (size_t)(row_base + i) * p.C + col0);
      const uint4 a = __ldg(src), b = __ldg(src + 1);
      w[p2][0] = a.x; w[p2][1] = a.y; w[p2][2] = a.z; w[p2][3] = a.w;
      w[p2][4] = b.x; w[p2][5] = b.y; w[p2][6] = b.z; w[p2][7] = b.w;
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) w[p2][j] = 0u;
    }
  }
  // ---- per tensor scales, Philox key
  float ts_r = 1.f, ts_c = 1.f;
  if (p.ts_mode == 0) {
    ts_r = __ldg(p.tsrc);
    ts_c = __ldg(p.tsrc + 1);
  } else {
    ts_r = ts_of(__ldg(p.tsrc) * p.amax_mul, 0);
    ts_c = ts_of(__ldg(p.tsrc + 1) * p.amax_mul, 0);
  }
  uint32_t k0 = 0, k1 = 0;
  if constexpr (SR) {
    const int64_t s = __ldg(p.seed);
    k0 = (uint32_t)s;
    k1 = (uint32_t)((uint64_t)s >> 32);
  }
  if constexpr (COL) {
#pragma unroll
    for (int p2 = 0; p2 < 2; p2++) {
      const int swz = ((rr >> 4) & 1) << 4;  // word-column XOR 16 for odd 16-token groups: the col phase reads rows 16 apart in the same instruction (bank conflict free)
      uint4* d0 = reinterpret_cast<uint4*>(&sm[(rr + 32 * p2) * LDW + ((blk * 8) ^ swz)]);
      uint4* d1 = reinterpret_cast<uint4*>(&sm[(rr + 32 * p2) * LDW + ((blk * 8 + 4) ^ swz)]);
      *d0 = make_uint4(w[p2][0], w[p2][1], w[p2][2], w[p2][3]);
      *d1 = make_uint4(w[p2][4], w[p2][5], w[p2][6], w[p2][7]);
    }
  }
  // ---- row copy
  if constexpr (ROW) {
    if (want_row) {
      const TsF tc = make_ts(ts_r);
      const int rk = p.C >> 6;
#pragma unroll
      for (int p2 = 0; p2 < 2; p2++) {
        const int i = rr + 32 * p2;
        const int row = row_base + i;
        const uint32_t a16 = amax16_bits(w[p2]);
        rmax = max(rmax, a16);
#ifdef R24_SKEL
        const uint32_t sb = a16 & 0xffu;
        const float d = 0.f;
        uint2 o = make_uint2(w[p2][0] ^ w[p2][3], w[p2][1] ^ w[p2][5]);
#else
        const uint32_t sb = sb_fast(__uint_as_float(a16 << 16), tc);
        const float d = e4m3_to_f32(sb) * tc.ts;
        uint2 o = make_uint2(0u, 0u);
#endif
        if (d > 0.f) {
          float y[16];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            y[2 * j] = bf_lo(w[p2][j]);
            y[2 * j + 1] = bf_hi(w[p2][j]);
          }
          const float r = tc.ok ? __frcp_rn(d) : 0.f;
          o = quant16<SR, NR, SRM>(y, d, r, tc.ok, (uint32_t)(((size_t)row * p.C + col0) >> 3), p.sr_off * 2u, k0, k1);
        }
#ifndef R24_NOQ
        if (i < nvalid) *reinterpret_cast<uint2*>(p.q + (size_t)row * (p.C >> 1) + (col0 >> 1)) = o;
#endif
#ifndef R24_NOSF
        if (i < nvalid || row < p.rows_pad) p.sf[(size_t)((row >> 7) * rk + cb) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + blk] = (uint8_t)sb;
#endif
      }

    }
  }
  // ---- col copy: thread = (column pair cp, 16-token group g); lanes l / l + 16 = adjacent groups of the same columns, so every column row of qt gets 16 B per instruction
  if constexpr (COL) {
    __syncthreads();
    const TsF tc = make_ts(ts_c);
    const int cp = 16 * (warp & 1) + (lane & 15);
    const int g = 2 * (warp >> 1) + (lane >> 4);
    const int swz = (g & 1) << 4;
    uint32_t cw[16];
#pragma unroll
    for (int k = 0; k < 16; k++) cw[k] = sm[(g * 16 + k) * LDW + (cp ^ swz)];
    uint32_t m = cw[0] & 0x7fff7fffu;
#pragma unroll
    for (int k = 1; k < 16; k++) m = __vmaxu2(m, cw[k] & 0x7fff7fffu);
    const int rkc = p.Kt >> 6;
    const int kt0 = ty * 64 + g * 16;
#pragma unroll
    for (int e = 0; e < 2; e++) {
      const int col = cb * 64 + 2 * cp + e;
      const uint32_t a16 = e == 0 ? (m & 0xffffu) : (m >> 16);
      cmax = max(cmax, a16);
#ifdef R24_SKEL
      const uint32_t sb = a16 & 0xffu;
      const float d = 0.f;
      uint2 o = make_uint2(cw[0] ^ cw[3], cw[1] ^ cw[5]);
#else
      const uint32_t sb = sb_fast(__uint_as_float(a16 << 16), tc);
      const float d = e4m3_to_f32(sb) * tc.ts;
      uint2 o = make_uint2(0u, 0u);
#endif
      if (d > 0.f) {
        float y[16];
#pragma unroll
        for (int k = 0; k < 16; k++) y[k] = e == 0 ? bf_lo(cw[k]) : bf_hi(cw[k]);
        const float r = tc.ok ? __frcp_rn(d) : 0.f;
        o = quant16<SR, NR, SRM>(y, d, r, tc.ok, (uint32_t)(((size_t)col * p.Kt + kt0) >> 3), p.sr_off * 2u + 1u, k0, k1);
      }
      *reinterpret_cast<uint2*>(p.qt + (size_t)col * (p.Kt >> 1) + (kt0 >> 1)) = o;
      p.sft[(size_t)((col >> 7) * rkc + ty) * 512 + (col & 31) * 16 + ((col >> 5) & 3) * 4 + g] = (uint8_t)sb;
      // rows >= C of the last 128-row atom (C % 128 == 64) must be zero bytes
      if ((p.C & 127) == 64 && cb == (p.C >> 6) - 1)
        p.sft[(size_t)(((col + 64) >> 7) * rkc + ty) * 512 + ((col + 64) & 31) * 16 + (((col + 64) >> 5) & 3) * 4 + g] = 0;
    }
  }
  // ---- amax publication.  One request per CTA and copy, spread over NPART sectors (a single hot line serialised every warp of the grid: it was the bound of the whole kernel)
  if (p.amax_parts) {
    rmax = warp_max_u(rmax);
    cmax = warp_max_u(cmax);
    if (lane == 0) {
      s_red[warp] = rmax;
      s_red[4 + warp] = cmax;
    }
    __syncthreads();
    if (tid == 0) {
      const int slot = (blockIdx.y * gridDim.x + blockIdx.x) % NPART;
      if (ROW) atomicMax(p.amax_parts + slot * 8, max(max(s_red[0], s_red[1]), max(s_red[2], s_red[3])) << 16);
      if (COL) atomicMax(p.amax_parts + (NPART + slot) * 8, max(max(s_red[4], s_red[5]), max(s_red[6], s_red[7])) << 16);
    }
  } else if (p.amax_next) {
    if (ROW) {
      rmax = warp_max_u(rmax);
      if (lane == 0) amax_publish(p.amax_next + 0, rmax << 16);
    }
    if (COL) {
      cmax = warp_max_u(cmax);
      if (lane == 0) amax_publish(p.amax_next + 1, cmax << 16);
    }
  }
  if (cb == 0 && ty == 0 && tid == 0 && p.ts_out) {
    p.ts_out[0] = ts_r;
    p.ts_out[1] = ts_c;
  }
}

// ------------------------------------------------------------------------------------------------ weights: 2D block scale, RN, both copies from one read
struct WParams {
  const void* w;
  uint8_t *q, *sf, *qt, *sft;
  const float* tsrc;
  float* ts_out;
  unsigned* amax_next;
  int N, K, ts_mode, div_true, amax_only;
  float amax_mul;
};

template <typename T, int FMT, bool WROW>
__global__ void __launch_bounds__(128) w_kernel(const WParams p) {
  constexpr int V = FMT == 0 ? 16 : 32;
  constexpr int NB = 64 / V;  // blocks per tile side
  constexpr int LD = 68;
  __shared__ __align__(16) float sm[64 * LD];
  __shared__ uint32_t sc[64 * 8];
  __shared__ uint8_t sS[16];
  const int tid = threadIdx.x, lane = tid & 31;
  const int cb = blockIdx.x, ty = blockIdx.y;  // cb: K tile, ty: N tile
  const T* W = reinterpret_cast<const T*>(p.w);
  const int cg = tid & 7, rg = tid >> 3;
  const int n0 = ty * 64, k0 = cb * 64;
#pragma unroll
  for (int pp = 0; pp < 4; pp++) {
    const int i = pp * 16 + rg;
    float v[8];
    load8(W + (size_t)(n0 + i) * p.K + k0 + cg * 8, v);
    float4* d = reinterpret_cast<float4*>(&sm[i * LD + cg * 8]);
    d[0] = make_float4(v[0], v[1], v[2], v[3]);
    d[1] = make_float4(v[4], v[5], v[6], v[7]);
  }
  __syncthreads();
  // block amax of the 2D tiles
  unsigned a = 0u;
  int blk, br, bc;
  if constexpr (FMT == 0) {
    blk = tid >> 3;
    br = blk >> 2;
    bc = blk & 3;
    const int sub = tid & 7;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const float* rowp = &sm[(br * 16 + sub * 2 + r) * LD + bc * 16];
#pragma unroll
      for (int j = 0; j < 16; j++) a = max(a, __float_as_uint(fabsf(rowp[j])));
    }
    a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, 1));
    a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, 2));
    a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, 4));
  } else {
    blk = tid >> 5;
    br = blk >> 1;
    bc = blk & 1;
    const float* rowp = &sm[(br * 32 + lane) * LD + bc * 32];
#pragma unroll
    for (int j = 0; j < 32; j++) a = max(a, __float_as_uint(fabsf(rowp[j])));
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) a = max(a, (unsigned)__shfl_xor_sync(FULL_MASK, (int)a, o));
  }
  if (p.amax_only) {
    const unsigned wm = warp_max_u(a);
    if (lane == 0 && p.amax_next) amax_publish(p.amax_next, wm);
    return;
  }
  float ts = 1.f;
  if constexpr (FMT == 0) {
    ts = p.ts_mode == 0 ? p.tsrc[0] : ts_of(p.tsrc[0] * p.amax_mul, p.div_true);
    if (cb == 0 && ty == 0 && tid == 0 && p.ts_out) p.ts_out[0] = ts;
  }
  uint32_t sbyte;
  if constexpr (FMT == 0) sbyte = nv4_sf_byte(__uint_as_float(a), ts, p.div_true);
  else sbyte = mx4_byte(__uint_as_float(a), p.div_true);
  if ((FMT == 0 && (tid & 7) == 0) || (FMT == 1 && lane == 0)) sS[blk] = (uint8_t)sbyte;
  if (p.amax_next && FMT == 0) {
    const unsigned wm = warp_max_u(a);
    if (lane == 0) amax_publish(p.amax_next, wm);
  }
  __syncthreads();
  // row copy
  const int rkq = FMT == 0 ? (p.K >> 6) : (p.K >> 7);
#pragma unroll
  for (int pp = 0; pp < 4; pp++) {
    const int i = pp * 16 + rg;
    const int n = n0 + i;
    const int brr = i / V, bcc = (cg * 8) / V;
    const uint32_t sb = sS[brr * NB + bcc];
    float y[8];
    {
      const float4* s = reinterpret_cast<const float4*>(&sm[i * LD + cg * 8]);
      float4 x0 = s[0], x1 = s[1];
      y[0] = x0.x; y[1] = x0.y; y[2] = x0.z; y[3] = x0.w; y[4] = x1.x; y[5] = x1.y; y[6] = x1.z; y[7] = x1.w;
    }
    uint32_t w = 0u;
    float t8[8];
    if constexpr (FMT == 0) {
      const float denom = e4m3_to_f32(sb) * ts;
      if (denom > 0.f) {
        div_vec<8>(y, t8, denom);
        w = pack8_rn(t8);
      }
    } else {
      const float inv = mx4_inv(sb);
#pragma unroll
      for (int j = 0; j < 8; j++) t8[j] = y[j] * inv;
      w = pack8_rn(t8);
    }
    if constexpr (WROW) *reinterpret_cast<uint32_t*>(p.q + (size_t)n * (p.K >> 1) + ((k0 + cg * 8) >> 1)) = w;
    sc[i * 8 + cg] = w;
    if (WROW && cg == 0) {
      if constexpr (FMT == 0) {
        const uint32_t word = sS[brr * 4 + 0] | (sS[brr * 4 + 1] << 8) | (sS[brr * 4 + 2] << 16) | ((uint32_t)sS[brr * 4 + 3] << 24);
        reinterpret_cast<uint32_t*>(p.sf)[(size_t)((n >> 7) * rkq + cb) * 128 + (n & 31) * 4 + ((n >> 5) & 3)] = word;
      } else {
        uint8_t* base = p.sf + (size_t)((n >> 7) * rkq + (cb >> 1)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (cb & 1) * 2;
        *reinterpret_cast<uint16_t*>(base) = (uint16_t)(sS[brr * 2] | (sS[brr * 2 + 1] << 8));
      }
    }
  }
  __syncthreads();
  // col copy (W^T rows = k, contraction along n), codes are the transposed codes of the row copy
  {
    const int c = tid & 63, half = tid >> 6;
    const int kcol = k0 + c;
    uint32_t b4[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int kk = 0; kk < 32; kk++) {
      const uint32_t nib = (sc[(half * 32 + kk) * 8 + (c >> 3)] >> ((c & 7) * 4)) & 0xFu;
      b4[kk >> 3] |= nib << ((kk & 7) * 4);
    }
    *reinterpret_cast<uint4*>(p.qt + (size_t)kcol * (p.N >> 1) + ((n0 + half * 32) >> 1)) = make_uint4(b4[0], b4[1], b4[2], b4[3]);
    const int rkt = FMT == 0 ? (p.N >> 6) : (p.N >> 7);
    const int bcc = c / V;
    if constexpr (FMT == 0) {
      const int brb = half * 2;
      const uint16_t s2 = (uint16_t)(sS[(brb)*4 + bcc] | (sS[(brb + 1) * 4 + bcc] << 8));
      uint8_t* base = p.sft + (size_t)((kcol >> 7) * rkt + ty) * 512 + (kcol & 31) * 16 + ((kcol >> 5) & 3) * 4 + half * 2;
      *reinterpret_cast<uint16_t*>(base) = s2;
    } else {
      const int bn = ty * 2 + half;
      uint8_t* base = p.sft + (size_t)((kcol >> 7) * rkt + (bn >> 2)) * 512 + (kcol & 31) * 16 + ((kcol >> 5) & 3) * 4 + (bn & 3);
      *base = sS[half * 2 + bcc];
    }
  }
}

// ------------------------------------------------------------------------------------------------ small kernels
template <typename T>
__global__ void amax_abs_kernel(const T* __restrict__ x, size_t n8, unsigned* __restrict__ out) {
  unsigned m = 0u;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
    float v[8];
    load8(x + i * 8, v);
#pragma unroll
    for (int j = 0; j < 8; j++) m = max(m, __float_as_uint(fabsf(v[j])));
  }
  m = warp_max_u(m);
  if ((threadIdx.x & 31) == 0) amax_publish(out, m);
}

__global__ void amax_roll_kernel(float* amax, int n, int64_t* seed, int64_t* step) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    amax[i] = amax[n + i];
    amax[n + i] = 0.f;
  }
  if (i == 0) {
    seed[0] += 1;
    step[0] += 1;
  }
}

// roll + saturation monitor (not part of the quantizers).  Slot i = site*2 + copy (0 row copy / weight, 1 transposed copy).  At the roll of step t+1, `prev` is the amax that scaled step t
// (ts = headroom * prev / 2688) and `next` is the amax step t actually produced: the nv4 block scale saturates (e4m3 448) iff next > prev * headroom.  Per slot:
//   cnt[0][i] += 1 (comparisons), cnt[1][i] += (next > prev * headroom), maxr[i] = max(maxr[i], next / prev)  (prev == 0, next > 0: counted as exceeded with ratio FLT_MAX; both 0: skipped).
// One thread per slot, so plain read-modify-write (no atomics); roll semantics (prev <- next, next <- 0, seed += 1, step += 1) are identical to amax_roll_kernel.
__global__ void amax_roll_mon_kernel(float* amax, int n, const float* headroom, int* cnt, float* maxr, int64_t* seed, int64_t* step, unsigned* parts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float nxt = amax[n + i];
    if (parts) {  // fold the spread publication (R24 fast kernels) of this slot, then clear it
      unsigned m = 0u;
#pragma unroll
      for (int j = 0; j < NPART; j++) {
        unsigned* q = parts + ((size_t)i * NPART + j) * 8;
        m = max(m, *q);
        *q = 0u;
      }
      nxt = fmaxf(nxt, __uint_as_float(m));
    }
    const float prev = amax[i];
    if (prev > 0.f) {
      cnt[i] += 1;
      if (nxt > prev * headroom[i >> 1]) cnt[n + i] += 1;
      maxr[i] = fmaxf(maxr[i], nxt / prev);
    } else if (nxt > 0.f) {
      cnt[i] += 1;
      cnt[n + i] += 1;
      maxr[i] = 3.402823466e38f;
    }
    amax[i] = nxt;
    amax[n + i] = 0.f;
  }
  if (i == 0) {
    seed[0] += 1;
    step[0] += 1;
  }
}

__global__ void philox_uniform_kernel(float* out, int64_t n, const int64_t* seed, uint32_t w1, int nr) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;  // 8 elements per thread
  if (i * 8 >= n) return;
  const int64_t s = seed ? *seed : 0;
  float u[8];
  get_u8(u, nullptr, (size_t)(i * 8), (uint32_t)s, (uint32_t)((uint64_t)s >> 32), w1, nr);
#pragma unroll
  for (int j = 0; j < 8; j++)
    if (i * 8 + j < n) out[i * 8 + j] = u[j];
}

// brute-force check of div_vec's formula against IEEE division: d random (all mantissa patterns incl. all-ones / zero / 1), y near the e2m1 decision thresholds * d (+-3 ulp), uniform in [0, 7] d, random
__global__ void div_check_kernel(unsigned long long* bad, unsigned* first, uint32_t seed, int iters) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long nb = 0;
  const float thr[8] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 6.0f};
  for (int it = 0; it < iters; it++) {
    const uint4 r = philox4x32_10(gid, (uint32_t)it, 7u, 0u, seed, 0x1234u);
    const int e = (int)(r.x % 121u) - 60;
    uint32_t mant = r.y & 0x7fffffu;
    const uint32_t sel = r.z & 31u;
    if (sel == 0) mant = 0x7fffffu; else if (sel == 1) mant = 0u; else if (sel == 2) mant = 0x7ffffeu; else if (sel == 3) mant = 1u;
    const float d = __uint_as_float(((uint32_t)(e + 127) << 23) | mant);
    const uint32_t m = r.w;
    float y;
    const int mode = m & 3;
    if (mode == 0) y = __fmul_rn(d, thr[(m >> 2) & 7]);
    else if (mode == 1) y = __fmul_rn(d, (float)(r.z >> 8) * 5.9604645e-08f * 7.0f);
    else y = __uint_as_float(((uint32_t)(e + 127 + (int)((m >> 5) & 7) - 3) << 23) | (r.z & 0x7fffffu));
    y = __uint_as_float(__float_as_uint(y) + (uint32_t)(((m >> 8) & 7u)) - 3u);
    if ((__float_as_uint(y) & 0x7fffffffu) < 0x00800000u) continue;  // subnormal / zero y (|v| < 1e-30: code 0 either way; the sign of zero is restored by div_vec)
    const float rr = __frcp_rn(d);
    const float q = __fmul_rn(y, rr);
    const float res = fmaf(-d, q, y);
    const float t = fmaf(res, rr, q);
    const bool mm = __float_as_uint(t) != __float_as_uint(__fdiv_rn(y, d));
    nb += mm;
    if (mm && atomicCAS(&first[0], 0u, 1u) == 0u) { first[1] = __float_as_uint(y); first[2] = __float_as_uint(d); first[3] = __float_as_uint(t); first[4] = __float_as_uint(__fdiv_rn(y, d)); }
  }
  atomicAdd(bad, nb);
}

// exhaustive check of the hardware RN conversion against the software decision thresholds of fp4_ref.e2m1_rn over ALL 2^32 fp32 bit patterns (NaN excluded)
__global__ void rn_check_kernel(unsigned long long* bad) {
  unsigned long long nb = 0;
  for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < (1ull << 32); k += (uint64_t)gridDim.x * blockDim.x) {
    const float v = __uint_as_float((uint32_t)k);
    if (v != v) continue;
    float t[8] = {v, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const uint32_t hw = pack8_rn(t) & 0xFu;
    const float a = fabsf(v);
    const uint32_t c = (uint32_t)(a > 0.25f) + (uint32_t)(a >= 0.75f) + (uint32_t)(a > 1.25f) + (uint32_t)(a >= 1.75f) + (uint32_t)(a > 2.5f) + (uint32_t)(a >= 3.5f) + (uint32_t)(a > 5.0f);
    nb += (hw != (c | ((__float_as_uint(v) >> 28) & 8u)));
  }
  atomicAdd(bad, nb);
}
__global__ void sr_codes_kernel(const float* v, const float* u, uint8_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (uint8_t)e2m1_sr_code(v[i], u[i]);
}

// test helper: the two SR rules on given |v| (fp32) and 16 random bits h: exact (u = (2h+1)/2^17 through sr_mag) and dither
__global__ void sr_rules_kernel(const float* v, const int* h, uint8_t* out_exact, uint8_t* out_dither, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t hb = (uint32_t)h[i];
  const uint32_t ab = min(__float_as_uint(v[i]) & 0x7fffffffu, 0x40C00000u);
  out_exact[i] = (uint8_t)sr_mag(ab, hb);
  const uint32_t hi_bits = (ab + 0x3FFFDFu - (hb << 6)) & 0xFFC00000u;
  const float noisy = fmaf((float)((int)(2u * hb) - 65535), 3.814697265625e-06f, __uint_as_float(ab));
  const uint32_t sel = ab >= 0x3F800000u ? hi_bits : __float_as_uint(noisy);
  float t[8] = {__uint_as_float(sel), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  out_dither[i] = (uint8_t)(pack8_rn(t) & 0xFu);
}
std::vector<at::Tensor> sr_rules(const at::Tensor& v, const at::Tensor& h) {
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kFloat && h.is_contiguous() && h.scalar_type() == at::kInt && v.numel() == h.numel(), "v fp32, h int32, same size");
  at::Tensor a = at::empty({v.numel()}, v.options().dtype(at::kByte)), b = at::empty({v.numel()}, v.options().dtype(at::kByte));
  const int n = (int)v.numel();
  sr_rules_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<float>(), h.data_ptr<int>(), a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b};
}

// ------------------------------------------------------------------------------------------------ host
static inline int64_t pad128(int64_t n) { return (n + 127) / 128 * 128; }
static const float* fptr(const c10::optional<at::Tensor>& t) { return t.has_value() ? t->data_ptr<float>() : nullptr; }

static bool g_fast = true;  // set_fast(false): generic kernel only (A/B + tests)
template <typename T>
static void launch_act(const ActParams& p, int fmt, bool row, bool col, dim3 grid, cudaStream_t s) {
#define LA(F, R, C) act_kernel<T, F, R, C><<<grid, 128, 0, s>>>(p)
  if (fmt == 0) {
    if (row && col) LA(0, true, true);
    else if (row) LA(0, true, false);
    else LA(0, false, true);
  } else {
    if (row && col) LA(1, true, true);
    else if (row) LA(1, true, false);
    else LA(1, false, true);
  }
#undef LA
}
// fast kernels: nv4, bf16, no RHT / prefix / explicit u / div_true / amax_only, SR (Philox) of the copies equal when both are produced
static bool fast_ok(const ActParams& p, int fmt, bool row, bool col, bool is_bf16) {
  if (!g_fast || fmt != 0 || !is_bf16 || p.npre != 0 || p.rht_r || p.rht_c || p.div_true || p.amax_only || p.u_row || p.u_col) return false;
  if (row && col && (p.sr_r != 0) != (p.sr_c != 0)) return false;
  return true;
}
static void launch_fast(const ActParams& p, bool row, bool col, dim3 grid, cudaStream_t s) {
  const bool sr = row && col ? p.sr_r != 0 : (row ? p.sr_r != 0 : p.sr_c != 0);
  TORCH_CHECK(!sr || p.nr == 10 || p.nr == 7, "fast kernels: philox_rounds must be 10 or 7");
  TORCH_CHECK(!p.srm || sr, "sr_mode dither needs sr_row / sr_col");
#define LF(R, C, SR, NR, M) actf_kernel<R, C, SR, NR, M><<<grid, 128, 0, s>>>(p)
  if (!sr) {
    if (row && col) LF(true, true, false, 10, 0);
    else if (row) LF(true, false, false, 10, 0);
    else LF(false, true, false, 10, 0);
  } else if (p.nr == 10) {
    if (p.srm) {
      if (row && col) LF(true, true, true, 10, 1);
      else if (row) LF(true, false, true, 10, 1);
      else LF(false, true, true, 10, 1);
    } else {
      if (row && col) LF(true, true, true, 10, 0);
      else if (row) LF(true, false, true, 10, 0);
      else LF(false, true, true, 10, 0);
    }
  } else {
    if (p.srm) {
      if (row && col) LF(true, true, true, 7, 1);
      else if (row) LF(true, false, true, 7, 1);
      else LF(false, true, true, 7, 1);
    } else {
      if (row && col) LF(true, true, true, 7, 0);
      else if (row) LF(true, false, true, 7, 0);
      else LF(false, true, true, 7, 0);
    }
  }
#undef LF
}
void set_fast(bool v) { g_fast = v; }

// returns {q, sf, qt, sft, xp, ts_out}; unused entries are empty tensors.
//   ts_mode 0: tsrc = [ts_row, ts_col] given;  1: tsrc = [amax_row, amax_col] and ts = f(amax * amax_mul)
//   amax_next: [2] uint32 view of float amax (row, col) updated with atomicMax;  amax_only: only compute amax (no outputs)
std::vector<at::Tensor> quant_act(const at::Tensor& x, int64_t fmt, bool row, bool col, int64_t rpad, int64_t B, int64_t N, int64_t npre, bool rht_r,
                                  bool rht_c, int64_t sign_mask, bool sr_r, bool sr_c, c10::optional<at::Tensor> u_row, c10::optional<at::Tensor> u_col,
                                  c10::optional<at::Tensor> seed, int64_t sr_off, int64_t ts_mode, c10::optional<at::Tensor> tsrc, double amax_mul,
                                  c10::optional<at::Tensor> amax_next, bool div_true, bool amax_only, int64_t philox_rounds, c10::optional<at::Tensor> amax_parts, int64_t sr_mode) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous CUDA matrix");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "bf16 / fp32 only");
  TORCH_CHECK(fmt == 0 || fmt == 1, "fmt: 0 nv4, 1 mx4");
  TORCH_CHECK(row || col, "nothing to do");
  const int64_t R = x.size(0), C = x.size(1);
  TORCH_CHECK(C % 64 == 0, "C must be a multiple of 64");
  const int vec = fmt == 0 ? 16 : 32;
  int64_t kt, P = 0, Tp = 0, PT = 0, ntiles;
  const int64_t rows_pad = pad128(R);
  if (npre > 0) {
    TORCH_CHECK(R == B * N && N > npre, "prefix: R must be B*N");
    P = N - npre;
    TORCH_CHECK(P % 64 == 0 && (B * P) % 128 == 0, "prefix: patch tokens per image must be a multiple of 64 and B*P of 128");
    kt = B * P;
    Tp = B * (P / 64);
    PT = (npre + 63) / 64;
    ntiles = Tp + B * PT + (rows_pad - R + 63) / 64;
  } else {
    kt = rpad > 0 ? rpad : rows_pad;
    TORCH_CHECK(kt % 128 == 0 && kt >= R, "rpad must be a multiple of 128 and >= R");
    ntiles = kt / 64;
    if (ntiles * 64 < rows_pad) ntiles = rows_pad / 64;
  }
  c10::cuda::CUDAGuard guard(x.device());
  auto u8 = x.options().dtype(at::kByte);
  auto empty = [&]() { return at::empty({0}, u8); };
  at::Tensor q = empty(), sf = empty(), qt = empty(), sft = empty(), xp = at::empty({0}, x.options().dtype(at::kBFloat16)), tso = at::empty({0}, x.options().dtype(at::kFloat));
  ActParams p{};
  p.x = x.data_ptr();
  if (!amax_only) {
    if (row) {
      const int64_t rk = fmt == 0 ? C / 64 : (C + 127) / 128;
      q = at::empty({R, C / 2}, u8);
      sf = at::empty({rows_pad / 128, rk, 32, 4, 4}, u8);
    }
    if (col) {
      const int64_t rkc = fmt == 0 ? kt / 64 : kt / 128;
      qt = at::empty({C, kt / 2}, u8);
      sft = at::empty({(C + 127) / 128, rkc, 32, 4, 4}, u8);
    }
    if (npre > 0) xp = at::empty({B * npre, C}, x.options().dtype(at::kBFloat16));
    if (fmt == 0) tso = at::empty({2}, x.options().dtype(at::kFloat));
  }
  p.q = row && !amax_only ? (uint8_t*)q.data_ptr() : nullptr;
  p.sf = row && !amax_only ? (uint8_t*)sf.data_ptr() : nullptr;
  p.qt = col && !amax_only ? (uint8_t*)qt.data_ptr() : nullptr;
  p.sft = col && !amax_only ? (uint8_t*)sft.data_ptr() : nullptr;
  p.xp = (npre > 0 && !amax_only) ? reinterpret_cast<__nv_bfloat16*>(xp.data_ptr()) : nullptr;
  p.u_row = fptr(u_row);
  p.u_col = fptr(u_col);
  p.tsrc = fptr(tsrc);
  p.ts_out = (fmt == 0 && !amax_only) ? tso.data_ptr<float>() : nullptr;
  p.amax_next = amax_next.has_value() ? reinterpret_cast<unsigned*>(amax_next->data_ptr()) : nullptr;
  p.amax_parts = nullptr;
  if (amax_parts.has_value()) {
    TORCH_CHECK(amax_parts->scalar_type() == at::kFloat && amax_parts->is_contiguous() && amax_parts->numel() == 2 * NPART * 8, "amax_parts: [2, NPART, 8] fp32");
    p.amax_parts = reinterpret_cast<unsigned*>(amax_parts->data_ptr());
  }
  p.seed = seed.has_value() ? seed->data_ptr<int64_t>() : nullptr;
  p.R = (int)R; p.C = (int)C; p.Kt = (int)kt; p.rows_pad = (int)rows_pad;
  p.B = (int)B; p.N = (int)N; p.npre = (int)npre; p.P = (int)P; p.Tp = (int)Tp; p.PT = (int)PT;
  p.ts_mode = (int)ts_mode; p.div_true = div_true; p.rht_r = rht_r; p.rht_c = rht_c; p.sr_r = sr_r; p.sr_c = sr_c; p.amax_only = amax_only; p.nr = (int)philox_rounds; p.srm = (int)sr_mode;
  p.sign_mask = (uint32_t)sign_mask; p.sr_off = (uint32_t)sr_off; p.amax_mul = (float)amax_mul;
  if (!amax_only && fmt == 0) TORCH_CHECK(p.tsrc != nullptr, "nv4 needs tsrc (ts_mode 0 / 1)");
  if (sr_r) TORCH_CHECK(p.u_row != nullptr || p.seed != nullptr, "SR needs seed or explicit u");
  if (sr_c) TORCH_CHECK(p.u_col != nullptr || p.seed != nullptr, "SR needs seed or explicit u");
  dim3 grid((unsigned)(C / 64), (unsigned)ntiles);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool is_bf16 = x.scalar_type() == at::kBFloat16;
  TORCH_CHECK(sr_mode == 0 || fast_ok(p, (int)fmt, row, col, is_bf16), "sr_mode='dither' needs the fast kernel (nv4, bf16 input, no RHT / prefix / explicit u, equal SR flags)");
  if (fast_ok(p, (int)fmt, row, col, is_bf16)) launch_fast(p, row, col, grid, stream);
  else if (is_bf16) launch_act<__nv_bfloat16>(p, (int)fmt, row, col, grid, stream);
  else launch_act<float>(p, (int)fmt, row, col, grid, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)vec;
  return {q, sf, qt, sft, xp, tso};
}

std::vector<at::Tensor> quant_weight(const at::Tensor& w, int64_t fmt, int64_t ts_mode, c10::optional<at::Tensor> tsrc, double amax_mul,
                                     c10::optional<at::Tensor> amax_next, bool div_true, bool amax_only, bool row_copy) {
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.is_contiguous(), "w must be a contiguous CUDA matrix");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 || w.scalar_type() == at::kFloat, "bf16 / fp32 only");
  const int64_t N = w.size(0), K = w.size(1);
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0, "weights: N and K must be multiples of 128");
  c10::cuda::CUDAGuard guard(w.device());
  auto u8 = w.options().dtype(at::kByte);
  at::Tensor q = at::empty({0}, u8), sf = at::empty({0}, u8), qt = at::empty({0}, u8), sft = at::empty({0}, u8), tso = at::empty({0}, w.options().dtype(at::kFloat));
  WParams p{};
  p.w = w.data_ptr();
  if (!amax_only) {
    const int64_t rkq = fmt == 0 ? K / 64 : K / 128, rkt = fmt == 0 ? N / 64 : N / 128;
    if (row_copy) {
      q = at::empty({N, K / 2}, u8);
      sf = at::empty({N / 128, rkq, 32, 4, 4}, u8);
    }
    qt = at::empty({K, N / 2}, u8);
    sft = at::empty({K / 128, rkt, 32, 4, 4}, u8);
    if (fmt == 0) tso = at::empty({1}, w.options().dtype(at::kFloat));
    p.q = row_copy ? (uint8_t*)q.data_ptr() : nullptr; p.sf = row_copy ? (uint8_t*)sf.data_ptr() : nullptr; p.qt = (uint8_t*)qt.data_ptr(); p.sft = (uint8_t*)sft.data_ptr();
    p.ts_out = fmt == 0 ? tso.data_ptr<float>() : nullptr;
  }
  p.tsrc = fptr(tsrc);
  p.amax_next = amax_next.has_value() ? reinterpret_cast<unsigned*>(amax_next->data_ptr()) : nullptr;
  p.N = (int)N; p.K = (int)K; p.ts_mode = (int)ts_mode; p.div_true = div_true; p.amax_only = amax_only; p.amax_mul = (float)amax_mul;
  if (!amax_only && fmt == 0) TORCH_CHECK(p.tsrc != nullptr, "nv4 needs tsrc");
  dim3 grid((unsigned)(K / 64), (unsigned)(N / 64));
  auto stream = at::cuda::getCurrentCUDAStream();
#define LW(T, F) do { if (row_copy || amax_only) w_kernel<T, F, true><<<grid, 128, 0, stream>>>(p); else w_kernel<T, F, false><<<grid, 128, 0, stream>>>(p); } while (0)
  if (w.scalar_type() == at::kFloat) { if (fmt == 0) LW(float, 0); else LW(float, 1); }
  else { if (fmt == 0) LW(__nv_bfloat16, 0); else LW(__nv_bfloat16, 1); }
#undef LW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, sf, qt, sft, tso};
}

// amax_out: [1] uint32 (float bits) updated with atomicMax of |x|
void amax_abs(const at::Tensor& x, at::Tensor amax_out) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.numel() % 8 == 0, "x must be contiguous, numel % 8 == 0");
  c10::cuda::CUDAGuard guard(x.device());
  const size_t n8 = x.numel() / 8;
  int blocks = (int)std::min<size_t>((n8 + 255) / 256, 170 * 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned* o = reinterpret_cast<unsigned*>(amax_out.data_ptr());
  if (x.scalar_type() == at::kFloat) amax_abs_kernel<float><<<blocks, 256, 0, stream>>>(x.data_ptr<float>(), n8, o);
  else amax_abs_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), n8, o);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void amax_roll(at::Tensor amax, at::Tensor seed, at::Tensor step) {
  TORCH_CHECK(amax.is_cuda() && amax.dim() == 3 && amax.size(0) == 2 && amax.is_contiguous(), "amax: [2, sites, 2] fp32");
  c10::cuda::CUDAGuard guard(amax.device());
  const int n = (int)(amax.numel() / 2);
  amax_roll_kernel<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(amax.data_ptr<float>(), n, seed.data_ptr<int64_t>(), step.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void amax_roll_mon(at::Tensor amax, at::Tensor headroom, at::Tensor cnt, at::Tensor maxr, at::Tensor seed, at::Tensor step, c10::optional<at::Tensor> parts) {
  TORCH_CHECK(amax.is_cuda() && amax.dim() == 3 && amax.size(0) == 2 && amax.size(2) == 2 && amax.is_contiguous(), "amax: [2, sites, 2] fp32");
  const int n = (int)(amax.numel() / 2);
  TORCH_CHECK(headroom.scalar_type() == at::kFloat && headroom.numel() == n / 2, "headroom: [sites] fp32");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() == 2 * n && maxr.scalar_type() == at::kFloat && maxr.numel() == n, "cnt: [2, 2*sites] int32, maxr: [2*sites] fp32");
  if (parts.has_value()) TORCH_CHECK(parts->scalar_type() == at::kFloat && parts->is_contiguous() && parts->numel() == (int64_t)n * NPART * 8, "parts: [sites*2, NPART, 8] fp32");
  c10::cuda::CUDAGuard guard(amax.device());
  amax_roll_mon_kernel<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(amax.data_ptr<float>(), n, headroom.data_ptr<float>(), cnt.data_ptr<int>(), maxr.data_ptr<float>(),
                                                                                      seed.data_ptr<int64_t>(), step.data_ptr<int64_t>(), parts.has_value() ? reinterpret_cast<unsigned*>(parts->data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// u[i] for flat i in [0, n): Philox counter (i/8, w1, 0, 0), 16 bit halves, key = seed (the exact stream the SR path of quant_act uses: w1 = sr_off*2 (row copy) / sr_off*2+1 (col copy))
at::Tensor philox_uniform(int64_t n, c10::optional<at::Tensor> seed, int64_t w1, c10::Device dev, int64_t rounds) {
  c10::cuda::CUDAGuard guard(dev);
  at::Tensor out = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(dev));
  const int64_t g = (n + 7) / 8;
  philox_uniform_kernel<<<(g + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(), n, seed.has_value() ? seed->data_ptr<int64_t>() : nullptr, (uint32_t)w1, (int)rounds);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<int64_t> div_check(int64_t iters, int64_t seed) {
  at::Tensor bad = at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  at::Tensor first = at::zeros({5}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  div_check_kernel<<<170 * 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned long long*>(bad.data_ptr()), reinterpret_cast<unsigned*>(first.data_ptr()), (uint32_t)seed, (int)iters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto f = first.cpu();
  return {bad.item<int64_t>(), iters * 170 * 8 * 256, (int64_t)(uint32_t)f[1].item<int>(), (int64_t)(uint32_t)f[2].item<int>(), (int64_t)(uint32_t)f[3].item<int>(), (int64_t)(uint32_t)f[4].item<int>()};
}

int64_t rn_check() {
  at::Tensor bad = at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  rn_check_kernel<<<170 * 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned long long*>(bad.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bad.item<int64_t>();
}
at::Tensor sr_codes(const at::Tensor& v, const at::Tensor& u) {
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && u.is_contiguous() && v.scalar_type() == at::kFloat && u.scalar_type() == at::kFloat && v.numel() == u.numel(), "v, u: contiguous fp32 cuda");
  at::Tensor out = at::empty({v.numel()}, v.options().dtype(at::kByte));
  const int n = (int)v.numel();
  sr_codes_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<float>(), u.data_ptr<float>(), out.data_ptr<uint8_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rn_check", &rn_check);
  m.def("sr_codes", &sr_codes);
  m.def("div_check", &div_check);
  m.def("quant_act", &quant_act);
  m.def("quant_weight", &quant_weight);
  m.def("amax_abs", &amax_abs);
  m.def("amax_roll", &amax_roll);
  m.def("amax_roll_mon", &amax_roll_mon);
  m.def("philox_uniform", &philox_uniform);
  m.def("set_fast", &set_fast);
  m.def("sr_rules", &sr_rules);
}
