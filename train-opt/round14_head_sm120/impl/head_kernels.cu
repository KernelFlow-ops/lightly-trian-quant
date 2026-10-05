// round14 fused EoMT-head stage kernels (sm_120, CUDA C++):  GELU -> depthwise 3x3 -> LayerNorm2D, forward and backward.
//
// Same math / layouts as head_triton.py (see its docstring):
//   Z  [B*h*w, 4C] bf16   ConvTranspose(k2,s2) GEMM output without bias, "pixel shuffle on read":
//                         output pixel (b,P,Q), P<2h, Q<2w  lives at row (b*h + P/2)*w + Q/2, columns ((P%2)*2 + Q%2)*C .. +C
//   Y  [B*H*W, C]  bf16   LN2D output (plain NHWC tokens), H = 2h, W = 2w
//
// One CTA streams a segment of one image row (b, P, q0..q0+PW) along Q.  Thread t owns CPT consecutive channels of EVERY pixel
// (CTA = C/CPT threads), so the per-pixel LayerNorm statistics are one block reduction.  The 3x3 stencil is evaluated by
// column-wise accumulation (each loaded column of 3 pixels feeds 3 output pixels), so only 2 partial outputs live in registers,
// and the GELU (single exp, A&S 7.1.26 erf, |err| < 1.5e-7) is evaluated once per loaded element.
// All reductions are fp32, no data atomics: per-CTA partials are written to a workspace and summed by the caller (deterministic).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
#define FULL_MASK 0xffffffffu

// ------------------------------------------------------------------------------------------------ helpers
template <int CPT>
struct Raw;  // CPT bf16 packed as CPT/2 words
template <>
struct Raw<4> { uint2 v; };
template <>
struct Raw<8> { uint4 v; };

template <int CPT>
__device__ __forceinline__ Raw<CPT> ldg_raw(const bf16* p);
template <>
__device__ __forceinline__ Raw<4> ldg_raw<4>(const bf16* p) { Raw<4> r; r.v = __ldg(reinterpret_cast<const uint2*>(p)); return r; }
template <>
__device__ __forceinline__ Raw<8> ldg_raw<8>(const bf16* p) { Raw<8> r; r.v = __ldg(reinterpret_cast<const uint4*>(p)); return r; }

template <int CPT>
__device__ __forceinline__ void stg_raw(bf16* p, const uint32_t (&w)[CPT / 2]);
template <>
__device__ __forceinline__ void stg_raw<4>(bf16* p, const uint32_t (&w)[2]) { *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]); }
template <>
__device__ __forceinline__ void stg_raw<8>(bf16* p, const uint32_t (&w)[4]) { *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]); }

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

template <int CPT>
__device__ __forceinline__ void unpack(const Raw<CPT>& r, float (&o)[CPT]);
template <>
__device__ __forceinline__ void unpack<4>(const Raw<4>& r, float (&o)[4]) {
  o[0] = bf_lo(r.v.x); o[1] = bf_hi(r.v.x); o[2] = bf_lo(r.v.y); o[3] = bf_hi(r.v.y);
}
template <>
__device__ __forceinline__ void unpack<8>(const Raw<8>& r, float (&o)[8]) {
  o[0] = bf_lo(r.v.x); o[1] = bf_hi(r.v.x); o[2] = bf_lo(r.v.y); o[3] = bf_hi(r.v.y);
  o[4] = bf_lo(r.v.z); o[5] = bf_hi(r.v.z); o[6] = bf_lo(r.v.w); o[7] = bf_hi(r.v.w);
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&t);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

__device__ __forceinline__ float fast_rcp(float x) { return __fdividef(1.0f, x); }

// GELU_erf(x) = 0.5 x (1 + erf(x / sqrt 2)),  erf via Abramowitz-Stegun 7.1.26 (abs error <= 1.5e-7)
// also returns GELU'(x) = Phi(x) + x phi(x)  (phi shares the exponential)
__device__ __forceinline__ void gelu_erf(float x, float& g, float& dg) {
  const float z = x * 0.70710678118654752f;
  const float a = fabsf(z);
  const float t = fast_rcp(fmaf(0.3275911f, a, 1.0f));
  float p = fmaf(1.061405429f, t, -1.453152027f);
  p = fmaf(p, t, 1.421413741f);
  p = fmaf(p, t, -0.284496736f);
  p = fmaf(p, t, 0.254829592f);
  const float ex = __expf(-a * a);
  const float erf_abs = fmaf(-p * t, ex, 1.0f);
  const float e = copysignf(erf_abs, z);
  const float cdf = 0.5f * (1.0f + e);
  g = x * cdf;
  dg = fmaf(x * 0.3989422804014327f, ex, cdf);
}
__device__ __forceinline__ float gelu_only(float x) {
  float g, d;
  gelu_erf(x, g, d);
  return g;
}

__device__ __forceinline__ size_t zoff(int b, int P, int Q, int h, int w, int C) {
  return ((size_t)((b * h + (P >> 1)) * w + (Q >> 1)) * 4 + (((P & 1) << 1) | (Q & 1))) * (size_t)C;
}

// one-warp-level -> block-level sum of K values using smem slots [NW][K], double buffered by the caller's parity
template <int NW, int K>
__device__ __forceinline__ void block_sum(float (&v)[K], float (*red)[NW][K], int par) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < K; ++k) v[k] = warp_sum(v[k]);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) red[par][wid][k] = v[k];
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float s = red[par][0][k];
#pragma unroll
    for (int j = 1; j < NW; ++j) s += red[par][j][k];
    v[k] = s;
  }
}

// ------------------------------------------------------------------------------------------------ forward
template <int C, int CPT>
__global__ void __launch_bounds__(C / CPT)
    fwd_kernel(const bf16* __restrict__ Z, const float* __restrict__ BIAS, const float* __restrict__ DWT, const float* __restrict__ GAM,
               const float* __restrict__ BET, bf16* __restrict__ Y, float* __restrict__ MEAN, float* __restrict__ RSTD, float eps, int h, int w,
               int PWpix) {
  constexpr int T = C / CPT, NW = T / 32;
  __shared__ float red[2][NW][2];
  const int tid = threadIdx.x, c0 = tid * CPT;
  const int H = 2 * h, W = 2 * w;
  const int r = blockIdx.y, b = r / H, P = r % H;
  const int q0 = blockIdx.x * PWpix;

  float bs[CPT], gm[CPT], bt[CPT], wt[9][CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    bs[i] = BIAS[c0 + i]; gm[i] = GAM[c0 + i]; bt[i] = BET[c0 + i];
#pragma unroll
    for (int k = 0; k < 9; ++k) wt[k][i] = DWT[k * C + c0 + i];
  }
  float accL[CPT], accM[CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) { accL[i] = 0.f; accM[i] = 0.f; }

  bool rowok[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) rowok[k] = (P + k - 1 >= 0) && (P + k - 1 < H);

  auto load_col = [&](int x, Raw<CPT> (&rw)[3]) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (rowok[k] && x >= 0 && x < W) rw[k] = ldg_raw<CPT>(Z + zoff(b, P + k - 1, x, h, w, C) + c0);
      else rw[k].v = decltype(rw[k].v){};
    }
  };

  Raw<CPT> cur[3], nxt[3];
  load_col(q0 - 1, cur);
  float pivot = 0.f;
  int par = 0;
  for (int x = q0 - 1; x <= q0 + PWpix; ++x) {
    load_col(x + 1, nxt);  // prefetch (independent of the math below)
    // column x of g = GELU(Z + bias), zero outside the image
    float col[3][CPT];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      float zv[CPT];
      unpack<CPT>(cur[k], zv);
      const bool ok = rowok[k] && x >= 0 && x < W;
#pragma unroll
      for (int i = 0; i < CPT; ++i) col[k][i] = ok ? gelu_only(zv[i] + bs[i]) : 0.f;
    }
    float u[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      u[i] = accL[i] + wt[2][i] * col[0][i] + wt[5][i] * col[1][i] + wt[8][i] * col[2][i];
      accL[i] = accM[i] + wt[1][i] * col[0][i] + wt[4][i] * col[1][i] + wt[7][i] * col[2][i];
      accM[i] = wt[0][i] * col[0][i] + wt[3][i] * col[1][i] + wt[6][i] * col[2][i];
    }
    if (x >= q0 + 1) {  // out[x-1] is complete: LayerNorm over the C channels of pixel (b,P,x-1)
      float s[2] = {0.f, 0.f};
      float ud[CPT];
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        ud[i] = u[i] - pivot;
        s[0] += ud[i];
        s[1] = fmaf(ud[i], ud[i], s[1]);
      }
      block_sum<NW, 2>(s, red, par);
      par ^= 1;
      const float m1 = s[0] * (1.0f / C);
      const float var = fmaxf(s[1] * (1.0f / C) - m1 * m1, 0.f);
      const float rstd = rsqrtf(var + eps);
      const float mean = pivot + m1;
      uint32_t o[CPT / 2];
#pragma unroll
      for (int i = 0; i < CPT; i += 2) {
        const float y0 = fmaf((ud[i] - m1) * rstd, gm[i], bt[i]);
        const float y1 = fmaf((ud[i + 1] - m1) * rstd, gm[i + 1], bt[i + 1]);
        o[i / 2] = pack2(y0, y1);
      }
      const size_t pix = (size_t)(b * H + P) * W + (x - 1);
      stg_raw<CPT>(Y + pix * C + c0, o);
      if (tid == 0) { MEAN[pix] = mean; RSTD[pix] = rstd; }
      pivot = mean;
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) cur[k] = nxt[k];
  }
}

// ------------------------------------------------------------------------------------------------ backward 1: LayerNorm backward -> dU
template <int C, int CPT, int MINB>
__global__ void __launch_bounds__(C / CPT, MINB)
    bwd1_kernel(const bf16* __restrict__ Z, const float* __restrict__ BIAS, const float* __restrict__ DWT, const float* __restrict__ GAM,
                const float* __restrict__ MEAN, const float* __restrict__ RSTD, const bf16* __restrict__ DY, bf16* __restrict__ DU,
                float* __restrict__ PART, int h, int w, int PWpix, int RPB) {
  constexpr int T = C / CPT, NW = T / 32;
  __shared__ float red[2][NW][2];
  const int tid = threadIdx.x, c0 = tid * CPT;
  const int H = 2 * h, W = 2 * w;
  const int q0 = blockIdx.x * PWpix;

  float bs[CPT], gm[CPT], wt[9][CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    bs[i] = BIAS[c0 + i]; gm[i] = GAM[c0 + i];
#pragma unroll
    for (int k = 0; k < 9; ++k) wt[k][i] = DWT[k * C + c0 + i];
  }
  float accL[CPT], accM[CPT], ag[CPT], ab[CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) { ag[i] = 0.f; ab[i] = 0.f; }
  for (int rr = 0; rr < RPB; ++rr) {
  const int r = blockIdx.y * RPB + rr, b = r / H, P = r % H;
#pragma unroll
  for (int i = 0; i < CPT; ++i) { accL[i] = 0.f; accM[i] = 0.f; }
  bool rowok[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) rowok[k] = (P + k - 1 >= 0) && (P + k - 1 < H);

  auto load_col = [&](int x, Raw<CPT> (&rw)[3]) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (rowok[k] && x >= 0 && x < W) rw[k] = ldg_raw<CPT>(Z + zoff(b, P + k - 1, x, h, w, C) + c0);
      else rw[k].v = decltype(rw[k].v){};
    }
  };
  const size_t rowpix = (size_t)(b * H + P) * W;

  Raw<CPT> cur[3], nxt[3];
  load_col(q0 - 1, cur);
  int par = 0;
  for (int x = q0 - 1; x <= q0 + PWpix; ++x) {
    load_col(x + 1, nxt);
    Raw<CPT> dy_raw;
    dy_raw.v = decltype(dy_raw.v){};
    if (x >= q0 + 1) dy_raw = ldg_raw<CPT>(DY + (rowpix + (x - 1)) * C + c0);
    float col[3][CPT];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      float zv[CPT];
      unpack<CPT>(cur[k], zv);
      const bool ok = rowok[k] && x >= 0 && x < W;
#pragma unroll
      for (int i = 0; i < CPT; ++i) col[k][i] = ok ? gelu_only(zv[i] + bs[i]) : 0.f;
    }
    float u[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      u[i] = accL[i] + wt[2][i] * col[0][i] + wt[5][i] * col[1][i] + wt[8][i] * col[2][i];
      accL[i] = accM[i] + wt[1][i] * col[0][i] + wt[4][i] * col[1][i] + wt[7][i] * col[2][i];
      accM[i] = wt[0][i] * col[0][i] + wt[3][i] * col[1][i] + wt[6][i] * col[2][i];
    }
    if (x >= q0 + 1) {
      const size_t pix = rowpix + (x - 1);
      const float mean = MEAN[pix], rstd = RSTD[pix];
      float dy[CPT], xh[CPT], a[CPT];
      unpack<CPT>(dy_raw, dy);
      float s[2] = {0.f, 0.f};
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        xh[i] = (u[i] - mean) * rstd;
        a[i] = dy[i] * gm[i];
        s[0] += a[i];
        s[1] = fmaf(a[i], xh[i], s[1]);
        ag[i] = fmaf(dy[i], xh[i], ag[i]);
        ab[i] += dy[i];
      }
      block_sum<NW, 2>(s, red, par);
      par ^= 1;
      const float s1 = s[0] * (1.0f / C), s2 = s[1] * (1.0f / C);
      uint32_t o[CPT / 2];
#pragma unroll
      for (int i = 0; i < CPT; i += 2) {
        const float d0 = rstd * (a[i] - s1 - xh[i] * s2);
        const float d1 = rstd * (a[i + 1] - s1 - xh[i + 1] * s2);
        o[i / 2] = pack2(d0, d1);
      }
      stg_raw<CPT>(DU + pix * C + c0, o);
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) cur[k] = nxt[k];
  }
  }  // rows
  const int pid = blockIdx.y * gridDim.x + blockIdx.x;
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    PART[((size_t)pid * 2 + 0) * C + c0 + i] = ag[i];
    PART[((size_t)pid * 2 + 1) * C + c0 + i] = ab[i];
  }
}

// ------------------------------------------------------------------------------------------------ backward 2: dwconv backward + GELU' -> dZ, dW(9 taps), dbias
// dh[Q'] = sum_{ky,kx} w[ky][kx] * dU[P'+1-ky][Q'+1-kx];   dW[ky][kx] += g[P',Q'] * dU[P'+1-ky][Q'+1-kx];   dZ = dh * GELU'(Z+bias)
// dU column x (rows r=0,1,2 = P-1,P,P+1; row r pairs with ky = 2-r) completes dh[x-1] (tap kx=0), adds to dh[x] (kx=1) and dh[x+1] (kx=2),
// and feeds dW[ky][kx] together with the g of pixel x-1+kx.
template <int C, int CPT, int MINB>
__global__ void __launch_bounds__(C / CPT, MINB)
    bwd2_kernel(const bf16* __restrict__ Z, const float* __restrict__ BIAS, const float* __restrict__ DWT, const bf16* __restrict__ DU,
                bf16* __restrict__ DZ, float* __restrict__ PDW, float* __restrict__ PDB, int h, int w, int PWpix, int RPB) {
  constexpr int T = C / CPT;
  const int tid = threadIdx.x, c0 = tid * CPT;
  const int H = 2 * h, W = 2 * w;
  const int q0 = blockIdx.x * PWpix, q1 = q0 + PWpix;

  float bs[CPT], wt[9][CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    bs[i] = BIAS[c0 + i];
#pragma unroll
    for (int k = 0; k < 9; ++k) wt[k][i] = DWT[k * C + c0 + i];
  }
  // dh partials: dh[x-1] = accL + w[2-r][0]*du_r[x] (complete), accL' = accM + w[2-r][1]*du_r[x] (partial dh[x]), accM' = w[2-r][2]*du_r[x] (partial dh[x+1])
  float accL[CPT], accM[CPT], dwacc[9][CPT], dbacc[CPT];
  float gm1[CPT], g0[CPT], dgm1[CPT], dg0[CPT];  // g (owned-masked) and GELU' of pixels x-1, x
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    dbacc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) dwacc[k][i] = 0.f;
  }
  for (int rr = 0; rr < RPB; ++rr) {
  const int r = blockIdx.y * RPB + rr, b = r / H, P = r % H;
#pragma unroll
  for (int i = 0; i < CPT; ++i) { accL[i] = 0.f; accM[i] = 0.f; gm1[i] = 0.f; g0[i] = 0.f; dgm1[i] = 0.f; dg0[i] = 0.f; }
  bool rowok[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) rowok[k] = (P + k - 1 >= 0) && (P + k - 1 < H);

  auto load_du = [&](int x, Raw<CPT> (&rw)[3]) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (rowok[k] && x >= 0 && x < W) rw[k] = ldg_raw<CPT>(DU + ((size_t)(b * H + P + k - 1) * W + x) * C + c0);
      else rw[k].v = decltype(rw[k].v){};
    }
  };
  auto load_z = [&](int q, Raw<CPT>& rz) {
    if (q >= q0 && q < q1 && q < W) rz = ldg_raw<CPT>(Z + zoff(b, P, q, h, w, C) + c0);
    else rz.v = decltype(rz.v){};
  };

  Raw<CPT> cur[3], nxt[3], zcur, znxt;
  load_du(q0 - 1, cur);
  load_z(q0, zcur);  // z of pixel x+1 at the first step x = q0-1
  for (int x = q0 - 1; x <= q1; ++x) {
    load_du(x + 1, nxt);
    load_z(x + 2, znxt);
    // g / GELU' of pixel x+1 (owned pixels only)
    float gn[CPT], dgn[CPT];
    {
      float zv[CPT];
      unpack<CPT>(zcur, zv);
      const bool own = (x + 1 >= q0) && (x + 1 < q1) && (x + 1 < W);
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        float g, d;
        gelu_erf(zv[i] + bs[i], g, d);
        gn[i] = own ? g : 0.f;
        dgn[i] = own ? d : 0.f;
      }
    }
    float du[3][CPT];
#pragma unroll
    for (int k = 0; k < 3; ++k) unpack<CPT>(cur[k], du[k]);  // out-of-image entries were loaded as zero
    // window row k <-> ky = 2-k
    float dh[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      dh[i] = accL[i] + wt[2 * 3 + 0][i] * du[0][i] + wt[1 * 3 + 0][i] * du[1][i] + wt[0 * 3 + 0][i] * du[2][i];  // kx = 0 completes dh[x-1]
      accL[i] = accM[i] + wt[2 * 3 + 1][i] * du[0][i] + wt[1 * 3 + 1][i] * du[1][i] + wt[0 * 3 + 1][i] * du[2][i];
      accM[i] = wt[2 * 3 + 2][i] * du[0][i] + wt[1 * 3 + 2][i] * du[1][i] + wt[0 * 3 + 2][i] * du[2][i];
      // dW[ky][kx] += g[x-1+kx] * du[row 2-ky][x]
      const float gk[3] = {gm1[i], g0[i], gn[i]};
#pragma unroll
      for (int ky = 0; ky < 3; ++ky)
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) dwacc[ky * 3 + kx][i] = fmaf(gk[kx], du[2 - ky][i], dwacc[ky * 3 + kx][i]);
    }
    if (x >= q0 + 1) {  // dh[x-1] complete; pixel x-1 is owned
      uint32_t o[CPT / 2];
      float dz[CPT];
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        dz[i] = dh[i] * dgm1[i];
        dbacc[i] += dz[i];
      }
#pragma unroll
      for (int i = 0; i < CPT; i += 2) o[i / 2] = pack2(dz[i], dz[i + 1]);
      stg_raw<CPT>(DZ + zoff(b, P, x - 1, h, w, C) + c0, o);
    }
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      gm1[i] = g0[i]; g0[i] = gn[i];
      dgm1[i] = dg0[i]; dg0[i] = dgn[i];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) cur[k] = nxt[k];
    zcur = znxt;
  }
  }  // rows
  const int pid = blockIdx.y * gridDim.x + blockIdx.x;
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
#pragma unroll
    for (int k = 0; k < 9; ++k) PDW[((size_t)pid * 9 + k) * C + c0 + i] = dwacc[k][i];
    PDB[(size_t)pid * C + c0 + i] = dbacc[i];
  }
}

// ------------------------------------------------------------------------------------------------ host side
template <int C, int CPT>
static void launch_fwd(const at::Tensor& Z, const at::Tensor& bias, const at::Tensor& dwt, const at::Tensor& gam, const at::Tensor& bet,
                       at::Tensor& Y, at::Tensor& mean, at::Tensor& rstd, double eps, int B, int h, int w, int PW) {
  const int H = 2 * h, W = 2 * w;
  dim3 grid(W / PW, B * H), block(C / CPT);
  fwd_kernel<C, CPT><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(Z.data_ptr<at::BFloat16>()), bias.data_ptr<float>(), dwt.data_ptr<float>(), gam.data_ptr<float>(),
      bet.data_ptr<float>(), reinterpret_cast<bf16*>(Y.data_ptr<at::BFloat16>()), mean.data_ptr<float>(), rstd.data_ptr<float>(), (float)eps, h, w, PW);
}
template <int C, int CPT, int MINB>
static void launch_bwd1(const at::Tensor& Z, const at::Tensor& bias, const at::Tensor& dwt, const at::Tensor& gam, const at::Tensor& mean,
                        const at::Tensor& rstd, const at::Tensor& dy, at::Tensor& du, at::Tensor& part, int B, int h, int w, int PW, int RPB) {
  const int H = 2 * h, W = 2 * w;
  dim3 grid(W / PW, B * H / RPB), block(C / CPT);
  bwd1_kernel<C, CPT, MINB><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(Z.data_ptr<at::BFloat16>()), bias.data_ptr<float>(), dwt.data_ptr<float>(), gam.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), reinterpret_cast<const bf16*>(dy.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(du.data_ptr<at::BFloat16>()), part.data_ptr<float>(), h, w, PW, RPB);
}
template <int C, int CPT, int MINB>
static void launch_bwd2(const at::Tensor& Z, const at::Tensor& bias, const at::Tensor& dwt, const at::Tensor& du, at::Tensor& dz, at::Tensor& pdw,
                        at::Tensor& pdb, int B, int h, int w, int PW, int RPB) {
  const int H = 2 * h, W = 2 * w;
  dim3 grid(W / PW, B * H / RPB), block(C / CPT);
  bwd2_kernel<C, CPT, MINB><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(Z.data_ptr<at::BFloat16>()), bias.data_ptr<float>(), dwt.data_ptr<float>(),
      reinterpret_cast<const bf16*>(du.data_ptr<at::BFloat16>()), reinterpret_cast<bf16*>(dz.data_ptr<at::BFloat16>()), pdw.data_ptr<float>(),
      pdb.data_ptr<float>(), h, w, PW, RPB);
}

#define DISPATCH(C_, CPT_, FN, ...)                         \
  if ((C_) == 1024 && (CPT_) == 4) FN<1024, 4>(__VA_ARGS__); \
  else if ((C_) == 1024 && (CPT_) == 8) FN<1024, 8>(__VA_ARGS__); \
  else if ((C_) == 768 && (CPT_) == 4) FN<768, 4>(__VA_ARGS__);   \
  else if ((C_) == 768 && (CPT_) == 8) FN<768, 8>(__VA_ARGS__);   \
  else TORCH_CHECK(false, "unsupported C/CPT ", (C_), "/", (CPT_));

#define DISPATCH_B(C_, CPT_, MINB_, FN, ...)                                   \
  if ((C_) == 1024 && (CPT_) == 4 && (MINB_) == 1) FN<1024, 4, 1>(__VA_ARGS__);   \
  else if ((C_) == 1024 && (CPT_) == 4 && (MINB_) == 2) FN<1024, 4, 2>(__VA_ARGS__); \
  else if ((C_) == 1024 && (CPT_) == 8 && (MINB_) == 1) FN<1024, 8, 1>(__VA_ARGS__); \
  else if ((C_) == 1024 && (CPT_) == 8 && (MINB_) == 2) FN<1024, 8, 2>(__VA_ARGS__); \
  else if ((C_) == 768 && (CPT_) == 4 && (MINB_) == 1) FN<768, 4, 1>(__VA_ARGS__);   \
  else if ((C_) == 768 && (CPT_) == 4 && (MINB_) == 2) FN<768, 4, 2>(__VA_ARGS__);   \
  else if ((C_) == 768 && (CPT_) == 8 && (MINB_) == 1) FN<768, 8, 1>(__VA_ARGS__);   \
  else if ((C_) == 768 && (CPT_) == 8 && (MINB_) == 2) FN<768, 8, 2>(__VA_ARGS__);   \
  else TORCH_CHECK(false, "unsupported C/CPT/MINB ", (C_), "/", (CPT_), "/", (MINB_));

#define CHECK_IN(x, dt) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == dt, #x " must be a contiguous CUDA tensor of dtype " #dt)

std::vector<at::Tensor> stage_fwd(at::Tensor Z, at::Tensor bias, at::Tensor dwt, at::Tensor gam, at::Tensor bet, int64_t B, int64_t h, int64_t w,
                                  double eps, int64_t PW, int64_t CPT) {
  c10::cuda::CUDAGuard guard(Z.device());
  CHECK_IN(Z, at::kBFloat16); CHECK_IN(bias, at::kFloat); CHECK_IN(dwt, at::kFloat); CHECK_IN(gam, at::kFloat); CHECK_IN(bet, at::kFloat);
  const int C = bias.numel(), H = 2 * h, W = 2 * w;
  TORCH_CHECK(W % PW == 0);
  auto Y = at::empty({B * H * W, C}, Z.options());
  auto mean = at::empty({B * H * W}, bias.options());
  auto rstd = at::empty({B * H * W}, bias.options());
  DISPATCH(C, CPT, launch_fwd, Z, bias, dwt, gam, bet, Y, mean, rstd, eps, (int)B, (int)h, (int)w, (int)PW);
  return {Y, mean, rstd};
}

std::vector<at::Tensor> stage_bwd1(at::Tensor Z, at::Tensor bias, at::Tensor dwt, at::Tensor gam, at::Tensor mean, at::Tensor rstd, at::Tensor dy,
                                   int64_t B, int64_t h, int64_t w, int64_t PW, int64_t CPT, int64_t RPB, int64_t MINB) {
  c10::cuda::CUDAGuard guard(Z.device());
  CHECK_IN(Z, at::kBFloat16); CHECK_IN(dy, at::kBFloat16); CHECK_IN(mean, at::kFloat); CHECK_IN(rstd, at::kFloat);
  const int C = bias.numel(), H = 2 * h, W = 2 * w;
  TORCH_CHECK(W % PW == 0);
  TORCH_CHECK((B * H) % RPB == 0);
  auto du = at::empty({B * H * W, C}, Z.options());
  const int64_t npr = (W / PW) * (B * H / RPB);
  auto part = at::empty({npr, 2, C}, bias.options());
  DISPATCH_B(C, CPT, MINB, launch_bwd1, Z, bias, dwt, gam, mean, rstd, dy, du, part, (int)B, (int)h, (int)w, (int)PW, (int)RPB);
  return {du, part};
}

std::vector<at::Tensor> stage_bwd2(at::Tensor Z, at::Tensor bias, at::Tensor dwt, at::Tensor du, int64_t B, int64_t h, int64_t w, int64_t PW,
                                   int64_t CPT, int64_t RPB, int64_t MINB) {
  c10::cuda::CUDAGuard guard(Z.device());
  CHECK_IN(Z, at::kBFloat16); CHECK_IN(du, at::kBFloat16);
  const int C = bias.numel(), H = 2 * h, W = 2 * w;
  TORCH_CHECK(W % PW == 0);
  TORCH_CHECK((B * H) % RPB == 0);
  auto dz = at::empty_like(Z);
  const int64_t npr = (W / PW) * (B * H / RPB);
  auto pdw = at::empty({npr, 9, C}, bias.options());
  auto pdb = at::empty({npr, C}, bias.options());
  DISPATCH_B(C, CPT, MINB, launch_bwd2, Z, bias, dwt, du, dz, pdw, pdb, (int)B, (int)h, (int)w, (int)PW, (int)RPB);
  return {dz, pdw, pdb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage_fwd", &stage_fwd);
  m.def("stage_bwd1", &stage_bwd1);
  m.def("stage_bwd2", &stage_bwd2);
}
