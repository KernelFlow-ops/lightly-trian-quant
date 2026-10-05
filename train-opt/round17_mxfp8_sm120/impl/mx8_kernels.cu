// round17 MXFP8 (E4M3 + UE8M0, 1x32) quantizers for the sm_120 training block (CUDA C++).
//
//   quant_dual : bf16 / fp32 x[M, C] -> row copy  (scale blocks along C, K-major data [M, C])
//                                   and column copy (scale blocks along M, data stays row-major [Mpad, C], rows >= M are zero)
//                in ONE read of x.  Both scale tensors are written straight into the cuBLAS / CUTLASS / QuACK blocked
//                (rm, rk, 32, 4, 4) layout (one 512 B atom = 128 MN x 4 K-blocks, byte (r%32)*16 + (r/32 %4)*4 + k%4).
//
// Scale rule = NVIDIA MXFP8 recipe, RCEIL: e = ceil(log2(amax / 448)) (smallest power of two with amax / 2^e <= 448),
// identical to quack.blockscaled.quantize.to_mx(scaling_mode="rceil") (bit-exact, tested in tests/test_quant.py);
// elements are x * 2^-e converted RN-even, saturating, to e4m3.  amax is an integer max of |x| bit patterns, so NaN / Inf
// propagate (scale byte 255, data NaN).  No atomics, no data-dependent control flow beyond row bounds: deterministic.
//
// Geometry: one CTA = 32 rows x 128 columns (= one SF atom column group), 128 threads, thread = 4 rows x 8 columns held in
// registers.  Row scales are 4-thread shuffles, column amax goes through one 4 KB smem exchange.  Every output byte of the
// SF atoms covered by [Mpad, C] is written (rows >= M: amax 0 -> scale byte 0, data 0).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

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

// biased e8m0 exponent of the RCEIL scale for a block amax given as |x| bits (NaN > Inf > finite as integers)
__device__ __forceinline__ uint32_t rceil_scale_byte(uint32_t amax_bits) {
  float ratio = __fdiv_rn(__uint_as_float(amax_bits), 448.0f);  // inf -> inf, nan -> nan
  uint32_t b = __float_as_uint(ratio);
  uint32_t e = (b >> 23) & 0xffu;
  uint32_t frac = (b & 0x7fffffu) != 0u;
  uint32_t s = e + frac;
  return s > 255u ? 255u : s;
}
__device__ __forceinline__ float inv_scale_of(uint32_t s) {  // 2^-(s-127); s in [0, 253]
  return __uint_as_float((254u - (s == 0u ? 1u : s)) << 23);
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {  // byte0 = lo, byte1 = hi
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return (uint32_t)r;
}

__device__ __forceinline__ uint2 quant8(const float (&v)[8], uint32_t s) {
  uint2 o;
  if (s >= 255u) {
    o.x = 0x7f7f7f7fu;
    o.y = 0x7f7f7f7fu;
    return o;
  }
  const float inv = inv_scale_of(s);
  o.x = cvt_e4m3x2(v[0] * inv, v[1] * inv) | (cvt_e4m3x2(v[2] * inv, v[3] * inv) << 16);
  o.y = cvt_e4m3x2(v[4] * inv, v[5] * inv) | (cvt_e4m3x2(v[6] * inv, v[7] * inv) << 16);
  return o;
}

template <typename T, bool ROW, bool COL>
__global__ void __launch_bounds__(128)
quant_dual_kernel(const T* __restrict__ X, int M, int C, int Mpad, uint8_t* __restrict__ QR, uint32_t* __restrict__ SFR,
                  uint8_t* __restrict__ QC, uint8_t* __restrict__ SFC) {
  __shared__ uint32_t sA[8][128];
  __shared__ uint8_t sS[128];
  const int tid = threadIdx.x;
  const int cg = tid & 15, rg = tid >> 4;
  const int ct = blockIdx.x, rt = blockIdx.y;
  const int col0 = ct * 128 + cg * 8;
  float v[4][8];
  uint32_t cmax[8];
#pragma unroll
  for (int j = 0; j < 8; j++) cmax[j] = 0u;
#pragma unroll
  for (int p = 0; p < 4; p++) {
    const int row = rt * 32 + p * 8 + rg;
    if (row < M) {
      load8(X + (size_t)row * C + col0, v[p]);
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) v[p][j] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) cmax[j] = max(cmax[j], __float_as_uint(fabsf(v[p][j])));
  }
  if (ROW) {
    const int rk_total = C >> 7;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const int row = rt * 32 + p * 8 + rg;
      uint32_t a = 0u;
#pragma unroll
      for (int j = 0; j < 8; j++) a = max(a, __float_as_uint(fabsf(v[p][j])));
      a = max(a, __shfl_xor_sync(FULL_MASK, a, 1));
      a = max(a, __shfl_xor_sync(FULL_MASK, a, 2));
      const uint32_t s = rceil_scale_byte(a);
      if (row < M) *reinterpret_cast<uint2*>(QR + (size_t)row * C + col0) = quant8(v[p], s);
      // the four blocks of this 128-column group are the 4 bytes of one atom row: gather them in lane (tid & 15) == 0
      const uint32_t s1 = __shfl_down_sync(FULL_MASK, s, 4);
      const uint32_t s2 = __shfl_down_sync(FULL_MASK, s, 8);
      const uint32_t s3 = __shfl_down_sync(FULL_MASK, s, 12);
      if (cg == 0) {
        const uint32_t word = (s & 0xffu) | ((s1 & 0xffu) << 8) | ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
        SFR[(size_t)((row >> 7) * rk_total + ct) * 128 + (row & 31) * 4 + ((row >> 5) & 3)] = word;
      }
    }
  }
  if (COL) {
#pragma unroll
    for (int j = 0; j < 8; j++) sA[rg][cg * 8 + j] = cmax[j];
    __syncthreads();
    {
      uint32_t a = sA[0][tid];
#pragma unroll
      for (int r = 1; r < 8; r++) a = max(a, sA[r][tid]);
      const uint32_t s = rceil_scale_byte(a);
      sS[tid] = (uint8_t)s;
      const int rkC = Mpad >> 7;
      const int c = ct * 128 + tid;
      SFC[(size_t)(ct * rkC + (rt >> 2)) * 512 + (tid & 31) * 16 + (tid >> 5) * 4 + (rt & 3)] = (uint8_t)s;
    }
    __syncthreads();
    uint32_t sc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) sc[j] = sS[cg * 8 + j];
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const int row = rt * 32 + p * 8 + rg;
      // quantize with per-column scales
      uint2 o;
      float inv[8];
      bool nanb = false;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        inv[j] = inv_scale_of(sc[j] >= 255u ? 1u : sc[j]);
        nanb |= (sc[j] >= 255u);
      }
      uint32_t w0 = cvt_e4m3x2(v[p][0] * inv[0], v[p][1] * inv[1]) | (cvt_e4m3x2(v[p][2] * inv[2], v[p][3] * inv[3]) << 16);
      uint32_t w1 = cvt_e4m3x2(v[p][4] * inv[4], v[p][5] * inv[5]) | (cvt_e4m3x2(v[p][6] * inv[6], v[p][7] * inv[7]) << 16);
      if (nanb) {  // rare path: per column NaN-scale handling (data NaN where the column block scale is the NaN sentinel)
        uint8_t b[8];
        const uint8_t* q0 = reinterpret_cast<const uint8_t*>(&w0);
        const uint8_t* q1 = reinterpret_cast<const uint8_t*>(&w1);
#pragma unroll
        for (int j = 0; j < 4; j++) { b[j] = q0[j]; b[4 + j] = q1[j]; }
#pragma unroll
        for (int j = 0; j < 8; j++) if (sc[j] >= 255u) b[j] = 0x7f;
        w0 = *reinterpret_cast<uint32_t*>(&b[0]);
        w1 = *reinterpret_cast<uint32_t*>(&b[4]);
      }
      o.x = w0;
      o.y = w1;
      *reinterpret_cast<uint2*>(QC + (size_t)row * C + col0) = o;
    }
  }
}

template <typename T>
static void launch_quant(const at::Tensor& x, int Mpad, uint8_t* qr, uint8_t* sfr, uint8_t* qc, uint8_t* sfc) {
  const int M = (int)x.size(0), C = (int)x.size(1);
  dim3 grid(C / 128, Mpad / 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  const T* xp = reinterpret_cast<const T*>(x.data_ptr());
  if (qr && qc) quant_dual_kernel<T, true, true><<<grid, 128, 0, stream>>>(xp, M, C, Mpad, qr, (uint32_t*)sfr, qc, sfc);
  else if (qr) quant_dual_kernel<T, true, false><<<grid, 128, 0, stream>>>(xp, M, C, Mpad, qr, (uint32_t*)sfr, qc, sfc);
  else quant_dual_kernel<T, false, true><<<grid, 128, 0, stream>>>(xp, M, C, Mpad, qr, (uint32_t*)sfr, qc, sfc);
}

// x [M, C] bf16 or fp32 contiguous (C % 128 == 0), Mpad = multiple of 128 >= M.
// Returns (qr [M, C] e4m3, sfr [Mpad/128, C/128, 32, 4, 4] u8, qc [Mpad, C] e4m3, sfc [C/128, Mpad/128, 32, 4, 4] u8); unused -> empty.
std::vector<at::Tensor> quant_dual(const at::Tensor& x, int64_t Mpad, bool row, bool col) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous CUDA matrix");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "bf16 / fp32 only");
  const int64_t M = x.size(0), C = x.size(1);
  TORCH_CHECK(C % 128 == 0, "C must be a multiple of 128");
  TORCH_CHECK(Mpad % 128 == 0 && Mpad >= M, "Mpad must be a multiple of 128 and >= M");
  TORCH_CHECK(row || col, "nothing to do");
  c10::cuda::CUDAGuard guard(x.device());
  auto o8 = x.options().dtype(at::kFloat8_e4m3fn);
  auto u8 = x.options().dtype(at::kByte);
  // one allocation per direction (data + scales): fewer allocator calls on the (host-bound) eager path
  at::Tensor qr, sfr, qc, sfc;
  if (row) {
    const int64_t nq = M * C, ns = (Mpad / 128) * (C / 128) * 512;
    at::Tensor buf = at::empty({nq + ns}, u8);
    qr = buf.narrow(0, 0, nq).view(at::kFloat8_e4m3fn).view({M, C});
    sfr = buf.narrow(0, nq, ns).view({Mpad / 128, C / 128, 32, 4, 4});
  } else {
    qr = at::empty({0}, o8);
    sfr = at::empty({0}, u8);
  }
  if (col) {
    const int64_t nq = Mpad * C, ns = (C / 128) * (Mpad / 128) * 512;
    at::Tensor buf = at::empty({nq + ns}, u8);
    qc = buf.narrow(0, 0, nq).view(at::kFloat8_e4m3fn).view({Mpad, C});
    sfc = buf.narrow(0, nq, ns).view({C / 128, Mpad / 128, 32, 4, 4});
  } else {
    qc = at::empty({0}, o8);
    sfc = at::empty({0}, u8);
  }
  uint8_t* pqr = row ? (uint8_t*)qr.data_ptr() : nullptr;
  uint8_t* psr = row ? (uint8_t*)sfr.data_ptr() : nullptr;
  uint8_t* pqc = col ? (uint8_t*)qc.data_ptr() : nullptr;
  uint8_t* psc = col ? (uint8_t*)sfc.data_ptr() : nullptr;
  if (x.scalar_type() == at::kBFloat16) launch_quant<__nv_bfloat16>(x, (int)Mpad, pqr, psr, pqc, psc);
  else launch_quant<float>(x, (int)Mpad, pqr, psr, pqc, psc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qr, sfr, qc, sfc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_dual", &quant_dual);
}
