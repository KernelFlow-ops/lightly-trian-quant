// round13 fused transformer-block elementwise kernels (sm_120, CUDA C++).
//
//   ln_fwd      : fp32 x -> bf16 LayerNorm output (+ mean / rstd), warp per row, row in registers.
//   rowbwd      : LayerNorm backward + residual-gradient add + g = gamma * dX (bf16) + deterministic
//                 per-CTA column partials (dW, dB, dGamma, dBias-of-branch).  One template covers
//                 LN1-bwd, LN2-bwd + proj prep and the fc2 prep (g -> gamma * g) kernels.
//   colsum_bf16 : per-CTA column partial sums of a bf16 [M, C] matrix (qkv bias gradient).
//   finalize    : fixed-order sum of the per-CTA partials -> fp32 vector(s) (deterministic).
//
// All reductions are fp32.  No atomics on data, so results are bitwise reproducible.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

__device__ __forceinline__ void unpack_bf16x4(const __nv_bfloat16* p, float (&o)[4]) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&u.x);
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&u.y);
  float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
  o[0] = fa.x; o[1] = fa.y; o[2] = fb.x; o[3] = fb.y;
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, const float (&v)[4]) {
  __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
  __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

// ------------------------------------------------------------------------------------------------
// LayerNorm forward: x fp32 [M, D] -> y bf16 [M, D], mean[M], rstd[M] (fp32).  Warp per row.
// Statistics are the two-pass centered form in fp32 (robust for the ~1e5 massive activations).
template <int D>
__global__ void __launch_bounds__(256) ln_fwd_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                                      const float* __restrict__ B, __nv_bfloat16* __restrict__ Y,
                                                      float* __restrict__ MEAN, float* __restrict__ RSTD, int M,
                                                      float eps) {
  constexpr int NJ = D / 128;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= M) return;
  const float* xr = X + (size_t)row * D;
  float4 v[NJ];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    v[j] = ld4(xr + j * 128 + lane * 4);
    s += (v[j].x + v[j].y) + (v[j].z + v[j].w);
  }
  s = warp_sum(s);
  const float mean = s * (1.0f / D);
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    float a = v[j].x - mean, b = v[j].y - mean, c = v[j].z - mean, d = v[j].w - mean;
    q += (a * a + b * b) + (c * c + d * d);
  }
  q = warp_sum(q);
  const float rstd = rsqrtf(q * (1.0f / D) + eps);
  __nv_bfloat16* yr = Y + (size_t)row * D;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int c = j * 128 + lane * 4;
    float4 w = ld4(W + c), b = ld4(B + c);
    float o[4] = {(v[j].x - mean) * rstd * w.x + b.x, (v[j].y - mean) * rstd * w.y + b.y,
                  (v[j].z - mean) * rstd * w.z + b.z, (v[j].w - mean) * rstd * w.w + b.w};
    store_bf16x4(yr + c, o);
  }
  if (lane == 0) {
    MEAN[row] = mean;
    RSTD[row] = rstd;
  }
}

// ------------------------------------------------------------------------------------------------
// Fused row backward.
//   HAS_LN   : dl = LN_bwd(DY, xhat(X, mean, rstd), W);  dx = DR + dl      else dx = DR
//   HAS_GAM  : G = bf16(gamma * dx)  and column partials  dgamma += dx * Y,  dbias_branch += gamma * dx
//   WRITE_DX : DX = dx (fp32)
// Column partial layout PART[blockIdx.x][k][D]; k order: [dW, dB] (HAS_LN) then [dGamma, dBiasBranch] (HAS_GAM).
//
// Mapping: WPR = D / (128 * NJ) warps cooperate on one row (each warp owns 128*NJ contiguous columns, NJ float4 per
// lane), RG row groups run concurrently in the block.  The two row sums (sum a, sum a*xhat) are exchanged through
// shared memory with a per-row-group named barrier.  Each thread keeps only K*NJ*4 column accumulators in
// registers across rows, so there are no spills and enough warps per SM to cover the memory latency.
__device__ __forceinline__ void named_bar(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }

template <int D, int NJ, int RG, bool HAS_LN, bool HAS_GAM, bool WRITE_DX>
__global__ void __launch_bounds__(RG * (D / (128 * NJ)) * 32)
rowbwd_kernel(const float* __restrict__ X, const float* __restrict__ MEAN, const float* __restrict__ RSTD,
              const float* __restrict__ W, const __nv_bfloat16* __restrict__ DY, const float* __restrict__ DR,
              const __nv_bfloat16* __restrict__ Y, const float* __restrict__ GAMMA, float* __restrict__ DX,
              __nv_bfloat16* __restrict__ G, float* __restrict__ PART, int M) {
  constexpr int WPR = D / (128 * NJ);
  static_assert(WPR * 128 * NJ == D, "D must be a multiple of 128*NJ");
  constexpr int K = (HAS_LN ? 2 : 0) + (HAS_GAM ? 2 : 0);
  constexpr int KLN = 0, KGAM = HAS_LN ? 2 : 0;
  extern __shared__ float smem[];
  float* red = smem + RG * D;  // [2 parity][RG][WPR][2]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rg = warp / WPR, wir = warp % WPR;
  const int col0 = wir * (128 * NJ);
  const int stride = gridDim.x * RG;

  float acc[K][NJ][4];
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[k][j][e] = 0.f;

  int it = 0;
  for (int row = blockIdx.x * RG + rg; row < M; row += stride, ++it) {
    const size_t base = (size_t)row * D;
    // ---- issue every global load of this row up front (memory-level parallelism)
    float4 x4[HAS_LN ? NJ : 1], w4[HAS_LN ? NJ : 1], g4[HAS_GAM ? NJ : 1], r4[NJ];
    float dy[HAS_LN ? NJ : 1][4], yb[HAS_GAM ? NJ : 1][4];
    float mean = 0.f, rstd = 0.f;
    if constexpr (HAS_LN) {
      mean = MEAN[row];
      rstd = RSTD[row];
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = col0 + j * 128 + lane * 4;
      r4[j] = ld4(DR + base + c);
      if constexpr (HAS_LN) {
        x4[j] = ld4(X + base + c);
        w4[j] = ld4(W + c);
        unpack_bf16x4(DY + base + c, dy[j]);
      }
      if constexpr (HAS_GAM) {
        g4[j] = ld4(GAMMA + c);
        unpack_bf16x4(Y + base + c, yb[j]);
      }
    }
    float s1 = 0.f, s2 = 0.f;
    float xh[HAS_LN ? NJ : 1][4];
    if constexpr (HAS_LN) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const float xv[4] = {x4[j].x, x4[j].y, x4[j].z, x4[j].w}, wl[4] = {w4[j].x, w4[j].y, w4[j].z, w4[j].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          xh[j][e] = (xv[e] - mean) * rstd;
          const float a = dy[j][e] * wl[e];
          s1 += a;
          s2 += a * xh[j][e];
        }
      }
      s1 = warp_sum(s1);
      s2 = warp_sum(s2);
      if constexpr (WPR > 1) {
        float* slot = red + ((it & 1) * RG + rg) * WPR * 2;
        if (lane == 0) {
          slot[wir * 2 + 0] = s1;
          slot[wir * 2 + 1] = s2;
        }
        named_bar(1 + rg, WPR * 32);
        s1 = 0.f;
        s2 = 0.f;
#pragma unroll
        for (int w = 0; w < WPR; ++w) {
          s1 += slot[w * 2 + 0];
          s2 += slot[w * 2 + 1];
        }
      }
      s1 *= (1.0f / D);
      s2 *= (1.0f / D);
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = col0 + j * 128 + lane * 4;
      float dx[4] = {r4[j].x, r4[j].y, r4[j].z, r4[j].w};
      if constexpr (HAS_LN) {
        const float wl[4] = {w4[j].x, w4[j].y, w4[j].z, w4[j].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float a = dy[j][e] * wl[e];
          dx[e] += rstd * (a - s1 - xh[j][e] * s2);
          acc[KLN + 0][j][e] += dy[j][e] * xh[j][e];
          acc[KLN + 1][j][e] += dy[j][e];
        }
      }
      if constexpr (WRITE_DX) st4(DX + base + c, make_float4(dx[0], dx[1], dx[2], dx[3]));
      if constexpr (HAS_GAM) {
        const float gl[4] = {g4[j].x, g4[j].y, g4[j].z, g4[j].w};
        float g[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          g[e] = gl[e] * dx[e];
          acc[KGAM + 0][j][e] += dx[e] * yb[j][e];
          acc[KGAM + 1][j][e] += g[e];
        }
        store_bf16x4(G + base + c, g);
      }
    }
  }

  // Block reduction of the per-warp column accumulators over the RG row groups -> PART[blockIdx.x][k][:]
#pragma unroll
  for (int k = 0; k < K; ++k) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = col0 + j * 128 + lane * 4;
      *reinterpret_cast<float4*>(&smem[rg * D + c]) = make_float4(acc[k][j][0], acc[k][j][1], acc[k][j][2], acc[k][j][3]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
      float s = 0.f;
#pragma unroll
      for (int g = 0; g < RG; ++g) s += smem[g * D + c];
      PART[((size_t)blockIdx.x * K + k) * D + c] = s;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Column partial sums of a bf16 [M, C] matrix.  grid = (P row-chunks, ceil(C / 1024)); 128 threads x 8 columns.
__global__ void __launch_bounds__(128) colsum_bf16_kernel(const __nv_bfloat16* __restrict__ X, float* __restrict__ PART,
                                                           int M, int C) {
  const int c0 = blockIdx.y * 1024 + threadIdx.x * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (c0 + 8 <= C) {
    for (int r = blockIdx.x; r < M; r += gridDim.x) {
      uint4 u = *reinterpret_cast<const uint4*>(X + (size_t)r * C + c0);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(h[i]);
        acc[2 * i] += f.x;
        acc[2 * i + 1] += f.y;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) PART[(size_t)blockIdx.x * C + c0 + i] = acc[i];
  }
}

// Fixed-order sum over P partial rows: OUT[c] = sum_p PART[p][c]
__global__ void __launch_bounds__(256) finalize_kernel(const float* __restrict__ PART, float* __restrict__ OUT, int P, int KD) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= KD) return;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  int p = 0;
  for (; p + 4 <= P; p += 4) {
    a0 += PART[(size_t)(p + 0) * KD + c];
    a1 += PART[(size_t)(p + 1) * KD + c];
    a2 += PART[(size_t)(p + 2) * KD + c];
    a3 += PART[(size_t)(p + 3) * KD + c];
  }
  for (; p < P; ++p) a0 += PART[(size_t)p * KD + c];
  OUT[c] = (a0 + a1) + (a2 + a3);
}

// ------------------------------------------------------------------------------------------------
// Host side
static int num_sms() {
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }
  return sms;
}

#define CHECK_F32(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kFloat && (t).is_contiguous(), #t " must be a contiguous fp32 cuda tensor")
#define CHECK_BF16(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kBFloat16 && (t).is_contiguous(), #t " must be a contiguous bf16 cuda tensor")

#define DISPATCH_D(D_, FN, ...)                          \
  switch (D_) {                                          \
    case 384: FN<384>(__VA_ARGS__); break;               \
    case 768: FN<768>(__VA_ARGS__); break;               \
    case 1024: FN<1024>(__VA_ARGS__); break;             \
    default: TORCH_CHECK(false, "unsupported hidden size ", D_); \
  }

template <int D>
static void ln_fwd_launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, at::Tensor& y,
                          at::Tensor& mean, at::Tensor& rstd, int64_t M, double eps) {
  const int threads = 256;
  const int blocks = (int)((M + 7) / 8);
  ln_fwd_kernel<D><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      (int)M, (float)eps);
}

// x [M,D] fp32 -> (y bf16 [M,D], mean [M], rstd [M])
std::vector<at::Tensor> ln_fwd(at::Tensor x, at::Tensor w, at::Tensor b, double eps) {
  CHECK_F32(x); CHECK_F32(w); CHECK_F32(b);
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), D = x.size(1);
  auto y = at::empty({M, D}, x.options().dtype(at::kBFloat16));
  auto mean = at::empty({M}, x.options());
  auto rstd = at::empty({M}, x.options());
  DISPATCH_D(D, ln_fwd_launch, x, w, b, y, mean, rstd, M, eps);
  return {y, mean, rstd};
}

constexpr int kRG = 2;  // row groups per block

template <int D, int NJ, bool HAS_LN, bool HAS_GAM, bool WRITE_DX>
static void rowbwd_launch(const at::Tensor& x, const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& w,
                          const at::Tensor& dy, const at::Tensor& dr, const at::Tensor& y, const at::Tensor& gamma,
                          at::Tensor& dx, at::Tensor& g, at::Tensor& part, int64_t M, int grid) {
  constexpr int WPR = D / (128 * NJ);
  constexpr int threads = kRG * WPR * 32;
  const size_t smem = (size_t)(kRG * D + 2 * kRG * WPR * 2) * sizeof(float);
  rowbwd_kernel<D, NJ, kRG, HAS_LN, HAS_GAM, WRITE_DX><<<grid, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
      HAS_LN ? x.data_ptr<float>() : nullptr, HAS_LN ? mean.data_ptr<float>() : nullptr,
      HAS_LN ? rstd.data_ptr<float>() : nullptr, HAS_LN ? w.data_ptr<float>() : nullptr,
      HAS_LN ? reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()) : nullptr, dr.data_ptr<float>(),
      HAS_GAM ? reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()) : nullptr,
      HAS_GAM ? gamma.data_ptr<float>() : nullptr, WRITE_DX ? dx.data_ptr<float>() : nullptr,
      HAS_GAM ? reinterpret_cast<__nv_bfloat16*>(g.data_ptr<at::BFloat16>()) : nullptr, part.data_ptr<float>(), (int)M);
}

// NJ: float4 per lane per row; D=384 uses NJ=1 (3 warps/row), D=768/1024 use NJ=2 (3/4 warps per row).
template <int D>
static void rowbwd_dispatch(bool has_ln, bool has_gam, const at::Tensor& x, const at::Tensor& mean,
                            const at::Tensor& rstd, const at::Tensor& w, const at::Tensor& dy, const at::Tensor& dr,
                            const at::Tensor& y, const at::Tensor& gamma, at::Tensor& dx, at::Tensor& g,
                            at::Tensor& part, int64_t M, int grid) {
  constexpr int NJ = (D == 384) ? 1 : 2;
  if (has_ln && has_gam) rowbwd_launch<D, NJ, true, true, true>(x, mean, rstd, w, dy, dr, y, gamma, dx, g, part, M, grid);
  else if (has_ln && !has_gam) rowbwd_launch<D, NJ, true, false, true>(x, mean, rstd, w, dy, dr, y, gamma, dx, g, part, M, grid);
  else if (!has_ln && has_gam) rowbwd_launch<D, NJ, false, true, false>(x, mean, rstd, w, dy, dr, y, gamma, dx, g, part, M, grid);
  else TORCH_CHECK(false, "rowbwd needs LN and/or gamma");
}

// Returns (dx or empty, g or empty, out[K, D]) where out holds the finalized column sums in the order
//   [dW, dB] (if has_ln) then [dGamma, dBiasBranch] (if has_gam).
// Optional tensors are passed as zero-element tensors when unused.  grid = num_SMs * grid_mult CTAs (persistent).
std::vector<at::Tensor> rowbwd(at::Tensor x, at::Tensor mean, at::Tensor rstd, at::Tensor w, at::Tensor dy,
                               at::Tensor dr, at::Tensor y, at::Tensor gamma, bool has_ln, bool has_gam,
                               int64_t grid_mult) {
  CHECK_F32(dr);
  c10::cuda::CUDAGuard guard(dr.device());
  const int64_t M = dr.size(0), D = dr.size(1);
  if (has_ln) { CHECK_F32(x); CHECK_F32(mean); CHECK_F32(rstd); CHECK_F32(w); CHECK_BF16(dy); }
  if (has_gam) { CHECK_BF16(y); CHECK_F32(gamma); }
  const int K = (has_ln ? 2 : 0) + (has_gam ? 2 : 0);
  at::Tensor dx = has_ln ? at::empty({M, D}, dr.options()) : at::empty({0}, dr.options());
  at::Tensor g = has_gam ? at::empty({M, D}, dr.options().dtype(at::kBFloat16)) : at::empty({0}, dr.options().dtype(at::kBFloat16));
  int grid = (int)std::min<int64_t>((M + kRG - 1) / kRG, (int64_t)num_sms() * grid_mult);
  at::Tensor part = at::empty({grid, K, D}, dr.options());
  DISPATCH_D(D, rowbwd_dispatch, has_ln, has_gam, x, mean, rstd, w, dy, dr, y, gamma, dx, g, part, M, grid);
  at::Tensor out = at::empty({K, D}, dr.options());
  const int KD = K * (int)D;
  finalize_kernel<<<(KD + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), out.data_ptr<float>(), grid, KD);
  return {dx, g, out};
}

// Column sum of bf16 [M, C] -> fp32 [C] (deterministic two-stage).
at::Tensor colsum_bf16(at::Tensor x) {
  CHECK_BF16(x);
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), C = x.size(1);
  TORCH_CHECK(C % 8 == 0, "colsum_bf16 needs C % 8 == 0");
  const int P = (int)std::min<int64_t>(M, (int64_t)num_sms() * 2);
  const int ctiles = (int)((C + 1023) / 1024);
  at::Tensor part = at::empty({P, C}, x.options().dtype(at::kFloat));
  colsum_bf16_kernel<<<dim3(P, ctiles), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), part.data_ptr<float>(), (int)M, (int)C);
  at::Tensor out = at::empty({C}, x.options().dtype(at::kFloat));
  finalize_kernel<<<(int)((C + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), out.data_ptr<float>(), P, (int)C);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_fwd", &ln_fwd);
  m.def("rowbwd", &rowbwd);
  m.def("colsum_bf16", &colsum_bf16);
}
