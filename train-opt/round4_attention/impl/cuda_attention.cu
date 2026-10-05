#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// A deliberately simple deterministic CUDA reference kernel for packed-QKV
// attention.  It is correctness-first: one CUDA thread owns one (batch,head,
// query/key) row and accumulates softmax values in fp32.  The production
// Triton kernels in this round remain available separately; this extension is
// the true CUDA C++ path used by the cuda_ref variant.

namespace {

constexpr float kScale = 0.125f;
constexpr float kLog2 = 0.6931471805599453094f;
// 32 keeps the float32 instantiation under the default 48 KiB static shared
// memory limit while still giving bf16 a coalesced two-stage tile.  A later
// specialization can opt into 99 KiB dynamic shared memory on sm_120.
constexpr int kTileN = 32;
constexpr int kWarps = 4;

template <typename scalar_t>
__device__ __forceinline__ float qkv_load(const scalar_t* x, int b, int n,
                                          int which, int h, int d,
                                          int N, int H) {
  const int64_t off = (((int64_t)b * N + n) * 3 + which) * H * 64
                    + (int64_t)h * 64 + d;
  return static_cast<float>(x[off]);
}

__device__ __forceinline__ bool key_allowed(const bool* mask, bool has_mask,
                                            int b, int m, int n, int Q,
                                            int prefix, int N) {
  if (!has_mask || m >= Q || n < prefix) return true;
  return mask[((int64_t)b * Q + m) * (N - prefix) + (n - prefix)];
}

template <typename scalar_t>
__device__ __forceinline__ float rotated_load(const scalar_t* x,
                                              const float* sin,
                                              const float* cos,
                                              bool has_rope, int b, int n,
                                              int which, int h, int d, int N,
                                              int H, int prefix) {
  float v = qkv_load(x, b, n, which, h, d, N, H);
  if (!has_rope || n < prefix || which == 2) return v;
  const int od = d < 32 ? d + 32 : d - 32;
  const int64_t po = (int64_t)(n - prefix) * 64 + d;
  const float s = sin[po];
  const float c = cos[po];
  const float ov = qkv_load(x, b, n, which, h, od, N, H);
  return v * c + ov * (d < 32 ? -s : s);
}

template <typename scalar_t>
__global__ void attention_forward_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    scalar_t* __restrict__ out, float* __restrict__ lse, int B, int N, int H,
    int prefix, int Q, bool has_rope, bool has_mask) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = B * H * N;
  if (row >= rows) return;
  const int m = row % N;
  const int bh = row / N;
  const int h = bh % H;
  const int b = bh / H;

  float q[64];
  #pragma unroll
  for (int d = 0; d < 64; ++d)
    q[d] = rotated_load(qkv, sin, cos, has_rope, b, m, 0, h, d, N, H, prefix);

  float mx = -INFINITY;
  for (int n = 0; n < N; ++n) {
    if (!key_allowed(mask, has_mask, b, m, n, Q, prefix, N)) continue;
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < 64; ++d)
      dot += q[d] * rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix);
    mx = fmaxf(mx, dot * kScale);
  }
  float den = 0.f;
  for (int n = 0; n < N; ++n) {
    if (!key_allowed(mask, has_mask, b, m, n, Q, prefix, N)) continue;
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < 64; ++d)
      dot += q[d] * rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix);
    den += expf(dot * kScale - mx);
  }
  const float inv = 1.f / den;
  lse[(int64_t)bh * N + m] = (mx + logf(den)) / kLog2;
  for (int d = 0; d < 64; ++d) {
    float acc = 0.f;
    for (int n = 0; n < N; ++n) {
      if (!key_allowed(mask, has_mask, b, m, n, Q, prefix, N)) continue;
      float dot = 0.f;
      #pragma unroll
      for (int k = 0; k < 64; ++k)
        dot += q[k] * rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, k, N, H, prefix);
      const float p = expf(dot * kScale - mx) * inv;
      acc += p * qkv_load(qkv, b, n, 2, h, d, N, H);
    }
    out[((int64_t)b * N + m) * H * 64 + (int64_t)h * 64 + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__device__ __forceinline__ float grad_out_load(const scalar_t* go, int b, int m,
                                               int h, int d, int N, int H) {
  return static_cast<float>(go[((int64_t)b * N + m) * H * 64 + (int64_t)h * 64 + d]);
}

// ---------------------------------------------------------------------------
// FlashAttention-style tiled kernels.
//
// One warp owns one query (or key) row.  K/V (forward and dQ) and Q (dK/dV)
// are staged in shared memory in 64-token tiles.  The softmax is updated with
// the online recurrence, so the score matrix and probabilities never reach
// global memory.  This is deliberately written against CUDA primitives that
// are available on sm_120 (warp shuffles, cp.async-compatible coalesced
// global loads and shared memory); it does not rely on wgmma/tcgen05, which
// are not a supported programming interface on Blackwell consumer parts.
template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void attention_forward_tiled_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    scalar_t* __restrict__ out, float* __restrict__ lse, int B, int N, int H,
    int prefix, int Q, bool has_rope, bool has_mask) {
  __shared__ scalar_t sk[kTileN * 64];
  __shared__ scalar_t sv[kTileN * 64];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int qblocks = (N + kWarps - 1) / kWarps;
  const int bh = blockIdx.x / qblocks;
  const int mpos = (blockIdx.x % qblocks) * kWarps + warp;
  const int row = bh * N + mpos;
  const int rows = B * H * N;
  if (row >= rows) return;
  const int h = bh % H;
  const int b = bh / H;
  float q0 = rotated_load(qkv, sin, cos, has_rope, b, mpos, 0, h, lane,
                          N, H, prefix);
  float q1 = rotated_load(qkv, sin, cos, has_rope, b, mpos, 0, h, lane + 32,
                          N, H, prefix);
  float mval = -INFINITY, den = 0.f, acc0 = 0.f, acc1 = 0.f;
  for (int base = 0; base < N; base += kTileN) {
    // All warps cooperatively stage this tile.  Each scalar is loaded once
    // from the packed QKV tensor and reused by four query rows.
    for (int idx = tid; idx < kTileN * 64; idx += 128) {
      const int kn = idx >> 6, d = idx & 63;
      const int n = base + kn;
      if (n < N) {
        sk[idx] = static_cast<scalar_t>(rotated_load(
            qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix));
        sv[idx] = static_cast<scalar_t>(qkv_load(qkv, b, n, 2, h, d, N, H));
      } else {
        sk[idx] = scalar_t(0);
        sv[idx] = scalar_t(0);
      }
    }
    __syncthreads();
    const int lim = min(kTileN, N - base);
    for (int kn = 0; kn < lim; ++kn) {
      const int n = base + kn;
      if (!key_allowed(mask, has_mask, b, mpos, n, Q, prefix, N)) continue;
      float part = q0 * static_cast<float>(sk[kn * 64 + lane]) +
                   q1 * static_cast<float>(sk[kn * 64 + lane + 32]);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off);
      float score = __shfl_sync(0xffffffffu, part, 0) * kScale;
      float old_scale, prob, new_m;
      if (lane == 0) {
        new_m = fmaxf(mval, score);
        old_scale = isfinite(mval) ? __expf(mval - new_m) : 0.f;
        prob = __expf(score - new_m);
        den = den * old_scale + prob;
        mval = new_m;
      }
      old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
      prob = __shfl_sync(0xffffffffu, prob, 0);
      acc0 = acc0 * old_scale + prob * static_cast<float>(sv[kn * 64 + lane]);
      acc1 = acc1 * old_scale + prob * static_cast<float>(sv[kn * 64 + lane + 32]);
    }
    __syncthreads();
  }
  den = __shfl_sync(0xffffffffu, den, 0);
  const int64_t ob = ((int64_t)b * N + mpos) * H * 64 + (int64_t)h * 64;
  den = __shfl_sync(0xffffffffu, den, 0);
  out[ob + lane] = static_cast<scalar_t>(acc0 / den);
  out[ob + lane + 32] = static_cast<scalar_t>(acc1 / den);
  if (lane == 0) lse[(int64_t)bh * N + mpos] = (mval + logf(den)) / kLog2;
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void attention_backward_q_tiled_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    const scalar_t* __restrict__ out, const float* __restrict__ lse,
    const scalar_t* __restrict__ grad_out, scalar_t* __restrict__ grad_qkv,
    int B, int N, int H, int prefix, int Q, bool has_rope, bool has_mask) {
  __shared__ scalar_t sk[kTileN * 64];
  __shared__ scalar_t sv[kTileN * 64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int qblocks = (N + kWarps - 1) / kWarps;
  const int bh = blockIdx.x / qblocks;
  const int mpos = (blockIdx.x % qblocks) * kWarps + warp;
  const int row = bh * N + mpos, rows = B * H * N;
  if (row >= rows) return;
  const int h = bh % H, b = bh / H;
  const float q0 = rotated_load(qkv, sin, cos, has_rope, b, mpos, 0, h, lane,
                                N, H, prefix);
  const float q1 = rotated_load(qkv, sin, cos, has_rope, b, mpos, 0, h, lane + 32,
                                N, H, prefix);
  const int64_t ob = ((int64_t)b * N + mpos) * H * 64 + (int64_t)h * 64;
  float delta_part = static_cast<float>(out[ob + lane]) * static_cast<float>(grad_out[ob + lane]) +
                     static_cast<float>(out[ob + lane + 32]) * static_cast<float>(grad_out[ob + lane + 32]);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) delta_part += __shfl_down_sync(0xffffffffu, delta_part, off);
  const float delta = __shfl_sync(0xffffffffu, delta_part, 0);
  const float lse_nat = lse[(int64_t)bh * N + mpos] * kLog2;
  float dq0 = 0.f, dq1 = 0.f;
  for (int base = 0; base < N; base += kTileN) {
    for (int idx = tid; idx < kTileN * 64; idx += 128) {
      const int kn = idx >> 6, d = idx & 63, n = base + kn;
      if (n < N) {
        sk[idx] = static_cast<scalar_t>(rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix));
        sv[idx] = static_cast<scalar_t>(qkv_load(qkv, b, n, 2, h, d, N, H));
      } else { sk[idx] = scalar_t(0); sv[idx] = scalar_t(0); }
    }
    __syncthreads();
    const int lim = min(kTileN, N - base);
    for (int kn = 0; kn < lim; ++kn) {
      const int n = base + kn;
      if (!key_allowed(mask, has_mask, b, mpos, n, Q, prefix, N)) continue;
      float part = q0 * static_cast<float>(sk[kn * 64 + lane]) + q1 * static_cast<float>(sk[kn * 64 + lane + 32]);
      float dp_part = static_cast<float>(grad_out[ob + lane]) * static_cast<float>(sv[kn * 64 + lane]) +
                      static_cast<float>(grad_out[ob + lane + 32]) * static_cast<float>(sv[kn * 64 + lane + 32]);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) { part += __shfl_down_sync(0xffffffffu, part, off); dp_part += __shfl_down_sync(0xffffffffu, dp_part, off); }
      const float score = __shfl_sync(0xffffffffu, part, 0) * kScale;
      const float dp = __shfl_sync(0xffffffffu, dp_part, 0);
      const float p = __expf(score - lse_nat);
      const float ds = p * (dp - delta) * kScale;
      dq0 += ds * static_cast<float>(sk[kn * 64 + lane]);
      dq1 += ds * static_cast<float>(sk[kn * 64 + lane + 32]);
    }
    __syncthreads();
  }
  if (has_rope && mpos >= prefix) {
    const int od0 = lane + 32;
    const float s0 = sin[(int64_t)(mpos - prefix) * 64 + lane], c0 = cos[(int64_t)(mpos - prefix) * 64 + lane];
    const float s1 = sin[(int64_t)(mpos - prefix) * 64 + od0], c1 = cos[(int64_t)(mpos - prefix) * 64 + od0];
    const float a0 = dq0, a1 = dq1; dq0 = a0 * c0 + a1 * s0; dq1 = a1 * c1 - a0 * s1;
  }
  const int64_t qb = (((int64_t)b * N + mpos) * 3) * H * 64 + (int64_t)h * 64;
  grad_qkv[qb + lane] = static_cast<scalar_t>(dq0);
  grad_qkv[qb + lane + 32] = static_cast<scalar_t>(dq1);
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void attention_backward_kv_tiled_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    const scalar_t* __restrict__ out, const float* __restrict__ lse,
    const scalar_t* __restrict__ grad_out, scalar_t* __restrict__ grad_qkv,
    int B, int N, int H, int prefix, int Q, bool has_rope, bool has_mask) {
  __shared__ scalar_t sq[kTileN * 64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int qblocks = (N + kWarps - 1) / kWarps;
  const int bh = blockIdx.x / qblocks;
  const int n = (blockIdx.x % qblocks) * kWarps + warp;
  const int row = bh * N + n, rows = B * H * N;
  if (row >= rows) return;
  const int h = bh % H, b = bh / H;
  const float k0 = rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, lane, N, H, prefix);
  const float k1 = rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, lane + 32, N, H, prefix);
  const float v0 = qkv_load(qkv, b, n, 2, h, lane, N, H), v1 = qkv_load(qkv, b, n, 2, h, lane + 32, N, H);
  float dk0 = 0.f, dk1 = 0.f, dv0 = 0.f, dv1 = 0.f;
  for (int base = 0; base < N; base += kTileN) {
    for (int idx = tid; idx < kTileN * 64; idx += 128) {
      const int m = idx >> 6, d = idx & 63, mp = base + m;
      sq[idx] = mp < N ? static_cast<scalar_t>(rotated_load(qkv, sin, cos, has_rope, b, mp, 0, h, d, N, H, prefix)) : scalar_t(0);
    }
    __syncthreads();
    const int lim = min(kTileN, N - base);
    for (int mi = 0; mi < lim; ++mi) {
      const int mpos = base + mi;
      if (!key_allowed(mask, has_mask, b, mpos, n, Q, prefix, N)) continue;
      const int64_t ob = ((int64_t)b * N + mpos) * H * 64 + (int64_t)h * 64;
      float part = static_cast<float>(sq[mi * 64 + lane]) * k0 + static_cast<float>(sq[mi * 64 + lane + 32]) * k1;
      float dp_part = static_cast<float>(grad_out[ob + lane]) * v0 + static_cast<float>(grad_out[ob + lane + 32]) * v1;
      float de_part = static_cast<float>(out[ob + lane]) * static_cast<float>(grad_out[ob + lane]) + static_cast<float>(out[ob + lane + 32]) * static_cast<float>(grad_out[ob + lane + 32]);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) { part += __shfl_down_sync(0xffffffffu, part, off); dp_part += __shfl_down_sync(0xffffffffu, dp_part, off); de_part += __shfl_down_sync(0xffffffffu, de_part, off); }
      const float score = __shfl_sync(0xffffffffu, part, 0) * kScale;
      const float dp = __shfl_sync(0xffffffffu, dp_part, 0), delta = __shfl_sync(0xffffffffu, de_part, 0);
      const float p = __expf(score - lse[(int64_t)bh * N + mpos] * kLog2);
      const float ds = p * (dp - delta) * kScale;
      dk0 += ds * static_cast<float>(sq[mi * 64 + lane]); dk1 += ds * static_cast<float>(sq[mi * 64 + lane + 32]);
      dv0 += p * static_cast<float>(grad_out[ob + lane]); dv1 += p * static_cast<float>(grad_out[ob + lane + 32]);
    }
    __syncthreads();
  }
  if (has_rope && n >= prefix) {
    const int od0 = lane + 32;
    const float s0 = sin[(int64_t)(n - prefix) * 64 + lane], c0 = cos[(int64_t)(n - prefix) * 64 + lane];
    const float s1 = sin[(int64_t)(n - prefix) * 64 + od0], c1 = cos[(int64_t)(n - prefix) * 64 + od0];
    const float a0 = dk0, a1 = dk1; dk0 = a0 * c0 + a1 * s0; dk1 = a1 * c1 - a0 * s1;
  }
  const int64_t kb = (((int64_t)b * N + n) * 3) * H * 64 + (int64_t)h * 64;
  grad_qkv[kb + H * 64 + lane] = static_cast<scalar_t>(dk0); grad_qkv[kb + H * 64 + lane + 32] = static_cast<scalar_t>(dk1);
  grad_qkv[kb + 2 * H * 64 + lane] = static_cast<scalar_t>(dv0); grad_qkv[kb + 2 * H * 64 + lane + 32] = static_cast<scalar_t>(dv1);
}

template <typename scalar_t>
__device__ __forceinline__ float out_load(const scalar_t* out, int b, int m,
                                          int h, int d, int N, int H) {
  return static_cast<float>(out[((int64_t)b * N + m) * H * 64 + (int64_t)h * 64 + d]);
}

__device__ __forceinline__ float inverse_rope(float g, float go, int d,
                                              int pos, const float* sin,
                                              const float* cos, bool active) {
  if (!active) return g;
  const float s = sin[(int64_t)pos * 64 + d];
  const float c = cos[(int64_t)pos * 64 + d];
  return g * c + go * (d < 32 ? s : -s);
}

template <typename scalar_t>
__global__ void attention_backward_q_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    const scalar_t* __restrict__ out, const float* __restrict__ lse,
    const scalar_t* __restrict__ grad_out, scalar_t* __restrict__ grad_qkv,
    int B, int N, int H, int prefix, int Q, bool has_rope, bool has_mask) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = B * H * N;
  if (row >= rows) return;
  const int m = row % N, bh = row / N, h = bh % H, b = bh / H;
  float q[64], dq[64];
  for (int d = 0; d < 64; ++d) {
    q[d] = rotated_load(qkv, sin, cos, has_rope, b, m, 0, h, d, N, H, prefix);
    dq[d] = 0.f;
  }
  float delta = 0.f;
  for (int d = 0; d < 64; ++d)
    delta += out_load(out, b, m, h, d, N, H) * grad_out_load(grad_out, b, m, h, d, N, H);
  const float lse_nat = lse[(int64_t)bh * N + m] * kLog2;
  for (int n = 0; n < N; ++n) {
    if (!key_allowed(mask, has_mask, b, m, n, Q, prefix, N)) continue;
    float dot = 0.f, dp = 0.f;
    float v[64];
    for (int d = 0; d < 64; ++d) {
      dot += q[d] * rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix);
      v[d] = qkv_load(qkv, b, n, 2, h, d, N, H);
      dp += grad_out_load(grad_out, b, m, h, d, N, H) * v[d];
    }
    const float p = expf(dot * kScale - lse_nat);
    const float ds = p * (dp - delta) * kScale;
    for (int d = 0; d < 64; ++d)
      dq[d] += ds * rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix);
  }
  for (int d = 0; d < 64; ++d) {
    float g = dq[d];
    if (has_rope && m >= prefix) {
      const int od = d < 32 ? d + 32 : d - 32;
      g = inverse_rope(dq[d], dq[od], d, m - prefix, sin, cos, true);
    }
    const int64_t off = (((int64_t)b * N + m) * 3) * H * 64 + (int64_t)h * 64 + d;
    grad_qkv[off] = static_cast<scalar_t>(g);
  }
}

template <typename scalar_t>
__global__ void attention_backward_kv_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ sin,
    const float* __restrict__ cos, const bool* __restrict__ mask,
    const scalar_t* __restrict__ out, const float* __restrict__ lse,
    const scalar_t* __restrict__ grad_out, scalar_t* __restrict__ grad_qkv,
    int B, int N, int H, int prefix, int Q, bool has_rope, bool has_mask) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = B * H * N;
  if (row >= rows) return;
  const int n = row % N, bh = row / N, h = bh % H, b = bh / H;
  float dk[64], dv[64];
  for (int d = 0; d < 64; ++d) { dk[d] = 0.f; dv[d] = 0.f; }
  float k[64], v[64];
  for (int d = 0; d < 64; ++d) {
    k[d] = rotated_load(qkv, sin, cos, has_rope, b, n, 1, h, d, N, H, prefix);
    v[d] = qkv_load(qkv, b, n, 2, h, d, N, H);
  }
  for (int m = 0; m < N; ++m) {
    if (!key_allowed(mask, has_mask, b, m, n, Q, prefix, N)) continue;
    float q[64], dot = 0.f, dp = 0.f, delta = 0.f;
    for (int d = 0; d < 64; ++d) {
      q[d] = rotated_load(qkv, sin, cos, has_rope, b, m, 0, h, d, N, H, prefix);
      dot += q[d] * k[d];
      delta += out_load(out, b, m, h, d, N, H) * grad_out_load(grad_out, b, m, h, d, N, H);
      dp += grad_out_load(grad_out, b, m, h, d, N, H) * v[d];
    }
    const float lse_nat = lse[(int64_t)bh * N + m] * kLog2;
    const float p = expf(dot * kScale - lse_nat);
    const float ds = p * (dp - delta) * kScale;
    for (int d = 0; d < 64; ++d) {
      dk[d] += ds * q[d];
      dv[d] += p * grad_out_load(grad_out, b, m, h, d, N, H);
    }
  }
  for (int d = 0; d < 64; ++d) {
    float gk = dk[d];
    if (has_rope && n >= prefix) {
      const int od = d < 32 ? d + 32 : d - 32;
      gk = inverse_rope(dk[d], dk[od], d, n - prefix, sin, cos, true);
    }
    const int64_t base = (((int64_t)b * N + n) * 3) * H * 64 + (int64_t)h * 64 + d;
    grad_qkv[base + (int64_t)H * 64] = static_cast<scalar_t>(gk);
    grad_qkv[base + (int64_t)2 * H * 64] = static_cast<scalar_t>(dv[d]);
  }
}

template <typename scalar_t>
void launch_forward(const torch::Tensor& qkv, const torch::Tensor& sin,
                    const torch::Tensor& cos, const torch::Tensor& mask,
                    torch::Tensor& out, torch::Tensor& lse, int prefix, int Q) {
  const int B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  const bool hr = sin.numel() != 0, hm = mask.numel() != 0;
  const int blocks = (B * H * N + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_forward_kernel<scalar_t><<<blocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), B, N, H, prefix, Q, hr, hm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_tiled(const torch::Tensor& qkv, const torch::Tensor& sin,
                          const torch::Tensor& cos, const torch::Tensor& mask,
                          torch::Tensor& out, torch::Tensor& lse, int prefix, int Q) {
  const int B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  const bool hr = sin.numel() != 0, hm = mask.numel() != 0;
  const int qblocks = (N + kWarps - 1) / kWarps;
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_forward_tiled_kernel<scalar_t><<<B * H * qblocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), B, N, H, prefix, Q, hr, hm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward(const torch::Tensor& qkv, const torch::Tensor& sin,
                     const torch::Tensor& cos, const torch::Tensor& mask,
                     const torch::Tensor& out, const torch::Tensor& lse,
                     const torch::Tensor& grad_out, torch::Tensor& grad_qkv,
                     int prefix, int Q) {
  const int B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  const bool hr = sin.numel() != 0, hm = mask.numel() != 0;
  const int blocks = (B * H * N + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_backward_q_kernel<scalar_t><<<blocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), grad_out.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(), B, N, H, prefix, Q, hr, hm);
  attention_backward_kv_kernel<scalar_t><<<blocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), grad_out.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(), B, N, H, prefix, Q, hr, hm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_tiled(const torch::Tensor& qkv, const torch::Tensor& sin,
                           const torch::Tensor& cos, const torch::Tensor& mask,
                           const torch::Tensor& out, const torch::Tensor& lse,
                           const torch::Tensor& grad_out, torch::Tensor& grad_qkv,
                           int prefix, int Q) {
  const int B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  const bool hr = sin.numel() != 0, hm = mask.numel() != 0;
  const int qblocks = (N + kWarps - 1) / kWarps;
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_backward_q_tiled_kernel<scalar_t><<<B * H * qblocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), grad_out.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(), B, N, H, prefix, Q, hr, hm);
  attention_backward_kv_tiled_kernel<scalar_t><<<B * H * qblocks, 128, 0, stream>>>(
      qkv.data_ptr<scalar_t>(), hr ? sin.data_ptr<float>() : nullptr,
      hr ? cos.data_ptr<float>() : nullptr, hm ? mask.data_ptr<bool>() : nullptr,
      out.data_ptr<scalar_t>(), lse.data_ptr<float>(), grad_out.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(), B, N, H, prefix, Q, hr, hm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> forward(torch::Tensor qkv, torch::Tensor sin,
                                   torch::Tensor cos, torch::Tensor mask,
                                   int64_t prefix, int64_t Q) {
  TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous(), "qkv must be contiguous CUDA");
  TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3 && qkv.size(4) == 64,
             "expected [B,N,3,H,64]");
  const auto B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  auto out = torch::empty({B, N, H, 64}, qkv.options());
  auto lse = torch::empty({B, H, N}, qkv.options().dtype(torch::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, qkv.scalar_type(),
                                  "cuda_attention_forward", [&] {
    launch_forward<scalar_t>(qkv, sin, cos, mask, out, lse, (int)prefix, (int)Q);
  });
  return {out, lse};
}

torch::Tensor backward(torch::Tensor qkv, torch::Tensor sin, torch::Tensor cos,
                       torch::Tensor mask, torch::Tensor out, torch::Tensor lse,
                       torch::Tensor grad_out, int64_t prefix, int64_t Q) {
  auto grad_qkv = torch::empty_like(qkv);
  const auto B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, qkv.scalar_type(),
                                  "cuda_attention_backward", [&] {
    launch_backward<scalar_t>(qkv, sin, cos, mask, out, lse, grad_out,
                              grad_qkv, (int)prefix, (int)Q);
  });
  return grad_qkv;
}

std::vector<torch::Tensor> forward_fast(torch::Tensor qkv, torch::Tensor sin,
                                         torch::Tensor cos, torch::Tensor mask,
                                         int64_t prefix, int64_t Q) {
  TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous(), "qkv must be contiguous CUDA");
  const auto B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  auto out = torch::empty({B, N, H, 64}, qkv.options());
  auto lse = torch::empty({B, H, N}, qkv.options().dtype(torch::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, qkv.scalar_type(),
                                  "cuda_attention_forward_fast", [&] {
    launch_forward_tiled<scalar_t>(qkv, sin, cos, mask, out, lse, (int)prefix, (int)Q);
  });
  return {out, lse};
}

torch::Tensor backward_fast(torch::Tensor qkv, torch::Tensor sin, torch::Tensor cos,
                            torch::Tensor mask, torch::Tensor out, torch::Tensor lse,
                            torch::Tensor grad_out, int64_t prefix, int64_t Q) {
  auto grad_qkv = torch::zeros_like(qkv);
  const auto B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, qkv.scalar_type(),
                                  "cuda_attention_backward_fast", [&] {
    launch_backward_tiled<scalar_t>(qkv, sin, cos, mask, out, lse, grad_out,
                                    grad_qkv, (int)prefix, (int)Q);
  });
  return grad_qkv;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "packed QKV attention forward (CUDA)");
  m.def("backward", &backward, "packed QKV attention backward (CUDA)");
  m.def("forward_fast", &forward_fast, "tiled online-softmax attention forward (CUDA)");
  m.def("backward_fast", &backward_fast, "tiled attention backward (CUDA)");
}
