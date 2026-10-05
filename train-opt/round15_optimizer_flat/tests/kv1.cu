// round15_optimizer_flat: two-kernel clip + AdamW over flat fp32 buffers.
//
//   K1 sumsq_prep : deterministic global sum of squares of the flat gradient (fixed grid, fixed reduction order,
//                   final reduction by the last block via a ticket) plus the whole device-side control computation:
//                   grad norm, clip coefficient, Adam step / scheduler step counters, per-group effective
//                   lr / decay / bias corrections (TwoStageWarmupPoly lr schedule evaluated on the device).
//   K2 update     : one block per 4096-float chunk; chunk -> group id table; reads the per-group effective
//                   scalars written by K1, applies clip + AdamW (torch AdamW math) and zeroes the gradient.
//
// Everything is stream-ordered with no host reads, so a whole step (and many steps) can be CUDA-graph captured.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
constexpr int THREADS = 256;
constexpr int CHUNK = 4096;                 // floats per chunk (= 1024 float4 = 4 float4 per thread)
constexpr int VEC_PER_THREAD = CHUNK / 4 / THREADS;
constexpr int HYP = 8;                      // double per group: base_lr wd b1 b2 eps kind omb1 omb2
constexpr int EFF = 4;                      // float per group: rate decay bc2_sqrt lr

__device__ __forceinline__ float4 ld4(const float4* p) { return __ldg(p); }

__global__ void __launch_bounds__(THREADS) sumsq_prep_kernel(
    const float4* __restrict__ grad, int n_chunks, float* partial, unsigned int* ticket,
    int* ctl, const double* hyp, const double* ext_lr, int n_groups, float* eff,
    float* norm_out, float* clip_out, double max_norm, double nw, double vw, double total, double power) {
  __shared__ float warp_sum[THREADS / 32];
  __shared__ double warp_sum_d[THREADS / 32];
  __shared__ bool is_last;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int c = blockIdx.x; c < n_chunks; c += gridDim.x) {
    const float4* p = grad + (size_t)c * (CHUNK / 4);
    float4 x[VEC_PER_THREAD];
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) x[k] = ld4(p + threadIdx.x + THREADS * k);
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
      acc0 += x[k].x * x[k].x; acc1 += x[k].y * x[k].y; acc2 += x[k].z * x[k].z; acc3 += x[k].w * x[k].w;
    }
  }
  float s = (acc0 + acc1) + (acc2 + acc3);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int w = 0; w < THREADS / 32; ++w) t += warp_sum[w];
    partial[blockIdx.x] = t;
    __threadfence();
    unsigned int prev = atomicAdd(ticket, 1u);
    is_last = (prev == gridDim.x - 1);
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  // ---- last block: deterministic final reduction (double, fixed order) ----
  double d = 0.0;
  for (int i = threadIdx.x; i < (int)gridDim.x; i += THREADS) d += (double)__ldcg(partial + i);
  for (int o = 16; o > 0; o >>= 1) d += __shfl_down_sync(0xffffffffu, d, o);
  if ((threadIdx.x & 31) == 0) warp_sum_d[threadIdx.x >> 5] = d;
  __syncthreads();
  if (threadIdx.x == 0) {
    double t = 0.0;
    for (int w = 0; w < THREADS / 32; ++w) t += warp_sum_d[w];
    float norm = sqrtf((float)t);
    float clip = max_norm > 0.0 ? fminf(1.f, (float)max_norm / (norm + 1e-6f)) : 1.f;
    *norm_out = norm;
    *clip_out = clip;
    *ticket = 0u;
  }
  __syncthreads();
  const int t_adam = ctl[0] + 1;       // Adam step count of this update (1-based)
  const int s_lr = ctl[1];             // scheduler step the lr of this update is taken from
  for (int g = threadIdx.x; g < n_groups; g += THREADS) {
    const double* h = hyp + g * HYP;
    double base = h[0], wd = h[1], b1 = h[2], b2 = h[3];
    int kind = (int)h[5];
    double lr = 0.0, st = (double)s_lr;
    if (kind == 0) {            // non-ViT group
      if (nw > 0.0 && st < nw) lr = base * (st / nw);
      else {
        double adjusted = fmax(0.0, st - nw), ms = fmax(1.0, total - nw);
        lr = base * pow(fmax(0.0, 1.0 - adjusted / ms), power);
      }
    } else if (kind == 1) {     // ViT (backbone) group
      if (st < nw) lr = 0.0;
      else if (st < nw + vw) lr = base * ((st - nw) / vw);
      else {
        double adjusted = fmax(0.0, st - nw - vw), ms = fmax(1.0, total - nw - vw);
        lr = base * pow(fmax(0.0, 1.0 - adjusted / ms), power);
      }
    } else if (kind == 2) {     // externally supplied lr
      lr = ext_lr[g];
    }
    double bc1 = 1.0 - pow(b1, (double)t_adam), bc2 = 1.0 - pow(b2, (double)t_adam);
    float* e = eff + g * EFF;
    e[0] = (float)(lr / bc1);
    e[1] = (float)(1.0 - lr * wd);
    e[2] = (float)sqrt(bc2);
    e[3] = (float)lr;
  }
  __syncthreads();
  if (threadIdx.x == 0) { ctl[0] = t_adam; ctl[1] = s_lr + 1; ctl[2] = s_lr; }
}

__global__ void __launch_bounds__(THREADS) update_kernel(
    float4* __restrict__ p, float4* __restrict__ g, float4* __restrict__ m, float4* __restrict__ v,
    const int* __restrict__ chunk_gid, const double* __restrict__ hyp, const float* __restrict__ eff,
    const float* __restrict__ clip_ptr) {
  const int c = blockIdx.x;
  const int gid = chunk_gid[c];
  const float clip = *clip_ptr;
  const double* h = hyp + gid * HYP;
  const float b2 = (float)h[3], eps = (float)h[4], omb1 = (float)h[6], omb2 = (float)h[7];
  const float rate = eff[gid * EFF], decay = eff[gid * EFF + 1], bc2s = eff[gid * EFF + 2];
  const size_t base = (size_t)c * (CHUNK / 4);
  float4 pv[VEC_PER_THREAD], gv[VEC_PER_THREAD], mv[VEC_PER_THREAD], vv[VEC_PER_THREAD];
#pragma unroll
  for (int k = 0; k < VEC_PER_THREAD; ++k) {
    size_t i = base + threadIdx.x + THREADS * k;
    gv[k] = __ldcs(g + i); mv[k] = __ldcs(m + i); vv[k] = __ldcs(v + i); pv[k] = __ldcs(p + i);
  }
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < VEC_PER_THREAD; ++k) {
    float* pp = reinterpret_cast<float*>(&pv[k]);
    float* gg = reinterpret_cast<float*>(&gv[k]);
    float* mm = reinterpret_cast<float*>(&mv[k]);
    float* vs = reinterpret_cast<float*>(&vv[k]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float gi = gg[j] * clip;
      float d = gi - mm[j];
      float mi = omb1 < 0.5f ? mm[j] + omb1 * d : gi - d * (1.f - omb1);   // torch lerp
      float vi = b2 * vs[j] + omb2 * gi * gi;
      float denom = sqrtf(vi) / bc2s + eps;
      pp[j] = pp[j] * decay - rate * mi / denom;
      mm[j] = mi;
      vs[j] = vi;
    }
  }
#pragma unroll
  for (int k = 0; k < VEC_PER_THREAD; ++k) {
    size_t i = base + threadIdx.x + THREADS * k;
    __stcs(p + i, pv[k]); __stcs(m + i, mv[k]); __stcs(v + i, vv[k]); __stcs(g + i, zero);
  }
}
}  // namespace

void sumsq_prep(torch::Tensor grad, torch::Tensor partial, torch::Tensor ticket, torch::Tensor ctl,
                torch::Tensor hyp, torch::Tensor ext_lr, torch::Tensor eff, torch::Tensor norm, torch::Tensor clip,
                int64_t grid, double max_norm, double nw, double vw, double total, double power) {
  TORCH_CHECK(grad.is_cuda() && grad.scalar_type() == torch::kFloat32 && grad.is_contiguous(), "flat fp32 CUDA grad required");
  TORCH_CHECK(grad.numel() % CHUNK == 0, "flat buffer must be a multiple of the chunk size");
  c10::cuda::CUDAGuard guard(grad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int n_chunks = (int)(grad.numel() / CHUNK);
  int g = (int)std::min<int64_t>(grid, n_chunks);
  TORCH_CHECK(partial.numel() >= g, "partial buffer too small");
  sumsq_prep_kernel<<<g, THREADS, 0, stream>>>(
      reinterpret_cast<const float4*>(grad.data_ptr<float>()), n_chunks, partial.data_ptr<float>(),
      reinterpret_cast<unsigned int*>(ticket.data_ptr<int>()), ctl.data_ptr<int>(), hyp.data_ptr<double>(),
      ext_lr.data_ptr<double>(), (int)hyp.size(0), eff.data_ptr<float>(), norm.data_ptr<float>(), clip.data_ptr<float>(),
      max_norm, nw, vw, total, power);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update(torch::Tensor master, torch::Tensor grad, torch::Tensor m, torch::Tensor v, torch::Tensor chunk_gid,
            torch::Tensor hyp, torch::Tensor eff, torch::Tensor clip) {
  TORCH_CHECK(master.is_cuda() && master.numel() == grad.numel() && m.numel() == grad.numel() && v.numel() == grad.numel(), "shape");
  c10::cuda::CUDAGuard guard(grad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int n_chunks = (int)(grad.numel() / CHUNK);
  TORCH_CHECK(chunk_gid.numel() == n_chunks, "chunk table size");
  update_kernel<<<n_chunks, THREADS, 0, stream>>>(
      reinterpret_cast<float4*>(master.data_ptr<float>()), reinterpret_cast<float4*>(grad.data_ptr<float>()),
      reinterpret_cast<float4*>(m.data_ptr<float>()), reinterpret_cast<float4*>(v.data_ptr<float>()),
      chunk_gid.data_ptr<int>(), hyp.data_ptr<double>(), eff.data_ptr<float>(), clip.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("sumsq_prep", &sumsq_prep, "deterministic sum of squares + device-side clip/lr/step control");
  mod.def("update", &update, "AdamW update over flat buffers (zeroes the gradient)");
}
