// Round 11: faster post_a  (new kernel; semantics of round7_head_postprocess/impl/postfuse_ext.cu :: post_a_kernel).
//   low-res mask logits (B,Q,h,w) -> bilinear x4 upsample to the crop (align_corners=False) -> sigmoid -> dot with softmax(class logits)
//   (classes 0,1; void only in the softmax denominator) -> untile (average of the two overlapping crops) -> lr (2, RH, RW) fp32.
// Differences:
//   * Thread = one 4x4 block of crop pixels.  For an exact x4 upsample the 16 pixels of a block use only a 3x3 neighbourhood of source
//     logits (rows/cols by-1, by, by+1 with border clamping) with fixed weights {.625,.875,.125,.375}: 9 loads + 24 FLOP of
//     separable lerps per (block, query) instead of 16 x 4 loads + 16 x 7 FLOP.
//   * sigmoid(v) = rcp(1 + ex2(-v*log2e)) (MUFU ex2/rcp .approx).
//   * q is split over NG warps of the CTA (warp = q-group, lane = block along x) -> coalesced loads, smem reduction at the end.
//   * the two crops are processed by two launches (crop 0 stores, crop 1 completes the overlap with (s0 + s1) * 0.5) -> deterministic.
//   * optional query skipping: queries with p0+p1 < eps are dropped (error per output <= sum of skipped p_c <= #skipped * eps).
//   Only valid for crop == 4*h == 4*w, Q <= 128, 3 classes, 2 crops.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef NG
#define NG 8             // q-groups (warps) per CTA
#endif
#define NTHR (NG * 32)

__device__ __forceinline__ float ld_f(const float* p) { return *p; }
__device__ __forceinline__ float ld_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ld_f(const __half* p) { return __half2float(*p); }

__device__ __forceinline__ float ex2a(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcpa(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

template <typename T>
__global__ void __launch_bounds__(NTHR)
post_a2_kernel(const T* __restrict__ ml,    // (2, Q, h, w)  (this launch reads crop `c`)
               const T* __restrict__ cl,    // (2, Q, 3)
               float* __restrict__ lr,      // (2, RH, RW)
               int c, int Q, int h, int w, int RH, int RW, int start0, int start, int tall, int first, float eps) {
  __shared__ float2 prob[128];
  __shared__ int act[128];
  __shared__ int wcnt[4];
  __shared__ int s_n;
  __shared__ float red[NG][16][32];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // ---- softmax probs of this crop's queries, active list (deterministic order)
  bool a = false;
  float2 pr = make_float2(0.f, 0.f);
  if (tid < Q) {
    const T* p = cl + ((size_t)c * Q + tid) * 3;
    float x0 = ld_f(p), x1 = ld_f(p + 1), x2 = ld_f(p + 2);
    float mx = fmaxf(x0, fmaxf(x1, x2));
    float e0 = expf(x0 - mx), e1 = expf(x1 - mx), e2 = expf(x2 - mx);
    float inv = 1.f / (e0 + e1 + e2);
    pr = make_float2(e0 * inv, e1 * inv);
    a = (eps <= 0.f) || (pr.x + pr.y >= eps);
  }
  if (tid < 128) prob[tid] = pr;
  const unsigned m = __ballot_sync(0xffffffffu, a);
  if (lane == 0 && warp < 4) wcnt[warp] = __popc(m);
  __syncthreads();
  {
    const int pos = __popc(m & ((1u << lane) - 1u));
    int base = 0;
    for (int i = 0; i < warp && i < 4; ++i) base += wcnt[i];
    if (a) act[base + pos] = tid;
    if (tid == 0) s_n = wcnt[0] + wcnt[1] + wcnt[2] + wcnt[3];
  }
  __syncthreads();
  const int nact = s_n;

  // ---- this lane's 4x4 block
  const int bw = w;                                   // source width; blocks per row == w
  const int bx = blockIdx.x * 32 + lane;
  const int by = blockIdx.y;
  const int xm = max(bx - 1, 0), xp = min(bx + 1, w - 1);
  const int ym = max(by - 1, 0), yp = min(by + 1, h - 1);
  const size_t hw = (size_t)h * w;
  const T* base = ml + (size_t)c * Q * hw;
  const int o[3][3] = {{ym * w + xm, ym * w + bx, ym * w + xp},
                       {by * w + xm, by * w + bx, by * w + xp},
                       {yp * w + xm, yp * w + bx, yp * w + xp}};
  float acc0[16], acc1[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) { acc0[k] = 0.f; acc1[k] = 0.f; }

  // x weights for the 4 output columns: (x0 side weight hx, x1 side weight lx) and which source col pair they use
  // tx 0,1: cols (m, 0) with lx = .625, .875 ; tx 2,3: cols (0, p) with lx = .125, .375
  const float LX[4] = {0.625f, 0.875f, 0.125f, 0.375f};

  for (int i = warp; i < nact; i += NG) {
    const int q = act[i];
    const float2 pr = prob[q];
    const T* mq = base + (size_t)q * hw;
    float g[3][3];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
      for (int cc = 0; cc < 3; ++cc) g[r][cc] = ld_f(mq + o[r][cc]);
    // horizontal lerps per source row
    float hrz[3][4];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
#pragma unroll
      for (int tx = 0; tx < 4; ++tx) {
        float a = tx < 2 ? g[r][0] : g[r][1];
        float b = tx < 2 ? g[r][1] : g[r][2];
        hrz[r][tx] = fmaf(LX[tx], b - a, a);
      }
    }
#pragma unroll
    for (int ty = 0; ty < 4; ++ty) {
      const float ly = LX[ty];
      const int ra = ty < 2 ? 0 : 1, rb = ty < 2 ? 1 : 2;
#pragma unroll
      for (int tx = 0; tx < 4; ++tx) {
        float v = fmaf(ly, hrz[rb][tx] - hrz[ra][tx], hrz[ra][tx]);
        float s = rcpa(1.f + ex2a(-1.4426950408889634f * v));
        acc0[ty * 4 + tx] = fmaf(s, pr.x, acc0[ty * 4 + tx]);
        acc1[ty * 4 + tx] = fmaf(s, pr.y, acc1[ty * 4 + tx]);
      }
    }
  }

  // ---- reduce the NG q-groups (smem), then write
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    red[warp][k][lane] = acc0[k];
  }
  __syncthreads();
  // thread (warp = k-group, lane = block): sums class-0 outputs k = warp, warp+NG ...
  const int cropsz = 4 * h;
  for (int pass = 0; pass < 2; ++pass) {
    for (int k = warp; k < 16; k += NG) {
      float sum = 0.f;
#pragma unroll
      for (int gI = 0; gI < NG; ++gI) sum += red[gI][k][lane];
      const int ty = k >> 2, tx = k & 3;
      const int cy = by * 4 + ty, cx = (blockIdx.x * 32 + lane) * 4 + tx;
      const int Y = cy + (tall ? start : 0), X = cx + (tall ? 0 : start);
      float* dst = lr + (size_t)pass * RH * RW + (size_t)Y * RW + X;
      const int pos = tall ? Y : X;
      if (first) {
        *dst = sum;
      } else if (pos < start0 + cropsz) {
        *dst = (*dst + sum) * 0.5f;
      } else {
        *dst = sum;
      }
    }
    if (pass == 0) {
      __syncthreads();
#pragma unroll
      for (int k = 0; k < 16; ++k) red[warp][k][lane] = acc1[k];
      __syncthreads();
    }
  }
}

void post_a2_launch(torch::Tensor ml, torch::Tensor cl, torch::Tensor lr, std::vector<int64_t> starts, int64_t crop, double eps) {
  TORCH_CHECK(ml.is_contiguous() && cl.is_contiguous() && lr.is_contiguous(), "contiguous");
  TORCH_CHECK(starts.size() == 2, "post_a2 supports exactly 2 crops");
  int Q = (int)ml.size(1), h = (int)ml.size(2), w = (int)ml.size(3);
  TORCH_CHECK(Q <= 128 && ml.size(0) == 2 && cl.scalar_type() == ml.scalar_type(), "shape");
  TORCH_CHECK(cl.size(2) == 3, "expects K+1=3 classes");
  TORCH_CHECK(crop == 4 * h && crop == 4 * w && w % 32 == 0, "post_a2 needs an exact x4 upsample and w % 32 == 0");
  int RH = (int)lr.size(1), RW = (int)lr.size(2);
  bool tall = RH > RW;
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(w / 32, h), block(NTHR);
#define LAUNCH(T)                                                                                           \
  for (int c = 0; c < 2; ++c)                                                                               \
    post_a2_kernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(ml.data_ptr()),               \
        reinterpret_cast<const T*>(cl.data_ptr()), lr.data_ptr<float>(), c, Q, h, w, RH, RW, (int)starts[0], \
        (int)starts[c], tall ? 1 : 0, c == 0 ? 1 : 0, (float)eps)
  if (ml.scalar_type() == torch::kFloat32) { LAUNCH(float); }
  else if (ml.scalar_type() == torch::kBFloat16) { LAUNCH(__nv_bfloat16); }
  else if (ml.scalar_type() == torch::kHalf) { LAUNCH(__half); }
  else TORCH_CHECK(false, "unsupported dtype");
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("post_a2_launch", &post_a2_launch); }
