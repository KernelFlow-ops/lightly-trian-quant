// Round 7: fused pre/post-processing kernels for DINOv3-EoMT semantic head segmentation.
//   prep_kernel   : BGR uint8 frame -> RGB, /255, normalize, antialiased bilinear resize
//                   (ATen triangle-filter weights, precomputed tables), tile into crops.
//   post_a_kernel : low-res mask logits (B,Q,h,w) -> bilinear upsample to crop size (on the fly)
//                   -> sigmoid -> dot with softmax(class logits) -> untile (average overlaps).
//   post_b_kernel : bilinear resize to frame size -> argmax -> uint8 mask + confidence stats.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define MAX_CROPS 8

struct CropStarts {
  int n;
  int start[MAX_CROPS];
};

__device__ __forceinline__ float ld_f(const float* p) { return *p; }
__device__ __forceinline__ float ld_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ld_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void st_o(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_o(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void st_o(__half* p, float v) { *p = __float2half_rn(v); }

// ------------------------------------------------------------------ P1: prep
template <typename OutT>
__global__ void prep_kernel(const uint8_t* __restrict__ frame,  // (H, W, 3) BGR
                            int W,
                            const int* __restrict__ ymin, const int* __restrict__ ysz,
                            const float* __restrict__ ywt, int KY,
                            const int* __restrict__ xmin, const int* __restrict__ xsz,
                            const float* __restrict__ xwt, int KX,
                            CropStarts crops, int crop, bool tall,
                            float m0, float m1, float m2, float s0, float s1, float s2,
                            OutT* __restrict__ out) {  // (ncrop, 3, crop, crop)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = crops.n * crop * crop;
  if (idx >= total) return;
  int x = idx % crop;
  int y = (idx / crop) % crop;
  int c = idx / (crop * crop);
  int X = x + (tall ? 0 : crops.start[c]);
  int Y = y + (tall ? crops.start[c] : 0);
  int y0 = ymin[Y], ny = ysz[Y];
  int x0 = xmin[X], nx = xsz[X];
  const float* wy = ywt + Y * KY;
  const float* wx = xwt + X * KX;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f;  // B, G, R
  for (int ty = 0; ty < ny; ++ty) {
    const uint8_t* row = frame + ((size_t)(y0 + ty) * W + x0) * 3;
    float r0 = 0.f, r1 = 0.f, r2 = 0.f;
    for (int tx = 0; tx < nx; ++tx) {
      float w = wx[tx];
      r0 += w * (float)row[tx * 3 + 0];
      r1 += w * (float)row[tx * 3 + 1];
      r2 += w * (float)row[tx * 3 + 2];
    }
    float w = wy[ty];
    a0 += w * r0; a1 += w * r1; a2 += w * r2;
  }
  size_t plane = (size_t)crop * crop;
  OutT* o = out + (size_t)c * 3 * plane + (size_t)y * crop + x;
  st_o(o, (a2 / 255.0f - m0) / s0);          // R
  st_o(o + plane, (a1 / 255.0f - m1) / s1);      // G
  st_o(o + 2 * plane, (a0 / 255.0f - m2) / s2);  // B
}

// ------------------------------------------------------------------ P2 kernel A
// grid: (ceil(RW/32), ceil(RH/8)); block (32, 8).  Output lr: (2, RH, RW) fp32 (bg, head).
template <typename T>
__global__ void post_a_kernel(const T* __restrict__ ml,  // (B, Q, h, w)
                              const T* __restrict__ cl,  // (B, Q, 3)
                              float* __restrict__ lr, int Q, int h, int w,
                              int crop, int RH, int RW, CropStarts crops) {
  __shared__ float2 prob[MAX_CROPS][128];  // softmax probs for classes 0,1 per (crop, query)
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < crops.n * Q; i += blockDim.x * blockDim.y) {
    int c = i / Q, q = i % Q;
    const T* p = cl + ((size_t)c * Q + q) * 3;
    float a = ld_f(p), b = ld_f(p + 1), d = ld_f(p + 2);
    float mx = fmaxf(a, fmaxf(b, d));
    float ea = expf(a - mx), eb = expf(b - mx), ed = expf(d - mx);
    float inv = 1.f / (ea + eb + ed);
    prob[c][q] = make_float2(ea * inv, eb * inv);
  }
  __syncthreads();
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;
  if (X >= RW || Y >= RH) return;
  bool tall = RH > RW;
  // bilinear source coords, crop (512) <- (h, w): scale = h/crop (align_corners=False)
  const float sh = (float)h / (float)crop, sw = (float)w / (float)crop;
  float acc0 = 0.f, acc1 = 0.f;
  int cnt = 0;
  size_t hw = (size_t)h * w;
  for (int c = 0; c < crops.n; ++c) {
    int cx = X - (tall ? 0 : crops.start[c]);
    int cy = Y - (tall ? crops.start[c] : 0);
    if (cx < 0 || cx >= crop || cy < 0 || cy >= crop) continue;
    ++cnt;
    float sy = fmaxf(sh * (cy + 0.5f) - 0.5f, 0.f);
    float sx = fmaxf(sw * (cx + 0.5f) - 0.5f, 0.f);
    int y0 = (int)sy, x0 = (int)sx;
    int y1 = y0 + (y0 < h - 1 ? 1 : 0), x1 = x0 + (x0 < w - 1 ? 1 : 0);
    float ly = fminf(fmaxf(sy - y0, 0.f), 1.f), lx = fminf(fmaxf(sx - x0, 0.f), 1.f);
    float hy = 1.f - ly, hx = 1.f - lx;
    int o00 = y0 * w + x0, o01 = y0 * w + x1, o10 = y1 * w + x0, o11 = y1 * w + x1;
    const T* base = ml + (size_t)c * Q * hw;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll 4
    for (int q = 0; q < Q; ++q) {
      const T* m = base + (size_t)q * hw;
      float v = hy * (hx * ld_f(m + o00) + lx * ld_f(m + o01)) +
                ly * (hx * ld_f(m + o10) + lx * ld_f(m + o11));
      float s = 1.f / (1.f + expf(-v));
      float2 p = prob[c][q];
      s0 += s * p.x;
      s1 += s * p.y;
    }
    acc0 += s0;
    acc1 += s1;
  }
  float inv = 1.f / (float)cnt;
  lr[(size_t)Y * RW + X] = acc0 * inv;
  lr[(size_t)RH * RW + (size_t)Y * RW + X] = acc1 * inv;
}

// ------------------------------------------------------------------ P2 kernel B
// stats (int64[4]): [0] = double sum of head score over head pixels,
//                   [1] = head pixel count, [2] = ordered-uint key of max head score.
__device__ __forceinline__ unsigned f2key(float f) {
  unsigned b = __float_as_uint(f);
  return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

template <int WRITE_LOGITS>
__global__ void post_b_kernel(const float* __restrict__ lr, int RH, int RW, int OH, int OW,
                              uint8_t* __restrict__ mask, float* __restrict__ logits,
                              unsigned long long* __restrict__ stats,
                              int cls0, int cls1, int head_ch, int head_cls) {
  // each thread: 4 consecutive x
  int x4 = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y;
  const float sh = (float)RH / (float)OH, sw = (float)RW / (float)OW;
  float sy = fmaxf(sh * (y + 0.5f) - 0.5f, 0.f);
  int y0 = (int)sy;
  int y1 = y0 + (y0 < RH - 1 ? 1 : 0);
  float ly = fminf(fmaxf(sy - y0, 0.f), 1.f), hy = 1.f - ly;
  float lsum = 0.f, lmax = -INFINITY;
  int lcnt = 0;
  {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int x = x4 * 4 + k;
      if (x >= OW) break;
      float sx = fmaxf(sw * (x + 0.5f) - 0.5f, 0.f);
      int xa = (int)sx;
      int xb = xa + (xa < RW - 1 ? 1 : 0);
      float lx = fminf(fmaxf(sx - xa, 0.f), 1.f), hx = 1.f - lx;
      float s[2];
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        const float* p = lr + (size_t)ch * RH * RW;
        s[ch] = hy * (hx * p[y0 * RW + xa] + lx * p[y0 * RW + xb]) +
                ly * (hx * p[y1 * RW + xa] + lx * p[y1 * RW + xb]);
        if (WRITE_LOGITS) logits[(size_t)ch * OH * OW + (size_t)y * OW + x] = s[ch];
      }
      int cls = (s[1] > s[0]) ? cls1 : cls0;  // numpy argmax: ties -> first
      mask[(size_t)y * OW + x] = (uint8_t)cls;
      float hs = head_ch ? s[1] : s[0];
      lmax = fmaxf(lmax, hs);
      if (cls == head_cls) { lsum += hs; ++lcnt; }
    }
  }
  // block reduction
  for (int o = 16; o > 0; o >>= 1) {
    lsum += __shfl_down_sync(0xffffffffu, lsum, o);
    lcnt += __shfl_down_sync(0xffffffffu, lcnt, o);
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffffu, lmax, o));
  }
  __shared__ float ssum[32];
  __shared__ int scnt[32];
  __shared__ float smax[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { ssum[wid] = lsum; scnt[wid] = lcnt; smax[wid] = lmax; }
  __syncthreads();
  if (wid == 0) {
    int nw = (blockDim.x + 31) >> 5;
    lsum = lane < nw ? ssum[lane] : 0.f;
    lcnt = lane < nw ? scnt[lane] : 0;
    lmax = lane < nw ? smax[lane] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) {
      lsum += __shfl_down_sync(0xffffffffu, lsum, o);
      lcnt += __shfl_down_sync(0xffffffffu, lcnt, o);
      lmax = fmaxf(lmax, __shfl_down_sync(0xffffffffu, lmax, o));
    }
    if (lane == 0) {
      atomicAdd(reinterpret_cast<double*>(&stats[0]), (double)lsum);
      atomicAdd(&stats[1], (unsigned long long)lcnt);
      atomicMax(reinterpret_cast<unsigned*>(&stats[2]), f2key(lmax));
    }
  }
}

// ------------------------------------------------------------------ host wrappers
static CropStarts make_crops(const std::vector<int64_t>& starts) {
  TORCH_CHECK((int)starts.size() <= MAX_CROPS, "too many crops");
  CropStarts c;
  c.n = (int)starts.size();
  for (int i = 0; i < c.n; ++i) c.start[i] = (int)starts[i];
  return c;
}

void prep_launch(torch::Tensor frame, torch::Tensor ymin, torch::Tensor ysz, torch::Tensor ywt,
                 torch::Tensor xmin, torch::Tensor xsz, torch::Tensor xwt,
                 std::vector<int64_t> starts, int64_t crop, std::vector<double> mean,
                 std::vector<double> stdv, torch::Tensor out) {
  CropStarts crops = make_crops(starts);
  int total = crops.n * (int)crop * (int)crop;
  int threads = 128, blocks = (total + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream();
  int W = (int)frame.size(1);
  bool tall = ywt.size(0) > xwt.size(0);
#define LAUNCH(T)                                                                             \
  prep_kernel<T><<<blocks, threads, 0, stream>>>(                                             \
      frame.data_ptr<uint8_t>(), W, ymin.data_ptr<int>(), ysz.data_ptr<int>(),                \
      ywt.data_ptr<float>(), (int)ywt.size(1), xmin.data_ptr<int>(), xsz.data_ptr<int>(),     \
      xwt.data_ptr<float>(), (int)xwt.size(1), crops, (int)crop, tall, (float)mean[0],        \
      (float)mean[1], (float)mean[2], (float)stdv[0], (float)stdv[1], (float)stdv[2],         \
      reinterpret_cast<T*>(out.data_ptr()))
  if (out.scalar_type() == torch::kFloat32) { LAUNCH(float); }
  else if (out.scalar_type() == torch::kBFloat16) { LAUNCH(__nv_bfloat16); }
  else if (out.scalar_type() == torch::kHalf) { LAUNCH(__half); }
  else TORCH_CHECK(false, "unsupported out dtype");
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void post_a_launch(torch::Tensor ml, torch::Tensor cl, torch::Tensor lr,
                   std::vector<int64_t> starts, int64_t crop) {
  CropStarts crops = make_crops(starts);
  TORCH_CHECK(ml.is_contiguous() && cl.is_contiguous() && lr.is_contiguous(), "contiguous");
  int Q = (int)ml.size(1), h = (int)ml.size(2), w = (int)ml.size(3);
  TORCH_CHECK(Q <= 128 && ml.size(0) == crops.n && cl.scalar_type() == ml.scalar_type(), "shape");
  TORCH_CHECK(cl.size(2) == 3, "expects K+1=3 classes");
  int RH = (int)lr.size(1), RW = (int)lr.size(2);
  dim3 block(32, 8), grid((RW + 31) / 32, (RH + 7) / 8);
  auto stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH(T)                                                                        \
  post_a_kernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(ml.data_ptr()), \
      reinterpret_cast<const T*>(cl.data_ptr()), lr.data_ptr<float>(), Q, h, w, (int)crop, RH, RW, crops)
  if (ml.scalar_type() == torch::kFloat32) { LAUNCH(float); }
  else if (ml.scalar_type() == torch::kBFloat16) { LAUNCH(__nv_bfloat16); }
  else if (ml.scalar_type() == torch::kHalf) { LAUNCH(__half); }
  else TORCH_CHECK(false, "unsupported dtype");
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void post_b_launch(torch::Tensor lr, torch::Tensor mask, torch::Tensor stats,
                   c10::optional<torch::Tensor> logits, int64_t cls0, int64_t cls1,
                   int64_t head_ch, int64_t head_cls) {
  int RH = (int)lr.size(1), RW = (int)lr.size(2);
  int OH = (int)mask.size(0), OW = (int)mask.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(stats.data_ptr(), 0, 4 * sizeof(int64_t), stream);
  int threads = 128;
  dim3 grid((((OW + 3) / 4) + threads - 1) / threads, OH);
  if (logits.has_value()) {
    post_b_kernel<1><<<grid, threads, 0, stream>>>(
        lr.data_ptr<float>(), RH, RW, OH, OW, mask.data_ptr<uint8_t>(),
        logits->data_ptr<float>(), reinterpret_cast<unsigned long long*>(stats.data_ptr()),
        (int)cls0, (int)cls1, (int)head_ch, (int)head_cls);
  } else {
    post_b_kernel<0><<<grid, threads, 0, stream>>>(
        lr.data_ptr<float>(), RH, RW, OH, OW, mask.data_ptr<uint8_t>(), nullptr,
        reinterpret_cast<unsigned long long*>(stats.data_ptr()),
        (int)cls0, (int)cls1, (int)head_ch, (int)head_cls);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep_launch", &prep_launch);
  m.def("post_a_launch", &post_a_launch);
  m.def("post_b_launch", &post_b_launch);
}
