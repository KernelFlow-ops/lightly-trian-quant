// Round 7 / P3: fused ScaleBlock tail.
//   ConvTranspose2d(k=2,s=2) is done as a plain bf16 GEMM  X[B*Hs*Ws, C] @ Wr[C, 4C]
//   (column = (ki*2+kj)*C + co, "pre-shuffle" layout).  This kernel then fuses
//     + conv bias, exact-erf GELU, pixel-shuffle read, depthwise 3x3 (zero pad),
//     LayerNorm over channels (fp32 stats, affine)  ->  NHWC bf16 output.
//   One block = a 2-row x 8-col output tile for all C channels; the 4x10 halo is staged in
//   smem as bf16 (post-GELU), one warp per output pixel, each lane owns C/32 channels.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define TILE_H 2
#define TILE_W 8
#define HALO_H (TILE_H + 2)
#define HALO_W (TILE_W + 2)
#define NTHREADS 512
#define MAXV 4  // channels per lane = MAXV*8 (C <= 1024)

__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__global__ void __launch_bounds__(NTHREADS)
dw_ln_kernel(const __nv_bfloat16* __restrict__ G,     // [B*Hs*Ws, 4C]
             const float* __restrict__ bias,          // [C]  (already bf16-rounded)
             const __nv_bfloat16* __restrict__ dww,   // [9, C]
             const float* __restrict__ gamma, const float* __restrict__ beta,  // [C]
             __nv_bfloat16* __restrict__ out,         // [B, H, W, C]
             int Hs, int Ws, int C, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [HALO_H*HALO_W][C]
  const int H = 2 * Hs, W = 2 * Ws;
  const int b = blockIdx.z;
  const int y0 = blockIdx.y * TILE_H, x0 = blockIdx.x * TILE_W;
  const int tid = threadIdx.x;
  const int vpp = C / 8;  // 16B vectors per pixel

  // ---- phase 1: halo load + bias + GELU -> smem
  for (int i = tid; i < HALO_H * HALO_W * vpp; i += NTHREADS) {
    int pix = i / vpp, v = i % vpp;
    int hy = pix / HALO_W, hx = pix % HALO_W;
    int Y = y0 - 1 + hy, X = x0 - 1 + hx;
    uint4 outv = make_uint4(0, 0, 0, 0);
    if (Y >= 0 && Y < H && X >= 0 && X < W) {
      size_t row = ((size_t)b * Hs + (Y >> 1)) * Ws + (X >> 1);
      int sub = ((Y & 1) << 1) | (X & 1);
      uint4 raw = *reinterpret_cast<const uint4*>(G + row * 4 * C + (size_t)sub * C + v * 8);
      float f[8];
      unpack8(raw, f);
      const float4* bp = reinterpret_cast<const float4*>(bias + v * 8);
      float4 b0 = bp[0], b1 = bp[1];
      float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      __nv_bfloat162 o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        // conv output is rounded to bf16 in the reference (bias add in bf16 GEMM epilogue)
        float a = gelu_erf(f[2 * k] + bb[2 * k]);
        float c = gelu_erf(f[2 * k + 1] + bb[2 * k + 1]);
        o[k] = __floats2bfloat162_rn(a, c);
      }
      outv = *reinterpret_cast<uint4*>(o);
    }
    *reinterpret_cast<uint4*>(tile + (size_t)pix * C + v * 8) = outv;
  }
  __syncthreads();

  // ---- phase 2: depthwise 3x3 + LN, one warp per output pixel
  const int lane = tid & 31, warp = tid >> 5;
  const int nv = C / 256;
  for (int p = warp; p < TILE_H * TILE_W; p += NTHREADS / 32) {
    int ty = p / TILE_W, tx = p % TILE_W;
    float acc[MAXV][8];
#pragma unroll
    for (int v = 0; v < MAXV; ++v)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[v][e] = 0.f;
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
      for (int dx = 0; dx < 3; ++dx) {
        const __nv_bfloat16* tp = tile + (size_t)((ty + dy) * HALO_W + (tx + dx)) * C;
        const __nv_bfloat16* wp = dww + (size_t)(dy * 3 + dx) * C;
#pragma unroll
        for (int v = 0; v < MAXV; ++v) {
          if (v < nv) {
            int ch = v * 256 + lane * 8;
            float t[8], w[8];
            unpack8(*reinterpret_cast<const uint4*>(tp + ch), t);
            unpack8(__ldg(reinterpret_cast<const uint4*>(wp + ch)), w);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[v][e] += t[e] * w[e];
          }
        }
      }
    }
    // dw conv output is bf16 in the reference; round to mimic before LN
    float s = 0.f;
#pragma unroll
    for (int v = 0; v < MAXV; ++v)
      if (v < nv)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          acc[v][e] = __bfloat162float(__float2bfloat16_rn(acc[v][e]));
          s += acc[v][e];
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float mean = s / (float)C;
    float q = 0.f;
#pragma unroll
    for (int v = 0; v < MAXV; ++v)
      if (v < nv)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float d = acc[v][e] - mean;
          q += d * d;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    float rstd = rsqrtf(q / (float)C + eps);
    int Y = y0 + ty, X = x0 + tx;
    __nv_bfloat16* op = out + (((size_t)b * H + Y) * W + X) * C;
#pragma unroll
    for (int v = 0; v < MAXV; ++v) {
      if (v < nv) {
        int ch = v * 256 + lane * 8;
        const float4* gp = reinterpret_cast<const float4*>(gamma + ch);
        const float4* bp = reinterpret_cast<const float4*>(beta + ch);
        float4 g0 = gp[0], g1 = gp[1], e0 = bp[0], e1 = bp[1];
        float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        float bt[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
        __nv_bfloat162 o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o[k] = __floats2bfloat162_rn((acc[v][2 * k] - mean) * rstd * g[2 * k] + bt[2 * k],
                                       (acc[v][2 * k + 1] - mean) * rstd * g[2 * k + 1] + bt[2 * k + 1]);
        *reinterpret_cast<uint4*>(op + ch) = *reinterpret_cast<uint4*>(o);
      }
    }
  }
}

void dw_ln_launch(torch::Tensor G, torch::Tensor bias, torch::Tensor dww, torch::Tensor gamma,
                  torch::Tensor beta, torch::Tensor out, int64_t B, int64_t Hs, int64_t Ws,
                  double eps) {
  int C = (int)bias.size(0);
  TORCH_CHECK(C % 256 == 0 && C <= 256 * MAXV, "C must be a multiple of 256 and <= 1024");
  int H = 2 * (int)Hs, W = 2 * (int)Ws;
  TORCH_CHECK(H % TILE_H == 0 && W % TILE_W == 0, "tile mismatch");
  TORCH_CHECK(G.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "bf16");
  size_t smem = (size_t)HALO_H * HALO_W * C * sizeof(__nv_bfloat16);
  static bool attr_set = false;
  static size_t attr_smem = 0;
  if (!attr_set || attr_smem < smem) {
    cudaFuncSetAttribute(dw_ln_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
    attr_smem = smem;
  }
  dim3 grid(W / TILE_W, H / TILE_H, (unsigned)B);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dw_ln_kernel<<<grid, NTHREADS, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(G.data_ptr()), bias.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(dww.data_ptr()), gamma.data_ptr<float>(),
      beta.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)Hs, (int)Ws,
      C, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dw_ln_launch", &dw_ln_launch); }
