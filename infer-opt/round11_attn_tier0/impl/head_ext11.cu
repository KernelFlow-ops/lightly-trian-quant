// Round 11: faster fused ScaleBlock tail  (copy-and-modify of round7_head_postprocess/impl/head_ext.cu :: dw_ln_kernel).
//   ConvTranspose2d(k=2,s=2) is a plain bf16 GEMM  X[B*Hs*Ws, C] @ Wr[C, 4C]  (column = (ki*2+kj)*C + co, "pre-shuffle").
//   This kernel fuses  + conv bias, exact-erf GELU, pixel-shuffle read, depthwise 3x3 (zero pad), LayerNorm over C -> NHWC bf16.
// Differences to round 7 (same numerics: GELU in fp32 -> bf16, dwconv fp32 accumulate in the same tap order -> bf16, LN fp32):
//   * channels are processed in passes of 256 (CP): the halo tile in smem holds only CP channels -> 8 KB * (TH+2)(TW+2)/8 smem per block
//     instead of all C, so 2-3 blocks/SM overlap each other's load phase and compute phase (round 7: 1 block/SM, 80 KB).
//   * each warp computes PPW adjacent output pixels per pass (sliding window): the 9 taps of PPW pixels need (PPW+2)*3 tile vectors
//     instead of 9*PPW and the per-channel weights are unpacked once per PPW pixels.
//   * dwconv results are kept as packed bf16x2 in registers (they are rounded to bf16 before LN anyway) until the LN statistics are known.
//   * halo loads of 4 vectors are batched before the GELU math (memory-level parallelism).
//   * optional FAST_GELU: erf from one ex2 + one rcp (A&S 7.1.26, |err| < 1.5e-7), instead of erff().
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef TH
#define TH 4
#endif
#ifndef TW
#define TW 8
#endif
#ifndef PPW
#define PPW 4
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef FAST_GELU
#define FAST_GELU 0
#endif
#define CP 256                       // channels per pass
#define VPC (CP / 8)                 // 16B vectors per halo pixel per pass (= 32 = warp size)
#define HALO_H (TH + 2)
#define HALO_W (TW + 2)
#define HALO_PIX (HALO_H * HALO_W)
#define NWARP (TH * TW / PPW)
#define NT (NWARP * 32)
#define MAXV 4                       // C <= 1024
static_assert(TW % PPW == 0, "TW % PPW");
static_assert(VPC == 32, "one warp spans one pixel's 256 channels");

__device__ __forceinline__ float gelu_exact(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}
// erf(z) for z >= 0 via A&S 7.1.26 (max abs err 1.5e-7); ex2.approx / rcp.approx
__device__ __forceinline__ float erf_pos(float z) {
  float t = __frcp_rn(fmaf(0.3275911f, z, 1.0f));
  float p = fmaf(fmaf(fmaf(fmaf(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f) * t;
  float e = exp2f(-1.4426950408889634f * z * z);
  return fmaf(-p, e, 1.0f);
}
__device__ __forceinline__ float gelu_fast(float x) {
  float z = fabsf(x) * 0.70710678118654752440f;
  float er = erf_pos(z);
  er = copysignf(er, x);
  return 0.5f * x * (1.0f + er);
}
__device__ __forceinline__ float gelu(float x) { return FAST_GELU ? gelu_fast(x) : gelu_exact(x); }

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__global__ void __launch_bounds__(NT, MINB)
dw_ln2_kernel(const __nv_bfloat16* __restrict__ G,     // [B*Hs*Ws, 4C]
              const float* __restrict__ bias,          // [C]
              const __nv_bfloat16* __restrict__ dww,   // [9, C]
              const float* __restrict__ gamma, const float* __restrict__ beta,  // [C]
              __nv_bfloat16* __restrict__ out,         // [B, H, W, C]
              int Hs, int Ws, int C, float eps) {
  extern __shared__ __align__(16) uint4 tile[];        // [HALO_PIX][VPC]
  const int H = 2 * Hs, W = 2 * Ws;
  const int b = blockIdx.z;
  const int y0 = blockIdx.y * TH, x0 = blockIdx.x * TW;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nv = C / CP;
  const int ty = warp / (TW / PPW), tx0 = (warp % (TW / PPW)) * PPW;

  uint32_t pk[MAXV][PPW][4];                           // dwconv results, bf16x2 packed

#pragma unroll
  for (int p = 0; p < MAXV; ++p) {
    if (p < nv) {
      __syncthreads();                                 // previous pass has finished reading the tile
      // ---- phase 1: halo load (+bias, GELU) -> smem, 4 vectors in flight per thread
      constexpr int TOTAL = HALO_PIX * VPC;
      for (int i0 = tid; i0 < TOTAL; i0 += NT * 4) {
        uint4 raw[4];
        bool ok[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          int i = i0 + u * NT;
          ok[u] = false;
          raw[u] = make_uint4(0, 0, 0, 0);
          if (i < TOTAL) {
            int pix = i >> 5;
            int hy = pix / HALO_W, hx = pix - hy * HALO_W;
            int Y = y0 - 1 + hy, X = x0 - 1 + hx;
            if (Y >= 0 && Y < H && X >= 0 && X < W) {
              size_t row = ((size_t)b * Hs + (Y >> 1)) * Ws + (X >> 1);
              int sub = ((Y & 1) << 1) | (X & 1);
              raw[u] = __ldg(reinterpret_cast<const uint4*>(G + row * 4 * C + (size_t)sub * C + p * CP + (i & 31) * 8));
              ok[u] = true;
            }
          }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          int i = i0 + u * NT;
          if (i < TOTAL) {
            uint4 outv = make_uint4(0, 0, 0, 0);
            if (ok[u]) {
              float f[8];
              unpack8(raw[u], f);
              const float4* bp = reinterpret_cast<const float4*>(bias + p * CP + (i & 31) * 8);
              float4 b0 = __ldg(bp), b1 = __ldg(bp + 1);
              float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
              __nv_bfloat162 o[4];
#pragma unroll
              for (int k = 0; k < 4; ++k)
                o[k] = __floats2bfloat162_rn(gelu(f[2 * k] + bb[2 * k]), gelu(f[2 * k + 1] + bb[2 * k + 1]));
              outv = *reinterpret_cast<uint4*>(o);
            }
            tile[i] = outv;
          }
        }
      }
      __syncthreads();
      // ---- phase 2: depthwise 3x3 for PPW adjacent pixels, taps in the same (dy, dx) order as round 7
      float acc[PPW][8];
#pragma unroll
      for (int j = 0; j < PPW; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.f;
#pragma unroll
      for (int dy = 0; dy < 3; ++dy) {
        float t[PPW + 2][8];
#pragma unroll
        for (int i = 0; i < PPW + 2; ++i) unpack8(tile[((ty + dy) * HALO_W + tx0 + i) * VPC + lane], t[i]);
#pragma unroll
        for (int dx = 0; dx < 3; ++dx) {
          float w[8];
          unpack8(__ldg(reinterpret_cast<const uint4*>(dww + (size_t)(dy * 3 + dx) * C + p * CP + lane * 8)), w);
#pragma unroll
          for (int j = 0; j < PPW; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j][e] = fmaf(t[j + dx][e], w[e], acc[j][e]);
        }
      }
#pragma unroll
      for (int j = 0; j < PPW; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          __nv_bfloat162 h = __floats2bfloat162_rn(acc[j][2 * k], acc[j][2 * k + 1]);   // dwconv output is bf16 in the reference
          pk[p][j][k] = *reinterpret_cast<uint32_t*>(&h);
        }
    }
  }

  // ---- LayerNorm over C per pixel (fp32 statistics from the bf16-rounded dwconv outputs, same order as round 7)
#pragma unroll
  for (int j = 0; j < PPW; ++j) {
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < MAXV; ++p)
      if (p < nv)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&pk[p][j][k]));
          s += a.x;
          s += a.y;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / (float)C;
    float q = 0.f;
#pragma unroll
    for (int p = 0; p < MAXV; ++p)
      if (p < nv)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&pk[p][j][k]));
          float d0 = a.x - mean, d1 = a.y - mean;
          q += d0 * d0;
          q += d1 * d1;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q / (float)C + eps);
    const int Y = y0 + ty, X = x0 + tx0 + j;
    __nv_bfloat16* op = out + (((size_t)b * H + Y) * W + X) * C;
#pragma unroll
    for (int p = 0; p < MAXV; ++p) {
      if (p < nv) {
        const int ch = p * CP + lane * 8;
        const float4* gp = reinterpret_cast<const float4*>(gamma + ch);
        const float4* bp = reinterpret_cast<const float4*>(beta + ch);
        float4 g0 = __ldg(gp), g1 = __ldg(gp + 1), e0 = __ldg(bp), e1 = __ldg(bp + 1);
        float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        float bt[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
        __nv_bfloat162 o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&pk[p][j][k]));
          o[k] = __floats2bfloat162_rn((a.x - mean) * rstd * g[2 * k] + bt[2 * k],
                                       (a.y - mean) * rstd * g[2 * k + 1] + bt[2 * k + 1]);
        }
        *reinterpret_cast<uint4*>(op + ch) = *reinterpret_cast<uint4*>(o);
      }
    }
  }
}

void dw_ln_launch(torch::Tensor G, torch::Tensor bias, torch::Tensor dww, torch::Tensor gamma,
                  torch::Tensor beta, torch::Tensor out, int64_t B, int64_t Hs, int64_t Ws, double eps) {
  int C = (int)bias.size(0);
  TORCH_CHECK(C % CP == 0 && C <= CP * MAXV, "C must be a multiple of 256 and <= 1024");
  int H = 2 * (int)Hs, W = 2 * (int)Ws;
  TORCH_CHECK(H % TH == 0 && W % TW == 0, "tile mismatch");
  TORCH_CHECK(G.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "bf16");
  size_t smem = (size_t)HALO_PIX * VPC * 16;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(dw_ln2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  dim3 grid(W / TW, H / TH, (unsigned)B);
  dw_ln2_kernel<<<grid, NT, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(G.data_ptr()), bias.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(dww.data_ptr()), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)Hs, (int)Ws, C, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ------------------------------------------------------------------------------------------------------------------
// Round 11 tail: bb.norm (LayerNorm over C, fp32) + bf16 cast + gather of the rows the head needs + class_head, in one kernel.
//   x   : fp32 [B, ntok, C]  final residual stream (ntok = Q + npre + patches)
//   q rows (i < Q)               -> qout [B*Q, C] bf16, and class logits clog [B, Q, 3] bf16 = bf16(LN(x)_bf16 . Wcls^T + bcls)
//   cls/register rows            -> skipped (never used by the head)
//   patch rows (i >= Q + npre)   -> xout [B*npatch, C] bf16 (contiguous: the ConvT GEMM A operand)
// One warp per row; lane owns float4 chunks j*32 + lane.
#define TAIL_MAXJ 8   // C <= 1024
__global__ void __launch_bounds__(256)
tail_ln_kernel(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
               __nv_bfloat16* __restrict__ qout, __nv_bfloat16* __restrict__ xout,
               const __nv_bfloat16* __restrict__ wcls, const float* __restrict__ bcls, __nv_bfloat16* __restrict__ clog,
               int nrows, int ntok, int Q, int npre, int C, float eps) {
  const int warp = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (warp >= nrows) return;
  const int b = warp / ntok, i = warp - b * ntok;
  if (i >= Q && i < Q + npre) return;
  const int nj = C / 128;
  const float4* rp = reinterpret_cast<const float4*>(x + (size_t)warp * C);
  float4 v[TAIL_MAXJ];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < TAIL_MAXJ; ++j)
    if (j < nj) { v[j] = rp[j * 32 + lane]; s += (v[j].x + v[j].y) + (v[j].z + v[j].w); }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / (float)C;
  float qv = 0.f;
#pragma unroll
  for (int j = 0; j < TAIL_MAXJ; ++j)
    if (j < nj) {
      float a0 = v[j].x - mean, a1 = v[j].y - mean, a2 = v[j].z - mean, a3 = v[j].w - mean;
      qv += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
    }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) qv += __shfl_xor_sync(0xffffffffu, qv, o);
  const float rstd = rsqrtf(qv / (float)C + eps);
  __nv_bfloat16* op = i < Q ? qout + ((size_t)b * Q + i) * C : xout + ((size_t)b * (ntok - Q - npre) + (i - Q - npre)) * C;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f;
#pragma unroll
  for (int j = 0; j < TAIL_MAXJ; ++j)
    if (j < nj) {
      const int ch = (j * 32 + lane) * 4;
      float4 g = __ldg(reinterpret_cast<const float4*>(gamma + ch)), bt = __ldg(reinterpret_cast<const float4*>(beta + ch));
      __nv_bfloat162 o0 = __floats2bfloat162_rn((v[j].x - mean) * rstd * g.x + bt.x, (v[j].y - mean) * rstd * g.y + bt.y);
      __nv_bfloat162 o1 = __floats2bfloat162_rn((v[j].z - mean) * rstd * g.z + bt.z, (v[j].w - mean) * rstd * g.w + bt.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&o0);
      pk.y = *reinterpret_cast<uint32_t*>(&o1);
      *reinterpret_cast<uint2*>(op + ch) = pk;
      if (i < Q) {
        float2 y0 = __bfloat1622float2(o0), y1 = __bfloat1622float2(o1);
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          uint2 w2 = __ldg(reinterpret_cast<const uint2*>(wcls + (size_t)k * C + ch));
          float2 w0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&w2.x));
          float2 w1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&w2.y));
          float d = y0.x * w0.x + y0.y * w0.y + y1.x * w1.x + y1.y * w1.y;
          if (k == 0) c0 += d; else if (k == 1) c1 += d; else c2 += d;
        }
      }
    }
  if (i < Q) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      c0 += __shfl_xor_sync(0xffffffffu, c0, o);
      c1 += __shfl_xor_sync(0xffffffffu, c1, o);
      c2 += __shfl_xor_sync(0xffffffffu, c2, o);
    }
    if (lane == 0) {
      __nv_bfloat16* cp = clog + ((size_t)b * Q + i) * 3;
      cp[0] = __float2bfloat16_rn(c0 + bcls[0]);
      cp[1] = __float2bfloat16_rn(c1 + bcls[1]);
      cp[2] = __float2bfloat16_rn(c2 + bcls[2]);
    }
  }
}

void tail_ln_launch(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor qout, torch::Tensor xout,
                    torch::Tensor wcls, torch::Tensor bcls, torch::Tensor clog, int64_t Q, int64_t npre, double eps) {
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.dim() == 3, "x fp32 [B,ntok,C]");
  int B = (int)x.size(0), ntok = (int)x.size(1), C = (int)x.size(2);
  TORCH_CHECK(C % 128 == 0 && C <= 128 * TAIL_MAXJ, "C");
  int nrows = B * ntok;
  tail_ln_kernel<<<(nrows + 7) / 8, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(qout.data_ptr()), reinterpret_cast<__nv_bfloat16*>(xout.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(wcls.data_ptr()), bcls.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(clog.data_ptr()), nrows, ntok, (int)Q, (int)npre, C, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dw_ln_launch", &dw_ln_launch);
  m.def("tail_ln_launch", &tail_ln_launch);
}

