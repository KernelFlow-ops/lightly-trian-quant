#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ Bias, const float* __restrict__ Cos, bfloat16_t* __restrict__ Out, const float* __restrict__ Sin, const bfloat16_t* __restrict__ W);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ Bias, const float* __restrict__ Cos, bfloat16_t* __restrict__ Out, const float* __restrict__ Sin, const bfloat16_t* __restrict__ W) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* w_s = ((void*)((char*)buf_dyn_shmem + 24576));
  float c_l[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(c_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a_s)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8))])), (((((((int)blockIdx.y) * 64) + (i_1 * 16)) + (((int)threadIdx.x) >> 3)) < 1029) && ((((((int)blockIdx.y) * 64) + (i_1 * 16)) + (((int)threadIdx.x) >> 3)) < 1029)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)w_s)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((int)blockIdx.x) * 65536) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a_s)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])), (((((((int)blockIdx.y) * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 3)) < 1029) && ((((((int)blockIdx.y) * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 3)) < 1029)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)w_s)[(((((i_4 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(W[(((((((int)blockIdx.x) * 65536) + (i_4 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 30; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a_s)[(((((((k + 2) % 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 131072) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])), (((((((int)blockIdx.y) * 64) + (i_5 * 16)) + (((int)threadIdx.x) >> 3)) < 1029) && ((((((int)blockIdx.y) * 64) + (i_5 * 16)) + (((int)threadIdx.x) >> 3)) < 1029)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)w_s)[(((((((k + 2) % 3) * 2048) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((((int)blockIdx.x) * 65536) + (i_6 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      bfloat16_t A_local[32];
      bfloat16_t B_local[16];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((((k % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_7 * 8)])));
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_s)[((((((((k % 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (i_8 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_8 * 8)])));
        }
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + ((i_9 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + (((i_9 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    bfloat16_t A_local_1[32];
    bfloat16_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[(i_10 * 8)])));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_s)[(((((((((int)threadIdx.x) >> 6) * 1024) + (i_11 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_11 * 8)])));
      }
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + ((i_12 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + (((i_12 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    bfloat16_t A_local_2[32];
    bfloat16_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_2[(i_13 * 8)])));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_s)[((((((((((int)threadIdx.x) >> 6) * 1024) + (i_14 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), (&(B_local_2[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + ((i_15 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + (((i_15 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 32; ++i_16) {
    bfloat16_t Bias_local_cast[2];
    *(uint1*)(Bias_local_cast + 0) = *(uint1*)(Bias + ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_16 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __1;
      float2 v_ = *(float2*)(c_l + (i_16 * 2));
      float2 __2;
      uint1 v__1 = *(uint1*)(Bias_local_cast + 0);
      ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
      *(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(__2.x)));
    *(float2*)(c_l + (i_16 * 2)) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 32; ++i_17) {
    *(float2*)(((float*)c_s) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_17 >> 3) * 1024)) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_17 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(c_l + (i_17 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 64; ++i_18) {
    float x = ((float*)c_s)[((i_18 * 128) + ((int)threadIdx.x))];
    float xp = ((float*)c_s)[(((i_18 * 128) + ((((int)threadIdx.x) >> 6) * 64)) + ((((int)threadIdx.x) & 63) ^ 32))];
    float condval;
    if (((((int)threadIdx.x) & 63) < 32)) {
      condval = -0x1p+0f/*-1.000000e+00*/;
    } else {
      condval = 0x1p+0f/*1.000000e+00*/;
    }
    float sgn = condval;
    bool do_rope = ((((int)blockIdx.x) < 32) && (5 <= ((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029)));
    float condval_1;
    if (((((int)threadIdx.x) & 63) < 32)) {
      condval_1 = -0x1p+0f/*-1.000000e+00*/;
    } else {
      condval_1 = 0x1p+0f/*1.000000e+00*/;
    }
    float r = ((x * Cos[((max((((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029) - 5), 0) * 64) + (((int)threadIdx.x) & 63))]) + ((condval_1 * xp) * Sin[((max((((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029) - 5), 0) * 64) + (((int)threadIdx.x) & 63))]));
    float condval_2;
    if (((((int)blockIdx.x) < 32) && (5 <= ((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029)))) {
      condval_2 = r;
    } else {
      condval_2 = x;
    }
    float val = condval_2;
    if (((((int)blockIdx.y) * 64) + i_18) < 1029) {
      float condval_3;
      if (((((int)blockIdx.x) < 32) && (5 <= ((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029)))) {
        condval_3 = r;
      } else {
        condval_3 = x;
      }
      Out[((((((((int)blockIdx.x) >> 4) * 2107392) + (((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) / 1029) * 1053696)) + ((((int)blockIdx.x) & 15) * 65856)) + (((((((int)blockIdx.y) * 128) + (i_18 * 2)) + (((int)threadIdx.x) >> 6)) % 1029) * 64)) + (((int)threadIdx.x) & 63))] = ((bfloat16_t)condval_3);
    }
  }
}

