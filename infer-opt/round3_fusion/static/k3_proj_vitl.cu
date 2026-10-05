#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, const bfloat16_t* __restrict__ Bias, float* __restrict__ C, const float* __restrict__ Gamma, const float* __restrict__ R, __grid_constant__ const CUtensorMap W_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, const bfloat16_t* __restrict__ Bias, float* __restrict__ C, const float* __restrict__ Gamma, const float* __restrict__ R, __grid_constant__ const CUtensorMap W_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* w_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_s = ((void*)((char*)buf_dyn_shmem + 24576));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float c_l[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(W_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(256);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(4096);
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)a_s)[((k % 3) * 2048)])), (k * 32), (((int)blockIdx.y) * 64));
        mbarrier[(k % 3)].arrive_and_expect_tx(8192);
        tl::tma_load(W_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)w_s)[((k % 3) * 4096)])), (k * 32), (((int)blockIdx.x) * 128));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(c_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        bfloat16_t A_local[16];
        bfloat16_t B_local[16];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 2; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((((k_1 % 3) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_1 * 8)])));
          }
          #pragma unroll
          for (int i_2 = 0; i_2 < 2; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_s)[((((((((k_1 % 3) * 4096) + ((((((int)threadIdx.x) >> 6) + 2) & 3) * 1024)) + (i_2 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
          }
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + ((i_3 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + (((i_3 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      bfloat16_t Bias_local_cast[2];
      float Gamma_local_cast_1[2];
      float R_local_cast_2[2];
      *(uint1*)(Bias_local_cast + 0) = *(uint1*)(Bias + (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_4 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      *(float2*)(Gamma_local_cast_1 + 0) = *(float2*)(Gamma + (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_4 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      float2 condval;
      if (((((((((int)blockIdx.y) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_4 >> 3) * 8)) + ((i_4 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1029)) {
        condval = *(float2*)(R + ((((((((((((int)blockIdx.y) * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 32768)) + ((i_4 >> 3) * 16384)) + ((i_4 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_4 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      } else {
        condval = make_float2(broadcast_var_1, broadcast_var_1);
      }
      *(float2*)(R_local_cast_2 + 0) = condval;
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      float2 __1;
        float2 __2;
          float2 v_ = *(float2*)(c_l + (i_4 * 2));
          float2 __3;
          uint1 v__1 = *(uint1*)(Bias_local_cast + 0);
          ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
          *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(__3.x)));
        float2 v__2 = *(float2*)(Gamma_local_cast_1 + 0);
        float2 condval_1;
        if (((((((((int)blockIdx.y) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_4 >> 3) * 8)) + ((i_4 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1029)) {
          condval_1 = *(float2*)(R_local_cast_2 + 0);
        } else {
          condval_1 = make_float2(broadcast_var_2, broadcast_var_2);
        }
        *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(__2.x)), *(float2*)(&(v__2.x)), *(float2*)(&(condval_1.x)));
      *(float2*)(c_l + (i_4 * 2)) = __1;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      if ((((((((int)blockIdx.y) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_5 >> 3) * 8)) + ((i_5 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1029) {
        *(float2*)(C + ((((((((((((int)blockIdx.y) * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 32768)) + ((i_5 >> 3) * 16384)) + ((i_5 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_5 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(c_l + (i_5 * 2));
      }
    }
  }
}

