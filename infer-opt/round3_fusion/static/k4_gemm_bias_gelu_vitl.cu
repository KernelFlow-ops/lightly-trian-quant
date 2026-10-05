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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, const bfloat16_t* __restrict__ Bias, bfloat16_t* __restrict__ C, __grid_constant__ const CUtensorMap W_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, const bfloat16_t* __restrict__ Bias, bfloat16_t* __restrict__ C, __grid_constant__ const CUtensorMap W_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* w_s = ((void*)((char*)buf_dyn_shmem + 32768));
  __shared__ __align__(16) uint64_t mbarrier_mem[8];
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
    mbarrier[3].init(1);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k & 3) + 4)].wait((((k & 7) >> 2) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 3)].expect_transaction(8192);
        tl::tma_load(A_desc, mbarrier[(k & 3)], (&(((bfloat16_t*)a_s)[((k & 3) * 4096)])), (k * 32), (((int)blockIdx.y) * 128));
        mbarrier[(k & 3)].arrive_and_expect_tx(4096);
        tl::tma_load(W_desc, mbarrier[(k & 3)], (&(((bfloat16_t*)w_s)[((k & 3) * 2048)])), (k * 32), (((int)blockIdx.x) * 64));
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
      mbarrier[(k_1 & 3)].wait(((k_1 & 7) >> 2));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[8];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 4; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((((k_1 & 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_1 * 8)])));
          }
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_s)[(((((((k_1 & 3) * 2048) + ((((((int)threadIdx.x) >> 6) + 2) & 3) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[0])));
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + (i_2 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_2 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(c_l + ((i_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_2 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
          }
        }
      }
      mbarrier[((k_1 & 3) + 4)].arrive();
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      bfloat16_t Bias_local_cast[2];
      *(uint1*)(Bias_local_cast + 0) = *(uint1*)(Bias + (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_3 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 32));
      for (int vec = 0; vec < 2; ++vec) {
        c_l[((i_3 * 2) + vec)] = (((c_l[((i_3 * 2) + vec)] + ((float)Bias_local_cast[vec])) * 0x1p-1f/*5.000000e-01*/) * (0x1p+0f/*1.000000e+00*/ + erff(((c_l[((i_3 * 2) + vec)] + ((float)Bias_local_cast[vec])) * 0x1.6a09e667f3bcdp-1f/*7.071068e-01*/))));
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      bfloat16_t C_local_cast_1[2];
      if ((((((((int)blockIdx.y) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_4 >> 2) * 8)) + ((i_4 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1029) {
        uint1 __1;
        float2 v_ = *(float2*)(c_l + (i_4 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        *(uint1*)(C_local_cast_1 + 0) = __1;
        *(uint1*)(C + ((((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_4 >> 2) * 65536)) + ((i_4 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 32)) = *(uint1*)(C_local_cast_1 + 0);
      }
    }
  }
}

