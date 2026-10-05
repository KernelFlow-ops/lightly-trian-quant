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
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Out_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(640, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Out_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_s = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_s = ((void*)((char*)buf_dyn_shmem + 49152));
  __shared__ __align__(16) uint64_t mbarrier_mem[9];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc_o[32];
  float logsum[2];
  float m_cur[2];
  float acc_s[32];
  float m_prev[2];
  float m_cur_clear[2];
  float s_scale[2];
  float s_sum[2];
  bfloat16_t acc_c[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Out_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(512);
    mbarrier[5].init(512);
    mbarrier[6].init(512);
    mbarrier[7].init(512);
    mbarrier[8].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<640>()) {
    mbarrier[8].arrive_and_expect_tx(32768);
    tl::tma_load(Q_desc, mbarrier[8], (&(((bfloat16_t*)Q_s)[0])), 0, (((int)blockIdx.x) * 256), ((int)blockIdx.y), 0, ((int)blockIdx.z));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    for (int k = 0; k < 18; ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((bfloat16_t*)K_s)[((k & 1) * 4096)])), 0, (k * 64), ((int)blockIdx.y), 1, ((int)blockIdx.z));
      }
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 2)].arrive_and_expect_tx(8192);
        tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((bfloat16_t*)V_s)[((k & 1) * 4096)])), 0, (k * 64), ((int)blockIdx.y), 2, ((int)blockIdx.z));
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(m_cur + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    for (int k_1 = 0; k_1 < 18; ++k_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + (i_1 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      }
      if (k_1 == 0) {
        mbarrier[8].wait(0);
      }
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        bfloat16_t A_local[8];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_s)[((((((((((int)threadIdx.x) >> 5) + 12) & 15) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
          #pragma unroll
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)K_s)[((((((((k_1 & 1) * 4096) + (i_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
          }
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      mbarrier[((k_1 & 1) + 4)].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        float condval;
        if ((((((k_1 * 64) + ((i_3 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1)) < 1129)) {
          condval = acc_s[i_3];
        } else {
          condval = -CUDART_INF_F;
        }
        acc_s[i_3] = condval;
      }
      *(float2*)(m_prev + 0) = *(float2*)(m_cur + 0);
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        m_cur_clear[i_4] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          m_cur_clear[i_4] = max(m_cur_clear[i_4], acc_s[((((rv & 7) * 4) + (i_4 * 2)) + (rv >> 3))]);
        }
        m_cur_clear[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<512>>::run(m_cur_clear[i_4]);
        m_cur[i_4] = max(m_cur[i_4], m_cur_clear[i_4]);
      }
      float broadcast_var_4 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float broadcast_var_5 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(m_prev + 0);
          float2 v__1 = make_float2(broadcast_var_4, broadcast_var_4);
          *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
        float2 __4;
          float2 v__2 = *(float2*)(m_cur + 0);
          float2 v__3 = make_float2(broadcast_var_5, broadcast_var_5);
          *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(s_scale + 0) = __1;
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        float broadcast_var_6 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __5;
        float2 __6;
          float2 __7;
            float2 v__4 = *(float2*)(acc_s + (i_5 * 2));
            float2 v__5 = make_float2(broadcast_var_6, broadcast_var_6);
            *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
          float2 v__6 = make_float2((m_cur[(i_5 & 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/), (m_cur[(i_5 & 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/));
          *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(__7.x)), *(float2*)(&(v__6.x)));
        __5.x = exp2f(__6.x);
        __5.y = exp2f(__6.y);
        *(float2*)(acc_s + (i_5 * 2)) = __5;
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        s_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          s_sum[i_6] = (s_sum[i_6] + acc_s[((((rv_1 & 7) * 4) + (i_6 * 2)) + (rv_1 >> 3))]);
        }
        s_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<512>>::run(s_sum[i_6]);
      }
      float2 __8;
        float2 v__7 = *(float2*)(logsum + 0);
        float2 v__8 = *(float2*)(s_scale + 0);
        float2 v__9 = *(float2*)(s_sum + 0);
        *(float2*)(&(__8.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
      *(float2*)(logsum + 0) = __8;
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        uint2 __9;
        float4 v__10 = *(float4*)(acc_s + (i_7 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__9))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
        *(uint2*)(acc_c + (i_7 * 4)) = __9;
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float2 __10;
          float2 v__11 = *(float2*)(acc_o + (i_8 * 2));
          float2 v__12 = make_float2(s_scale[(i_8 & 1)], s_scale[(i_8 & 1)]);
          *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
        *(float2*)(acc_o + (i_8 * 2)) = __10;
      }
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      {
        bfloat16_t B_local_1[32];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          #pragma unroll
          for (int i_9 = 0; i_9 < 4; ++i_9) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_s)[(((((((k_1 & 1) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_9 * 8)])));
          }
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_c + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_c + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
      mbarrier[((k_1 & 1) + 6)].arrive();
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      float2 __11;
        float2 v__13 = *(float2*)(acc_o + (i_10 * 2));
        float2 v__14 = make_float2(logsum[(i_10 & 1)], logsum[(i_10 & 1)]);
        __11.x = (v__13.x/v__14.x);
        __11.y = (v__13.y/v__14.y);
      *(float2*)(acc_o + (i_10 * 2)) = __11;
    }
    tl::__sync_thread_partial(3, 512);
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)O_s)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)acc_o[(i_11 * 8)]), ((bfloat16_t)acc_o[((i_11 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_11 * 8) + 2)]), ((bfloat16_t)acc_o[((i_11 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_11 * 8) + 4)]), ((bfloat16_t)acc_o[((i_11 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_11 * 8) + 6)]), ((bfloat16_t)acc_o[((i_11 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 512);
    if (tl::tl_shuffle_elect<512>()) {
      tl::fence_proxy_async();
      tl::tma_store(Out_desc, (&(((bfloat16_t*)O_s)[0])), 0, (((int)blockIdx.x) * 256), ((int)blockIdx.y), ((int)blockIdx.z));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

