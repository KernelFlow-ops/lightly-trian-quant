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

extern "C" __global__ void main_kernel(const float* __restrict__ COS, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Out_desc, const bfloat16_t* __restrict__ Q, const float* __restrict__ SIN, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(640, 1) main_kernel(const float* __restrict__ COS, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Out_desc, const bfloat16_t* __restrict__ Q, const float* __restrict__ SIN, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_raw = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_s = ((void*)((char*)buf_dyn_shmem + 32768));
  void* K_raw = ((void*)((char*)buf_dyn_shmem + 65536));
  void* V_s = ((void*)((char*)buf_dyn_shmem + 73728));
  void* K_s = ((void*)((char*)buf_dyn_shmem + 81920));
  __shared__ __align__(16) uint64_t mbarrier_mem[8];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc_o[32];
  float logsum[2];
  float m_cur[2];
  float acc_s[16];
  float m_prev[2];
  float m_cur_clear[2];
  float s_scale[2];
  float s_sum[2];
  bfloat16_t acc_c[16];
  if (tl::tl_shuffle_elect<0>()) {
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
  }
  tl::fence_barrier_init();
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 13; ++i) {
    if (((i * 5) + (((int)threadIdx.x) >> 7)) < 64) {
      bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint1 condval;
      if (((((((int)blockIdx.x) * 256) + (i * 20)) + (((int)threadIdx.x) >> 5)) < 1129)) {
        condval = *(uint1*)(Q + ((((((((int)blockIdx.z) * 3468288) + (((int)blockIdx.x) * 786432)) + (i * 61440)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 31) * 2)));
      } else {
        condval = make_uint1(__pack_nv_bfloat162(broadcast_var, broadcast_var));
      }
      *(uint1*)(((bfloat16_t*)Q_raw) + ((i * 1280) + (((int)threadIdx.x) * 2))) = condval;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 13; ++i_1) {
    if (((i_1 * 5) + (((int)threadIdx.x) >> 7)) < 64) {
      float c = COS[((min(max(((((((int)blockIdx.x) * 256) + (i_1 * 20)) + (((int)threadIdx.x) >> 5)) - 105), 0), 1023) * 64) + (((int)threadIdx.x) & 31))];
      float s = SIN[((min(max(((((((int)blockIdx.x) * 256) + (i_1 * 20)) + (((int)threadIdx.x) >> 5)) - 105), 0), 1023) * 64) + (((int)threadIdx.x) & 31))];
      float x1 = ((float)((bfloat16_t*)Q_raw)[(((i_1 * 1280) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))]);
      float x2 = ((float)((bfloat16_t*)Q_raw)[((((i_1 * 1280) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)]);
      float condval_1;
      if ((105 <= (((((int)blockIdx.x) * 256) + (i_1 * 20)) + (((int)threadIdx.x) >> 5)))) {
        condval_1 = ((x1 * c) - (x2 * s));
      } else {
        condval_1 = x1;
      }
      ((bfloat16_t*)Q_s)[((((((i_1 * 1280) + ((((int)threadIdx.x) >> 5) * 64)) + ((((i_1 * 5) + (((int)threadIdx.x) >> 7)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = ((bfloat16_t)condval_1);
      float condval_2;
      if ((105 <= (((((int)blockIdx.x) * 256) + (i_1 * 20)) + (((int)threadIdx.x) >> 5)))) {
        condval_2 = ((x2 * c) + (x1 * s));
      } else {
        condval_2 = x2;
      }
      ((bfloat16_t*)Q_s)[((((((i_1 * 1280) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((i_1 * 5) + (((int)threadIdx.x) >> 7)) & 1) + 1) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = ((bfloat16_t)condval_2);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::fence_proxy_async();
    for (int k = 0; k < 36; ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].arrive_and_expect_tx(4096);
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((bfloat16_t*)K_raw)[((k & 1) * 2048)])), 0, (k * 32), ((int)blockIdx.y), 1, ((int)blockIdx.z));
      }
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 2)].arrive_and_expect_tx(4096);
        tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((bfloat16_t*)V_s)[((k & 1) * 2048)])), 0, (k * 32), ((int)blockIdx.y), 2, ((int)blockIdx.z));
      }
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_2 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(m_cur + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    for (int k_1 = 0; k_1 < 36; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      float2 c_1 = *(float2*)(COS + ((min(max((((k_1 * 32) + (((int)threadIdx.x) >> 4)) - 113), 0), 1023) * 64) + ((((int)threadIdx.x) & 15) * 2)));
      float2 s_1 = *(float2*)(SIN + ((min(max((((k_1 * 32) + (((int)threadIdx.x) >> 4)) - 113), 0), 1023) * 64) + ((((int)threadIdx.x) & 15) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(((bfloat16_t*)K_raw) + (((((k_1 & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 15) * 2)) - 512));
      ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      float2 x1_1 = __1;
      float2 __2;
      uint1 v__1 = *(uint1*)(((bfloat16_t*)K_raw) + (((((k_1 & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 15) * 2)) - 480));
      ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
      float2 x2_1 = __2;
      tl::__sync_thread_partial(3, 512);
      uint1 __3;
      float2 condval_3;
      if ((113 <= ((k_1 * 32) + (((int)threadIdx.x) >> 4)))) {
        float2 __4;
          float2 __5;
            *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(x1_1.x)), *(float2*)(&(c_1.x)));
          float2 __6;
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(x2_1.x)), *(float2*)(&(s_1.x)));
          *(float2*)(&(__4.x)) = tl::sub2(*(float2*)(&(__5.x)), *(float2*)(&(__6.x)));
        condval_3 = __4;
      } else {
        condval_3 = x1_1;
      }
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&condval_3))[0]);
      *(uint1*)(((bfloat16_t*)K_s) + ((((((((k_1 & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((int)threadIdx.x) & 127) >> 6) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 512)) = __3;
      uint1 __7;
      float2 condval_4;
      if ((113 <= ((k_1 * 32) + (((int)threadIdx.x) >> 4)))) {
        float2 __8;
          float2 __9;
            *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(x1_1.x)), *(float2*)(&(s_1.x)));
          *(float2*)(&(__8.x)) = tl::fma2(*(float2*)(&(x2_1.x)), *(float2*)(&(c_1.x)), *(float2*)(&(__9.x)));
        condval_4 = __8;
      } else {
        condval_4 = x2_1;
      }
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&condval_4))[0]);
      *(uint1*)(((bfloat16_t*)K_s) + ((((((((k_1 & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + 1) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 512)) = __7;
      mbarrier[((k_1 & 1) + 4)].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + (i_3 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      }
      {
        bfloat16_t A_local[8];
        bfloat16_t B_local[16];
        tl::__sync_thread_partial(3, 512);
        for (int ki = 0; ki < 4; ++ki) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_s)[((((((((((int)threadIdx.x) >> 5) + 12) & 15) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
          #pragma unroll
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)K_s)[((((((((k_1 & 1) * 2048) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        float condval_5;
        if ((((((k_1 * 32) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < 1129)) {
          condval_5 = acc_s[i_5];
        } else {
          condval_5 = -CUDART_INF_F;
        }
        acc_s[i_5] = condval_5;
      }
      *(float2*)(m_prev + 0) = *(float2*)(m_cur + 0);
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        m_cur_clear[i_6] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          m_cur_clear[i_6] = max(m_cur_clear[i_6], acc_s[((((rv & 3) * 4) + (i_6 * 2)) + (rv >> 2))]);
        }
        m_cur_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<512>>::run(m_cur_clear[i_6]);
        m_cur[i_6] = max(m_cur[i_6], m_cur_clear[i_6]);
      }
      float broadcast_var_5 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float broadcast_var_6 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __10;
      float2 __11;
        float2 __12;
          float2 v__2 = *(float2*)(m_prev + 0);
          float2 v__3 = make_float2(broadcast_var_5, broadcast_var_5);
          *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        float2 __13;
          float2 v__4 = *(float2*)(m_cur + 0);
          float2 v__5 = make_float2(broadcast_var_6, broadcast_var_6);
          *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
        *(float2*)(&(__11.x)) = tl::sub2(*(float2*)(&(__12.x)), *(float2*)(&(__13.x)));
      __10.x = exp2f(__11.x);
      __10.y = exp2f(__11.y);
      *(float2*)(s_scale + 0) = __10;
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        float broadcast_var_7 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __14;
        float2 __15;
          float2 __16;
            float2 v__6 = *(float2*)(acc_s + (i_7 * 2));
            float2 v__7 = make_float2(broadcast_var_7, broadcast_var_7);
            *(float2*)(&(__16.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
          float2 v__8 = make_float2((m_cur[(i_7 & 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/), (m_cur[(i_7 & 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/));
          *(float2*)(&(__15.x)) = tl::sub2(*(float2*)(&(__16.x)), *(float2*)(&(v__8.x)));
        __14.x = exp2f(__15.x);
        __14.y = exp2f(__15.y);
        *(float2*)(acc_s + (i_7 * 2)) = __14;
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        s_sum[i_8] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          s_sum[i_8] = (s_sum[i_8] + acc_s[((((rv_1 & 3) * 4) + (i_8 * 2)) + (rv_1 >> 2))]);
        }
        s_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<512>>::run(s_sum[i_8]);
      }
      float2 __17;
        float2 v__9 = *(float2*)(logsum + 0);
        float2 v__10 = *(float2*)(s_scale + 0);
        float2 v__11 = *(float2*)(s_sum + 0);
        *(float2*)(&(__17.x)) = tl::fma2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
      *(float2*)(logsum + 0) = __17;
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        uint2 __18;
        float4 v__12 = *(float4*)(acc_s + (i_9 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__18))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__18))[1] = __float22bfloat162_rn(((float2*)(&v__12))[1]);
        *(uint2*)(acc_c + (i_9 * 4)) = __18;
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        float2 __19;
          float2 v__13 = *(float2*)(acc_o + (i_10 * 2));
          float2 v__14 = make_float2(s_scale[(i_10 & 1)], s_scale[(i_10 & 1)]);
          *(float2*)(&(__19.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
        *(float2*)(acc_o + (i_10 * 2)) = __19;
      }
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      {
        bfloat16_t B_local_1[32];
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          #pragma unroll
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_s)[(((((((k_1 & 1) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_11 * 8)])));
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
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      float2 __20;
        float2 v__15 = *(float2*)(acc_o + (i_12 * 2));
        float2 v__16 = make_float2(logsum[(i_12 & 1)], logsum[(i_12 & 1)]);
        __20.x = (v__15.x/v__16.x);
        __20.y = (v__15.y/v__16.y);
      *(float2*)(acc_o + (i_12 * 2)) = __20;
    }
    tl::__sync_thread_partial(3, 512);
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)O_s)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)acc_o[(i_13 * 8)]), ((bfloat16_t)acc_o[((i_13 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_13 * 8) + 2)]), ((bfloat16_t)acc_o[((i_13 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_13 * 8) + 4)]), ((bfloat16_t)acc_o[((i_13 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_13 * 8) + 6)]), ((bfloat16_t)acc_o[((i_13 * 8) + 7)])));
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

