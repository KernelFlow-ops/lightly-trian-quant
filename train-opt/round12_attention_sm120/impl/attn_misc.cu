// Round12 mask builder: mask logits fp32 [B,Q,128,128] -> QueryBitmask words u32 [B,Q,32].
//
// Source semantics (round1/round4, the EoMT ``_disable_attn_mask`` path):
//   interp = F.interpolate(mask_logits, (32, 32), mode="bilinear")   # align_corners=False, 128 -> 32 is a 2x2 average
//   allow  = interp > 0                                                # patch p = 32*y + x
//   rows with rand[b, q] > prob are fully enabled ("disabled" mask rows)
// For scale 4 the bilinear source coordinate is 4*d + 1.5, i.e. pixels 4d+1 and 4d+2 with weight 1/2 each.  ATen computes
// h0*(w0*x00 + w1*x01) + h1*(w0*x10 + w1*x11) with all weights 0.5 (exact scalings), so the only roundings are the three adds:
// reproduced below with plain fp32 ops (never contracted).  One warp builds one 32-bit word (grid row y of one query): lane x
// loads the aligned float4 at columns 4x..4x+3 of image rows 4y+1 and 4y+2 (coalesced) and a ballot assembles the word.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "attn_common.cuh"
#include "attn_api.h"

namespace a12 {

namespace {

template <typename T>
__device__ __forceinline__ void load4(const T* p, float (&v)[4]);
template <>
__device__ __forceinline__ void load4<float>(const float* p, float (&v)[4]) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
}
template <>
__device__ __forceinline__ void load4<bf16>(const bf16* p, float (&v)[4]) {
    const uint2 t = *reinterpret_cast<const uint2*>(p);
    v[0] = bf16lo(t.x); v[1] = bf16hi(t.x); v[2] = bf16lo(t.y); v[3] = bf16hi(t.y);
}

template <typename T>
__global__ void build_bits_kernel(const T* __restrict__ logits, const float* __restrict__ rnd, const float* __restrict__ prob, unsigned* __restrict__ bits,
                                  int rows_total /* B*Q */, int Q) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // one warp per (b*Q + q, y)
    const int lane = threadIdx.x & 31;
    if (warp >= rows_total * 32) return;
    const int bq = warp >> 5, y = warp & 31;
    const T* base = logits + (size_t)bq * 128 * 128;
    float v0[4], v1[4];
    load4<T>(base + (size_t)(4 * y + 1) * 128 + 4 * lane, v0);
    load4<T>(base + (size_t)(4 * y + 2) * 128 + 4 * lane, v1);
    const float h1 = __fadd_rn(__fmul_rn(v0[1], 0.5f), __fmul_rn(v0[2], 0.5f));
    const float h2 = __fadd_rn(__fmul_rn(v1[1], 0.5f), __fmul_rn(v1[2], 0.5f));
    const float v = __fadd_rn(__fmul_rn(h1, 0.5f), __fmul_rn(h2, 0.5f));
    unsigned word = __ballot_sync(0xffffffffu, v > 0.f);
    if (rnd != nullptr && rnd[bq] > prob[0]) word = 0xffffffffu;  // row disabled -> every patch visible
    if (lane == 0) bits[(size_t)bq * 32 + y] = word;
}

}  // namespace

// logits [B,Q,128,128] fp32 contiguous; rnd [B,Q] fp32 or None; prob fp32 device scalar (required with rnd).
Tensor build_bits(const Tensor& logits, const std::optional<Tensor>& rnd, const std::optional<Tensor>& prob) {
    TORCH_CHECK(logits.is_cuda() && (logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16) && logits.is_contiguous() && logits.dim() == 4 &&
                    logits.size(2) == 128 && logits.size(3) == 128,
                "mask logits must be contiguous fp32/bf16 [B,Q,128,128]");
    const c10::cuda::CUDAGuard guard(logits.device());
    const int B = logits.size(0), Q = logits.size(1);
    Tensor bits = at::empty({B, Q, 32}, logits.options().dtype(at::kUInt32));
    const bool has_rnd = rnd.has_value();
    if (has_rnd) {
        TORCH_CHECK(prob.has_value() && prob->is_cuda() && prob->numel() == 1 && prob->scalar_type() == at::kFloat, "prob must be an fp32 device scalar");
        TORCH_CHECK(rnd->scalar_type() == at::kFloat && rnd->is_contiguous() && rnd->numel() == (int64_t)B * Q, "rand must be fp32 [B,Q]");
    }
    const int warps = B * Q * 32;
    const int threads = 256;
    const float* rp = has_rnd ? rnd->data_ptr<float>() : nullptr;
    const float* pp = has_rnd ? prob->data_ptr<float>() : nullptr;
    unsigned* bp = reinterpret_cast<unsigned*>(bits.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (logits.scalar_type() == at::kFloat)
        build_bits_kernel<float><<<cdiv(warps * 32, threads), threads, 0, stream>>>(logits.data_ptr<float>(), rp, pp, bp, B * Q, Q);
    else
        build_bits_kernel<bf16><<<cdiv(warps * 32, threads), threads, 0, stream>>>(reinterpret_cast<const bf16*>(logits.data_ptr()), rp, pp, bp, B * Q, Q);
    A12_CUDA_CHECK(cudaGetLastError());
    return bits;
}

}  // namespace a12
