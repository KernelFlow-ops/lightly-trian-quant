// Round12 forward attention for sm_120 (RTX 5090): packed qkv in, fused RoPE, query bitmask, tail-aware tiles.
//
// Design (adapted from spark-kernels attention.cu variants 4/5, MIT, commit a401bfa9e363):
//   * 8 consumer warps x 16 query rows = 128-row Q tile, online softmax on the mma.sync.m16n8k16 accumulators,
//     P re-packed in registers, K/V tiles of 64 keys streamed by TMA through a STAGES-deep mbarrier ring.
//   * warp 8 is the TMA producer (one lane); warp 9 is the RoPE rotator: after a K tile lands it rotates the
//     keys of the patch range in shared memory (fp32 math, bf16 rounding, bit-identical to rope_apply) and only
//     then releases the stage to the consumers.  Q is rotated in registers once per CTA.
//   * The key loop covers N exactly: full 64-key tiles, then one tail tile whose n8 / k16 steps are predicated
//     on the number of valid keys (5 of 64 for N = 1029).  Warps whose 16 rows lie past N retire immediately.
//   * Joint blocks: query rows [0, mask_rows) are masked against patch keys >= mask_key0 by a [B][100][32] bit
//     mask applied on the S accumulators (only the CTA that owns rows < mask_rows runs this code).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "attn_common.cuh"
#include "attn_api.h"

namespace a12 {

namespace {

constexpr int BM = 128, BN = 64, CW = 8, STAGES = 4;
constexpr int THREADS = CW * 32 + 64;  // consumers + producer warp + rotator warp
constexpr int Q_BYTES = BM * 128;
constexpr int KV_BYTES = BN * 128;  // one K or V tile
constexpr int STAGE_BYTES = 2 * KV_BYTES;
constexpr int SMEM_BYTES = Q_BYTES + STAGES * STAGE_BYTES + ROPE_TAB_FLOATS * 4 + 256;

struct FwdArgs {
    bf16* O;
    float* lse2;
    bf16* qk_rot;  // optional [B][N][2][H][64]: rotated q (every CTA) and k (CTAs of q-tile 0) for the backward
    const float* sinT;
    const float* cosT;
    const unsigned* bits;
    int N, H, BH, prefix, mask_rows, mask_key0;
    float scale_log2;
};

template <bool ROPE, bool MASK>
__global__ void __launch_bounds__(THREADS, 1) fwd_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmKV, FwdArgs a) {
    extern __shared__ __align__(1024) unsigned char smem[];
    bf16* Qs = reinterpret_cast<bf16*>(smem);
    unsigned char* stages = smem + Q_BYTES;
    float* tab = reinterpret_cast<float*>(stages + STAGES * STAGE_BYTES);
    uint64_t* bars = reinterpret_cast<uint64_t*>(tab + ROPE_TAB_FLOATS);
    uint64_t* q_full = bars;
    uint64_t* full = bars + 1;
    uint64_t* rot = full + STAGES;
    uint64_t* empty = rot + STAGES;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    const int bh = blockIdx.x % a.BH, qt = blockIdx.x / a.BH;  // q-tile 0 (masked rows) runs first
    const int b = bh / a.H, h = bh - b * a.H;
    const int q0 = qt * BM;
    const int N = a.N;
    const int T = cdiv(N, BN);
    const int kv_tail = N - BN * (T - 1);  // valid keys of the last tile, 1..64
    const int n_active = min(CW, cdiv(N - q0, 16));

    if (tid == 0) {
        mbar_init(q_full, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&rot[s], 1);
            mbar_init(&empty[s], n_active);
        }
        fence_mbar_init();
    }
    if constexpr (ROPE) load_rope_tab(tab, a.sinT, a.cosT, tid, THREADS);
    __syncthreads();

    if (warp == CW) {  // ---- TMA producer
        if (lane == 0) {
            prefetch_tensormap(&tmQ);
            prefetch_tensormap(&tmKV);
            mbar_arrive_expect_tx(q_full, Q_BYTES);
            tma_load_4d(Qs, &tmQ, q_full, 0, h, q0, b);
            for (int u = 0; u < T; ++u) {
                const int s = u % STAGES;
                const unsigned use = u / STAGES;
                if (use > 0) mbar_wait(&empty[s], (use - 1) & 1);
                mbar_arrive_expect_tx(&full[s], STAGE_BYTES);
                unsigned char* dst = stages + s * STAGE_BYTES;
                tma_load_4d(dst, &tmKV, &full[s], 0, a.H + h, u * BN, b);
                tma_load_4d(dst + KV_BYTES, &tmKV, &full[s], 0, 2 * a.H + h, u * BN, b);
            }
        }
        return;
    }
    if (warp == CW + 1) {  // ---- RoPE rotator
        if constexpr (ROPE) {
            bf16* kdst = (qt == 0 && a.qk_rot != nullptr) ? a.qk_rot + (((size_t)b * N) * 2 + 1) * a.H * 64 + h * 64 : nullptr;
            for (int u = 0; u < T; ++u) {
                const int s = u % STAGES;
                mbar_wait(&full[s], (u / STAGES) & 1);
                rope_rotate_tile(reinterpret_cast<bf16*>(stages + s * STAGE_BYTES), BN, u * BN, a.prefix, N, tab, lane, kdst, 2 * a.H * 64);
                __syncwarp();
                if (lane == 0) mbar_arrive(&rot[s]);
            }
        }
        return;
    }
    if (warp >= n_active) return;  // rows all past N

    // ---- consumers
    unsigned qf[4][4];
    mbar_wait(q_full, 0);
    {
        const int row = warp * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) ldmatrix_x4(qf[kk], Qs + swz_off(row, 2 * kk + (lane >> 4)));
    }
    if constexpr (ROPE) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int pos = q0 + warp * 16 + g + 8 * hh - a.prefix;
            if (pos >= 0 && pos < 1024) {
                const int y = pos >> 5, x = pos & 31;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int sidx = hh + 2 * half;
                        const int d0 = 16 * kk + 8 * half + c2;
                        const float2 cc = kk == 0 ? *reinterpret_cast<const float2*>(tab + (0 * 32 + y) * 16 + d0)
                                                  : *reinterpret_cast<const float2*>(tab + (2 * 32 + x) * 16 + d0 - 16);
                        const float2 ss = kk == 0 ? *reinterpret_cast<const float2*>(tab + (1 * 32 + y) * 16 + d0)
                                                  : *reinterpret_cast<const float2*>(tab + (3 * 32 + x) * 16 + d0 - 16);
                        const unsigned lo = qf[kk][sidx], hi = qf[kk + 2][sidx];
                        const float x1a = bf16lo(lo), x1b = bf16hi(lo), x2a = bf16lo(hi), x2b = bf16hi(hi);
                        qf[kk][sidx] = pack_bf16x2(__fsub_rn(__fmul_rn(x1a, cc.x), __fmul_rn(x2a, ss.x)),
                                                   __fsub_rn(__fmul_rn(x1b, cc.y), __fmul_rn(x2b, ss.y)));
                        qf[kk + 2][sidx] = pack_bf16x2(__fadd_rn(__fmul_rn(x2a, cc.x), __fmul_rn(x1a, ss.x)),
                                                       __fadd_rn(__fmul_rn(x2b, cc.y), __fmul_rn(x1b, ss.y)));
                    }
                }
            }
        }
    }

    if constexpr (ROPE) {
        if (a.qk_rot != nullptr) {  // rotated q (and untouched prefix rows) for the backward
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int sidx = 0; sidx < 4; ++sidx) {
                    const int row = q0 + warp * 16 + g + 8 * (sidx & 1);
                    if (row < N) *reinterpret_cast<unsigned*>(a.qk_rot + (((size_t)b * N + row) * 2 + 0) * a.H * 64 + h * 64 + 16 * kk + 8 * (sidx >> 1) + c2) = qf[kk][sidx];
                }
        }
    }

    float o[8][4];
#pragma unroll
    for (int dj = 0; dj < 8; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float sacc[8][4];
    unsigned pa[4][4];

    const int row0 = q0 + warp * 16 + g;  // rows row0 and row0 + 8
    const bool masked_warp = MASK && (q0 + warp * 16 < a.mask_rows);
    const unsigned* bits0 = MASK ? a.bits + ((size_t)b * a.mask_rows + min(row0, a.mask_rows - 1)) * 32 : nullptr;
    const unsigned* bits1 = MASK ? a.bits + ((size_t)b * a.mask_rows + min(row0 + 8, a.mask_rows - 1)) * 32 : nullptr;
    const bool row0_masked = MASK && row0 < a.mask_rows, row1_masked = MASK && row0 + 8 < a.mask_rows;

    auto qk = [&](const bf16* ks, bool tail) {
#pragma unroll
        for (int nj = 0; nj < 8; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int nj = 0; nj < 8; nj += 2) {
                if (tail && nj * 8 >= kv_tail) continue;
                const int row = nj * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + swz_off(row, 2 * kk + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sacc[nj], qf[kk], b0);
                if (!tail || (nj + 1) * 8 < kv_tail) mma_bf16_16816(sacc[nj + 1], qf[kk], b1);
            }
        }
    };
    auto softmax = [&](int t, bool tail) {
        const int kv0 = t * BN;
        if (tail) {
#pragma unroll
            for (int nj = 0; nj < 8; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (nj * 8 + c2 + (e & 1) >= kv_tail) sacc[nj][e] = -INFINITY;
        }
        if constexpr (MASK) {
            if (masked_warp && kv0 + BN > a.mask_key0) {  // warp-uniform
                const int p0 = kv0 - a.mask_key0;
                const int wbase = p0 >> 5, shift = p0 & 31;
                unsigned w0[2] = {~0u, ~0u}, w1[2] = {~0u, ~0u};
                if (row0_masked) {
                    w0[0] = mask_word(bits0, wbase, shift);
                    w0[1] = mask_word(bits0, wbase + 1, shift);
                }
                if (row1_masked) {
                    w1[0] = mask_word(bits1, wbase, shift);
                    w1[1] = mask_word(bits1, wbase + 1, shift);
                }
#pragma unroll
                for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const unsigned w = (e >> 1) ? w1[nj >> 2] : w0[nj >> 2];
                        const int sh = (nj & 3) * 8 + c2 + (e & 1);
                        if (!((w >> sh) & 1u)) sacc[nj][e] = -INFINITY;
                    }
                }
            }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(sacc[0][2 * r], sacc[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < 8; ++nj) mx = fmaxf(mx, fmaxf(sacc[nj][2 * r], sacc[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * a.scale_log2);  // running max: never reset per tile
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            alpha[r] = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const float p0 = ex2(fmaf(sacc[nj][2 * r], a.scale_log2, -m_use));
                const float p1 = ex2(fmaf(sacc[nj][2 * r + 1], a.scale_log2, -m_use));
                sacc[nj][2 * r] = p0;
                sacc[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha[r], rs);
            m[r] = m_new;
        }
#pragma unroll
        for (int dj = 0; dj < 8; ++dj) {
            o[dj][0] *= alpha[0];
            o[dj][1] *= alpha[0];
            o[dj][2] *= alpha[1];
            o[dj][3] *= alpha[1];
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            pa[kt][0] = pack_bf16x2(sacc[2 * kt][0], sacc[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(sacc[2 * kt][2], sacc[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(sacc[2 * kt + 1][0], sacc[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(sacc[2 * kt + 1][2], sacc[2 * kt + 1][3]);
        }
    };
    auto pv = [&](const bf16* vs, bool tail) {
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            if (tail && kt * 16 >= kv_tail) continue;
#pragma unroll
            for (int dj = 0; dj < 8; dj += 2) {
                const int row = kt * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, vs + swz_off(row, dj + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(o[dj], pa[kt], b0);
                mma_bf16_16816(o[dj + 1], pa[kt], b1);
            }
        }
    };

    for (int u = 0; u < T; ++u) {
        const int s = u % STAGES;
        mbar_wait(ROPE ? &rot[s] : &full[s], (u / STAGES) & 1);
        const bf16* ks = reinterpret_cast<const bf16*>(stages + s * STAGE_BYTES);
        const bf16* vs = ks + BN * 64;
        if (u + 1 < T || kv_tail == BN) {
            qk(ks, false);
            softmax(u, false);
            pv(vs, false);
        } else {
            qk(ks, true);
            softmax(u, true);
            pv(vs, true);
        }
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }

    // ---- epilogue: normalized bf16 rows (O is [B][N][H][64]) and the log2-domain lse.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int row = row0 + 8 * r;
        if (row >= N) continue;
        const float inv = 1.f / ls;
        bf16* out = a.O + (((size_t)b * N + row) * a.H + h) * 64 + c2;
#pragma unroll
        for (int dj = 0; dj < 8; ++dj)
            *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) = __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
        if ((lane & 3) == 0) a.lse2[((size_t)b * a.H + h) * N + row] = m[r] + __log2f(ls);
    }
}

template <bool ROPE, bool MASK>
void launch(const Tensor& qkv, Tensor& O, Tensor& lse2, bf16* qk_rot, const float* sinT, const float* cosT, const unsigned* bits, int H, int prefix,
            int mask_rows, int mask_key0, cudaStream_t stream) {
    const int B = qkv.size(0), N = qkv.size(1);
    static bool attr = false;
    if (!attr) {
        A12_CUDA_CHECK(cudaFuncSetAttribute(fwd_kernel<ROPE, MASK>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attr = true;
    }
    const CUtensorMap tmQ = make_slot_map(qkv.data_ptr(), B, N, 3 * H, BM);
    const CUtensorMap tmKV = make_slot_map(qkv.data_ptr(), B, N, 3 * H, BN);
    FwdArgs a{reinterpret_cast<bf16*>(O.data_ptr()), lse2.data_ptr<float>(), qk_rot, sinT, cosT, bits, N, H, B * H, prefix, mask_rows, mask_key0,
              0.125f * kLog2e};
    const int grid = B * H * cdiv(N, BM);
    fwd_kernel<ROPE, MASK><<<grid, THREADS, SMEM_BYTES, stream>>>(tmQ, tmKV, a);
    A12_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> attn_fwd(const Tensor& qkv, int64_t heads, const std::optional<Tensor>& sinT, const std::optional<Tensor>& cosT,
                                            const std::optional<Tensor>& bits, int64_t prefix, int64_t mask_rows, int64_t mask_key0, bool save_rot) {
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.is_contiguous(), "qkv must be contiguous bf16 cuda");
    TORCH_CHECK(qkv.size(2) == 3 * heads * 64 || (qkv.dim() == 5 && qkv.size(2) == 3 && qkv.size(3) == heads && qkv.size(4) == 64),
                "qkv must be [B,N,3*H*64] or [B,N,3,H,64]");
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int B = qkv.size(0), N = qkv.size(1), H = heads;
    Tensor O = at::empty({B, N, (int64_t)H * 64}, qkv.options());
    Tensor lse2 = at::empty({B, H, N}, qkv.options().dtype(at::kFloat));
    const bool rope = sinT.has_value() && prefix >= 0;
    const bool mask = bits.has_value() && mask_rows > 0;
    // rotated q|k for the backward (only when RoPE is fused and a gradient is needed)
    Tensor qk_rot = (rope && save_rot) ? at::empty({B, N, 2, H, 64}, qkv.options()) : at::empty({0}, qkv.options());
    bf16* qkr = (rope && save_rot) ? reinterpret_cast<bf16*>(qk_rot.data_ptr()) : nullptr;
    const float* s = rope ? sinT->data_ptr<float>() : nullptr;
    const float* c = rope ? cosT->data_ptr<float>() : nullptr;
    const unsigned* bp = mask ? reinterpret_cast<const unsigned*>(bits->data_ptr()) : nullptr;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    if (rope && mask) launch<true, true>(qkv, O, lse2, qkr, s, c, bp, H, (int)prefix, (int)mask_rows, (int)mask_key0, st);
    else if (rope) launch<true, false>(qkv, O, lse2, qkr, s, c, bp, H, (int)prefix, 0, 0, st);
    else if (mask) launch<false, true>(qkv, O, lse2, nullptr, s, c, bp, H, 1 << 30, (int)mask_rows, (int)mask_key0, st);
    else launch<false, false>(qkv, O, lse2, nullptr, s, c, bp, H, 1 << 30, 0, 0, st);
    return {O, lse2, qk_rot};
}

}  // namespace a12
