// Round 11: ViT patch embedding as an implicit-im2col GEMM, writing the fp32 residual stream x0 directly.
//   x    : bf16 [B, 3, H, W]  (the pipeline's normalized crops)
//   Wt   : bf16 [C, 768]      (= conv.weight.reshape(C, 3*16*16); K order (ch, ky, kx))
//   bias : fp32 [C]           (conv bias already rounded to bf16, i.e. what autocast feeds cuDNN)
//   pre  : fp32 [5, C]        (cls_token + 0*mask_token, then the 4 register tokens)
//   out  : fp32 [B, 1 + 4 + gy*gx, C]   rows 0..4 = prefix, rows 5.. = bf16round(patch @ W^T + bias)
// i.e. exactly bb.prepare_tokens_with_masks(x).float() under bf16 autocast, in one kernel instead of
// cuDNN implicit_convolve_sgemm + cat + .float() (+ per-call weight casts).
// The A operand is gathered straight from the image: one patch row (16 pixels) is 32 contiguous bytes = 2 x 16B cp.async.
// GEMM structure (mma.sync m16n8k16 bf16, cp.async multi-stage, ldmatrix with XOR swizzle) follows round5 mg_core.cu.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 32
#endif
#ifndef ST
#define ST 3
#endif
#ifndef WM
#define WM 2
#endif
#ifndef WN
#define WN 2
#endif
#ifndef MINB
#define MINB 3
#endif
#define PS 16
#define KTOT 768

typedef __nv_bfloat16 bf16;
static_assert(BK == 32 || BK == 64, "BK");
constexpr int NT = WM * WN * 32;
constexpr int WTM = BM / WM, WTN = BN / WN;
constexpr int MI = WTM / 16, NJ = WTN / 8;
static_assert(NJ % 2 == 0 && MI >= 1, "warp tile");
constexpr int CPR = BK / 8;
constexpr int A_CH = BM * BK / 8, W_CH = BN * BK / 8;
static_assert(A_CH % NT == 0 && W_CH % NT == 0, "loader divisibility");
constexpr int A_LD = A_CH / NT, W_LD = W_CH / NT;
constexpr int KPC = 256 / BK;        // k tiles per input channel
constexpr int NKT = KTOT / BK;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int swz_unit(int r, int c) {
    int u = r * CPR + c;
    return u ^ ((u >> 3) & 7);
}

struct Params {
    const bf16* x; const bf16* W; const float* bias; const float* pre; float* out;
    int C, Himg, Wimg, gx, npatch, ntok;
};

__global__ void __launch_bounds__(NT, MINB) patch_gemm_kernel(const Params p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    bf16* As = reinterpret_cast<bf16*>(smem_raw);
    bf16* Ws = As + ST * BM * BK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;       // m = b * npatch + patch
    const int b = m0 / p.npatch, pbase = m0 - b * p.npatch;      // tile never straddles two images (npatch % BM == 0)
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    const uint32_t As_u = smem_u32(As), Ws_u = smem_u32(Ws);
    const size_t plane = (size_t)p.Himg * p.Wimg;
    size_t a_off[A_LD]; uint32_t a_dst[A_LD];
    const bf16* w_src[W_LD]; uint32_t w_dst[W_LD];
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
        int u = tid + i * NT;
        int r = u / CPR, c = u % CPR;
        int pidx = pbase + r;
        int py = pidx / p.gx, px = pidx - py * p.gx;
        a_off[i] = ((size_t)(b * 3) * p.Himg + (size_t)(py * PS + (c >> 1))) * p.Wimg + px * PS + (c & 1) * 8;
        a_dst[i] = As_u + (uint32_t)(swz_unit(r, c) * 16);
    }
#pragma unroll
    for (int i = 0; i < W_LD; i++) {
        int u = tid + i * NT;
        int r = u / CPR, c = u % CPR;
        w_src[i] = p.W + (size_t)(n0 + r) * KTOT + c * 8;
        w_dst[i] = Ws_u + (uint32_t)(swz_unit(r, c) * 16);
    }
    uint32_t a_ld[BK / 16][MI], w_ld[BK / 16][NJ / 2];
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks++) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            int r = wm * WTM + mi * 16 + (lane & 15), c = ks * 2 + (lane >> 4);
            a_ld[ks][mi] = swz_unit(r, c) * 16;
        }
#pragma unroll
        for (int j2 = 0; j2 < NJ / 2; j2++) {
            int blk = wn * NJ + 2 * j2 + (lane >> 4);
            int r = blk * 8 + (lane & 7), c = ks * 2 + ((lane >> 3) & 1);
            w_ld[ks][j2] = swz_unit(r, c) * 16;
        }
    }
    auto load_tile = [&](int st, int kt) {
        const size_t koff = (size_t)(kt / KPC) * plane + (size_t)(kt % KPC) * (BK / 16) * p.Wimg;
#pragma unroll
        for (int i = 0; i < A_LD; i++)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a_dst[i] + st * (BM * BK * 2)),
                         "l"(p.x + a_off[i] + koff));
#pragma unroll
        for (int i = 0; i < W_LD; i++)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(w_dst[i] + st * (BN * BK * 2)),
                         "l"(w_src[i] + (size_t)kt * BK));
    };
#pragma unroll
    for (int s = 0; s < ST - 1; s++) { load_tile(s, s); cp_commit(); }

#pragma unroll 1
    for (int kt = 0; kt < NKT; kt++) {
        cp_wait<ST - 2>();
        __syncthreads();
        int nk = kt + ST - 1;
        if (nk < NKT) load_tile(nk % ST, nk);
        cp_commit();
        const int st = kt % ST;
        const uint32_t a_base = As_u + st * BM * BK * 2, w_base = Ws_u + st * BN * BK * 2;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t af[MI][4], bfr[NJ][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) ldsm_x4(a_base + a_ld[ks][mi], af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
#pragma unroll
            for (int j2 = 0; j2 < NJ / 2; j2++)
                ldsm_x4(w_base + w_ld[ks][j2], bfr[2 * j2][0], bfr[2 * j2][1], bfr[2 * j2 + 1][0], bfr[2 * j2 + 1][1]);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
        }
    }
    cp_wait<0>();

    // ---- epilogue: bf16(acc), + bf16 bias, round to bf16 again (bit-exact with torch's bf16 conv + bias add), store as fp32 into the token buffer
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int pidx = pbase + wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
            float* orow = p.out + ((size_t)b * p.ntok + 5 + pidx) * p.C;
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                const int col = n0 + (wn * NJ + nj) * 8 + (lane & 3) * 2;
                float2 bi = *reinterpret_cast<const float2*>(p.bias + col);
                // reference = bf16 conv (bf16 result) followed by a bf16 bias add -> two roundings
                float v0 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(acc[mi][nj][hh * 2])) + bi.x));
                float v1 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(acc[mi][nj][hh * 2 + 1])) + bi.y));
                *reinterpret_cast<float2*>(orow + col) = make_float2(v0, v1);
            }
        }
    }
    // ---- prefix rows (cls + registers) of this image: written by the CTAs of its first M tile
    if (pbase == 0) {
        for (int i = tid; i < 5 * (BN / 4); i += NT) {
            int row = i / (BN / 4), c4 = i % (BN / 4);
            float4 v = *reinterpret_cast<const float4*>(p.pre + (size_t)row * p.C + n0 + c4 * 4);
            *reinterpret_cast<float4*>(p.out + ((size_t)b * p.ntok + row) * p.C + n0 + c4 * 4) = v;
        }
    }
}

void patch_embed_launch(torch::Tensor x, torch::Tensor W, torch::Tensor bias, torch::Tensor pre, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16, "bf16 x/W");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && pre.scalar_type() == torch::kFloat32 &&
                out.scalar_type() == torch::kFloat32, "fp32 bias/prefix/out");
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && out.is_contiguous(), "contiguous");
    int B = (int)x.size(0), Himg = (int)x.size(2), Wimg = (int)x.size(3);
    int C = (int)W.size(0);
    int gx = Wimg / PS, gy = Himg / PS, npatch = gx * gy;
    TORCH_CHECK(x.size(1) == 3 && W.size(1) == KTOT && Himg % PS == 0 && Wimg % PS == 0, "shape");
    TORCH_CHECK(npatch % BM == 0 && C % BN == 0, "tile divisibility");
    TORCH_CHECK(out.size(0) == B && out.size(1) == npatch + 5 && out.size(2) == C, "out shape");
    Params p{reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<const bf16*>(W.data_ptr()),
             bias.data_ptr<float>(), pre.data_ptr<float>(), out.data_ptr<float>(), C, Himg, Wimg, gx, npatch, npatch + 5};
    size_t smem = (size_t)ST * (BM + BN) * BK * 2;
    static size_t attr = 0;
    if (smem > attr) {
        cudaFuncSetAttribute(patch_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = smem;
    }
    dim3 grid(C / BN, B * npatch / BM);
    patch_gemm_kernel<<<grid, NT, smem, c10::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("patch_embed_launch", &patch_embed_launch); }
