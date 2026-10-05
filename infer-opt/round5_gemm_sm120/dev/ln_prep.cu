// Round 5: LN "prep" kernel - one warp per row.  Builds the raw-residual triplet used by the LN-fused GEMMs:
//   x_out (fp32 residual, optional), xb (bf16 copy), stats (sum, M2 about the row mean; one partial per row).
// Optionally prepends Q query rows (joint-layer entry):  out row (b, i):  i < Q ? queries[i] : x[b, i - Q].
// Compiled to a cubin per D (feature dim: 1024 or 768) by mgb.py::build_prep.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#ifndef D
#define D 1024
#endif
static_assert(D % 256 == 0, "D");
constexpr int NV = D / 256;
struct PrepParams { const float* x; const float* queries; float* x_out; __nv_bfloat16* xb; float2* stats; int Mout, Nin, Q, pad; };

extern "C" __global__ void __launch_bounds__(128) ln_prep(const PrepParams p) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= p.Mout) return;
    const int Nout = p.Nin + p.Q;
    const int b = row / Nout, i = row % Nout;
    const float* src = (i < p.Q) ? p.queries + (size_t)i * D : p.x + ((size_t)b * p.Nin + (i - p.Q)) * D;
    float v[NV][8];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        const float4* q = reinterpret_cast<const float4*>(src + (j * 32 + lane) * 8);
        float4 a = __ldg(q), c = __ldg(q + 1);
        v[j][0] = a.x; v[j][1] = a.y; v[j][2] = a.z; v[j][3] = a.w; v[j][4] = c.x; v[j][5] = c.y; v[j][6] = c.z; v[j][7] = c.w;
#pragma unroll
        for (int e = 0; e < 8; e++) sum += v[j][e];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float mean = sum * (1.f / D);
    float m2 = 0.f;
#pragma unroll
    for (int j = 0; j < NV; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) { float t = v[j][e] - mean; m2 += t * t; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m2 += __shfl_xor_sync(0xffffffffu, m2, o);
#pragma unroll
    for (int j = 0; j < NV; j++) {
        const size_t e0 = (size_t)row * D + (j * 32 + lane) * 8;
        if (p.x_out) {
            float4* o = reinterpret_cast<float4*>(p.x_out + e0);
            o[0] = make_float4(v[j][0], v[j][1], v[j][2], v[j][3]);
            o[1] = make_float4(v[j][4], v[j][5], v[j][6], v[j][7]);
        }
        uint4 u;
        __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; e++) q[e] = __floats2bfloat162_rn(v[j][2 * e], v[j][2 * e + 1]);
        *reinterpret_cast<uint4*>(p.xb + e0) = u;
    }
    if (lane == 0) p.stats[row] = make_float2(sum, m2);
}
