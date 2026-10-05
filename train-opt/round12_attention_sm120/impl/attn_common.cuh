// Shared device helpers of the round12 attention kernels (forward, backward, mask builder).
//
// Conventions
//   * qkv is the packed qkv Linear output [B][N][3][H][64] bf16; O / dO are [B][N][H][64] bf16.
//   * RoPE (DINOv3 axial rope) rotates the last 1024 tokens (token >= prefix), fp32 math on the bf16
//     inputs, result rounded to bf16.  The sin/cos tables are fp32 [1024][64] with duplicated halves
//     and are separable: dims 0..15 depend only on y = pos / 32, dims 16..31 only on x = pos % 32
//     (rope_position_encoding.py: angles = 2*pi*coords[:, :, None] / periods -> [HW, 2, 16]).  The kernels
//     read 4 slices of 32 x 16 floats into shared memory instead of the whole table.
//   * Query bitmask u32 [B][100][32]: bit p%32 of word p/32 = query row r may see patch key mask_key0 + p.
//   * lse is stored in the log2 domain, scaled: lse2 = m + log2(l) with m the running max of s*scale*log2e.
#pragma once

#include "third_party/spark_common.cuh"

namespace a12 {

constexpr int HD = 64;                 // head dim
constexpr int ROPE_W = 32;             // patch grid width (512 / 16)
constexpr int ROPE_TAB_FLOATS = 4 * 32 * 16;  // cosY, sinY, cosX, sinX
constexpr float kLog2e = 1.4426950408889634f;

// Offset (elements) of logical (row, 16-byte chunk) in a [rows][64] tile loaded with a 128-byte-swizzled TMA box.
__device__ __forceinline__ int swz_off(int row, int ch) { return row * 64 + ((ch ^ (row & 7)) << 3); }

// Copies the four RoPE slices into shared memory: tab[0]=cosY, [1]=sinY, [2]=cosX, [3]=sinX, each [32][16] floats.
__device__ __forceinline__ void load_rope_tab(float* tab, const float* sinT, const float* cosT, int tid, int nthreads) {
    for (int i = tid; i < ROPE_TAB_FLOATS; i += nthreads) {
        const int t = i >> 9;         // table
        const int r = (i >> 4) & 31;  // y or x
        const int d = i & 15;
        const float* src = (t & 1) ? sinT : cosT;
        const int idx = (t < 2) ? (r * ROPE_W) * 64 + d : r * 64 + 16 + d;
        tab[i] = src[idx];
    }
}

// Forward rotation of one chunk pair: lo = dims d0..d0+7, hi = dims d0+32..d0+39, c/s = 8 cos/sin floats.
// Matches rope_apply in fp32 (x*cos + rotate_half(x)*sin, separate roundings): __f*_rn never contract.
__device__ __forceinline__ void rope_rot8(uint4& lo, uint4& hi, const float* c, const float* s) {
    unsigned a[4] = {lo.x, lo.y, lo.z, lo.w};
    unsigned b[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x1a = bf16lo(a[i]), x1b = bf16hi(a[i]);
        const float x2a = bf16lo(b[i]), x2b = bf16hi(b[i]);
        const float n1a = __fsub_rn(__fmul_rn(x1a, c[2 * i]), __fmul_rn(x2a, s[2 * i]));
        const float n1b = __fsub_rn(__fmul_rn(x1b, c[2 * i + 1]), __fmul_rn(x2b, s[2 * i + 1]));
        const float n2a = __fadd_rn(__fmul_rn(x2a, c[2 * i]), __fmul_rn(x1a, s[2 * i]));
        const float n2b = __fadd_rn(__fmul_rn(x2b, c[2 * i + 1]), __fmul_rn(x1b, s[2 * i + 1]));
        a[i] = pack_bf16x2(n1a, n1b);
        b[i] = pack_bf16x2(n2a, n2b);
    }
    lo = make_uint4(a[0], a[1], a[2], a[3]);
    hi = make_uint4(b[0], b[1], b[2], b[3]);
}

// Rotates the 64-wide rows of a swizzled [rows][64] smem tile whose first row has token index tok0.
// Rows with token < prefix or token >= limit (>= N: zero fill) are left alone.  Called by one warp.  When gdst is not null the
// (rotated or untouched) rows are also written to global memory: gdst + token * (2 * H * 64) (the qk_rot k slot of the head).
__device__ __forceinline__ void rope_rotate_tile(bf16* tile, int rows, int tok0, int prefix, int limit, const float* tab, int lane, bf16* gdst = nullptr,
                                                 int gstride = 0) {
    for (int i = lane; i < rows * 4; i += 32) {
        const int r = i >> 2, c = i & 3;  // chunk pair (c, c + 4)
        const int tok = tok0 + r;
        if (tok >= limit) continue;
        const int pos = tok - prefix;
        uint4* p_lo = reinterpret_cast<uint4*>(tile + swz_off(r, c));
        uint4* p_hi = reinterpret_cast<uint4*>(tile + swz_off(r, c + 4));
        uint4 lo = *p_lo, hi = *p_hi;
        if (pos >= 0) {
            const int y = pos >> 5, x = pos & 31;
            const float* ct = c < 2 ? tab + (0 * 32 + y) * 16 + 8 * c : tab + (2 * 32 + x) * 16 + 8 * (c - 2);
            const float* st = c < 2 ? tab + (1 * 32 + y) * 16 + 8 * c : tab + (3 * 32 + x) * 16 + 8 * (c - 2);
            float cv[8], sv[8];
            *reinterpret_cast<float4*>(cv) = *reinterpret_cast<const float4*>(ct);
            *reinterpret_cast<float4*>(cv + 4) = *reinterpret_cast<const float4*>(ct + 4);
            *reinterpret_cast<float4*>(sv) = *reinterpret_cast<const float4*>(st);
            *reinterpret_cast<float4*>(sv + 4) = *reinterpret_cast<const float4*>(st + 4);
            rope_rot8(lo, hi, cv, sv);
            *p_lo = lo;
            *p_hi = hi;
        }
        if (gdst != nullptr) {
            bf16* d = gdst + (size_t)tok * gstride + 8 * c;
            *reinterpret_cast<uint4*>(d) = lo;
            *reinterpret_cast<uint4*>(d + 32) = hi;
        }
    }
}

// 32-bit window of the query bitmask row for keys [kv0 + 32*half, +32): bit kk = key kv0 + 32*half + kk allowed.
// Words below index 0 (keys < mask_key0) and above 31 read as "allowed".
__device__ __forceinline__ unsigned mask_word(const unsigned* row_bits, int wbase, int shift) {
    unsigned wa = (wbase >= 0 && wbase < 32) ? __ldg(row_bits + wbase) : 0xffffffffu;
    unsigned wb = (wbase + 1 >= 0 && wbase + 1 < 32) ? __ldg(row_bits + wbase + 1) : 0xffffffffu;
    return __funnelshift_r(wa, wb, shift);
}

}  // namespace a12
