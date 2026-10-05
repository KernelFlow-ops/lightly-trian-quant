"""One-off: derive impl/mg8_core.cu (int8 W8A8, s32 accumulate, per-token x per-channel dequant epilogue) from impl/mg16_core.cu and mg8b.py from mg16b.py."""
s = open("impl/mg16_core.cu").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


rep("// Round 10 PROTOTYPE `mg_f16acc`:", """// Round 10 PROTOTYPE `mg_i8` (derived from the `mg_f16acc` prototype, itself a copy of round5_gemm_sm120/impl/mg_core.cu):
//   int8 W8A8 GEMM, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, int32 accumulators (exact), A [M,K] / W [N,K] int8 K-contiguous.
//   BK counts bf16-equivalent elements (bytes / 2): BK = 32 -> a k-tile of 64 int8 elements (2 MMA k32 steps), BK = 64 -> 128.  cp.async / ldmatrix / swizzle code is
//   byte-identical to the 16-bit kernels (an m16n8k32 s8 A/B fragment has the same register layout as m16n8k16 f16).
//   Dequant in every epilogue: acc * sx[row] * sw[col]   (sx = per-token activation scale [M] -> Params.stats_in, sw = per-output-channel weight scale [N] -> Params.cs),
//   then the R5 epilogues unchanged: bias (+GELU) / bias + RoPE + [3,B,N,H,64] layout / gamma + fp32 residual / plain.  No split-K (SPLITK = 1).
// Original header of the f16acc prototype follows.
// Round 10 PROTOTYPE `mg_f16acc`:""")
rep("    const bf16* A; const bf16* W; const void* bias;", "    const int8_t* A; const int8_t* W; const void* bias;")
rep("#ifndef F16ACC\n#define F16ACC 1", "#ifndef F16ACC\n#define F16ACC 0")
rep("#define OUTF16 1      // EPI 0/2 output dtype: 1 = fp16, 0 = bf16", "#define OUTF16 0      // EPI 0/2 output dtype: 1 = fp16, 0 = bf16")
# the s8 MMA replaces the bf16 one (F16ACC = 0 path)
rep('''__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])''',
    '''__device__ __forceinline__ void mma16816(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {   // = m16n8k32 s8 (32 bytes of k per step, like k16 for 16-bit)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])''')
rep("constexpr int NT = WM * WN * 32;", "constexpr int NT = WM * WN * 32;\nconstexpr int BKE = BK * 2;   // int8 elements per k-tile\nstatic_assert(SPLITK == 1, \"mg8: no split-K\");\n#define DEQ(a_, row_, col_) ((float)(a_) * __ldg(p.stats_in + (row_)) * __ldg(p.cs + (col_)))")
rep("    const int KT_ALL = K / BK;", "    const int KT_ALL = K / BKE;")
rep("    float acc[MI][NJ][4];", "    int acc[MI][NJ][4];")
rep("for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;", "for (int e = 0; e < 4; e++) acc[i][j][e] = 0;")
rep("    const bf16* a_src[A_LD]; uint32_t a_dst[A_LD]; int a_sz[A_LD];\n    const bf16* w_src[W_LD];", "    const int8_t* a_src[A_LD]; uint32_t a_dst[A_LD]; int a_sz[A_LD];\n    const int8_t* w_src[W_LD];")
rep("(size_t)kt0 * BK + c * 8;", "(size_t)kt0 * BKE + c * 16;", 2)
rep('"l"(a_src[i] + (size_t)kt * BK)', '"l"(a_src[i] + (size_t)kt * BKE)')
rep('"l"(w_src[i] + (size_t)kt * BK)', '"l"(w_src[i] + (size_t)kt * BKE)')
# epilogues
rep('''                        x0 = acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(b2.x);
                        x1 = acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(b2.y);''',
    '''                        x0 = DEQ(acc[mi][nj][hh * 2], row, col) + __bfloat162float(b2.x);
                        x1 = DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1) + __bfloat162float(b2.y);''')
rep('''                    float x0 = (acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(b2.x)) * g2.x;
                    float x1 = (acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(b2.y)) * g2.y;''',
    '''                    float x0 = (DEQ(acc[mi][nj][hh * 2], row, col) + __bfloat162float(b2.x)) * g2.x;
                    float x1 = (DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1) + __bfloat162float(b2.y)) * g2.y;''')
rep('''                        pk2(acc[mi][nj][hh * 2] * p.oscale, acc[mi][nj][hh * 2 + 1] * p.oscale);''',
    '''                        pk2(DEQ(acc[mi][nj][hh * 2], row, col), DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1));''')
rep('''                        make_float2(acc[mi][nj][hh * 2] * p.oscale, acc[mi][nj][hh * 2 + 1] * p.oscale);''',
    '''                        make_float2(DEQ(acc[mi][nj][hh * 2], row, col), DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1));''')
rep('''                        l0 = acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(bl.x); l1 = acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(bl.y);
                        h0 = acc[mi][nj + HALF][hh * 2] * p.oscale + __bfloat162float(bh.x); h1 = acc[mi][nj + HALF][hh * 2 + 1] * p.oscale + __bfloat162float(bh.y);''',
    '''                        l0 = DEQ(acc[mi][nj][hh * 2], row, col_lo) + __bfloat162float(bl.x); l1 = DEQ(acc[mi][nj][hh * 2 + 1], row, col_lo + 1) + __bfloat162float(bl.y);
                        h0 = DEQ(acc[mi][nj + HALF][hh * 2], row, col_hi) + __bfloat162float(bh.x); h1 = DEQ(acc[mi][nj + HALF][hh * 2 + 1], row, col_hi + 1) + __bfloat162float(bh.y);''')
open("impl/mg8_core.cu", "w").write(s)

b = open("impl/mg16b.py").read()
for a, c in (('mg16_core.cu', 'mg8_core.cu'), ('build/mg16"', 'build/mg8"'), ('name="mg16l"', 'name="mg8l"'), ('mg16_{h}', 'mg8_{h}'),
             ("f16acc=1, outf16=1, pkt=1", "f16acc=0, outf16=0, pkt=1")):
    assert a in b, a
    b = b.replace(a, c)
open("impl/mg8b.py", "w").write(b)
print("ok")
