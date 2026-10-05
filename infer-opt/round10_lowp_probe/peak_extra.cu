// Round 10: instruction-rate peaks not covered by Spark's bench_peak.cu (same method: register-resident operands, 8 independent
// accumulator chains per warp, 4 blocks x 8 warps per SM, SM clock sampled during the loop).
//   bf16 / fp16 inputs with fp32 accumulate (m16n8k16), fp16 inputs with fp16 accumulate (m16n8k16),
//   int8 x int8 -> s32 (m16n8k32), plain e4m3 fp32 accumulate (m16n8k32) for a same-harness cross-check.
// Spark's own peak_fp8_mma / peak_fp8_mma_plain / peak_fp8_mma_f16acc / peak_fp4_mma(_mx) are measured by building bench_peak.cu as is.
// Build: nvcc -arch=sm_120a -O3 -std=c++17 peak_extra.cu -o peak_extra
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <functional>

constexpr int kThreads = 256;
constexpr int kChains = 8;

enum Kind { BF16_F32, FP16_F32, FP16_F16, INT8_S32, E4M3_PLAIN_F32 };

template <int KIND>
__global__ void __launch_bounds__(kThreads) peak_kernel(int iters, float* sink, long long* clk) {
    unsigned a0, a1, a2, a3, b0, b1;
    a0 = a1 = a2 = a3 = b0 = b1 = 0x3c003c00u ^ (threadIdx.x & 1);   // fp16 1.0 pair (also harmless for the other kinds)
    if (KIND == BF16_F32) a0 = a1 = a2 = a3 = b0 = b1 = 0x3f803f80u ^ (threadIdx.x & 1);
    if (KIND == INT8_S32) a0 = a1 = a2 = a3 = b0 = b1 = 0x01010101u ^ (threadIdx.x & 1);
    if (KIND == E4M3_PLAIN_F32) a0 = a1 = a2 = a3 = b0 = b1 = 0x38383838u ^ (threadIdx.x & 1);
    float accf[kChains][4];
    unsigned acch[kChains][2];
    int acci[kChains][4];
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
        accf[c][0] = accf[c][1] = accf[c][2] = accf[c][3] = 0.f;
        acch[c][0] = acch[c][1] = 0u;
        acci[c][0] = acci[c][1] = acci[c][2] = acci[c][3] = 0;
    }
    long long c0 = 0, t0 = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        c0 = clock64();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            if (KIND == BF16_F32) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(accf[c][0]), "+f"(accf[c][1]), "+f"(accf[c][2]), "+f"(accf[c][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            } else if (KIND == FP16_F32) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(accf[c][0]), "+f"(accf[c][1]), "+f"(accf[c][2]), "+f"(accf[c][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            } else if (KIND == FP16_F16) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(acch[c][0]), "+r"(acch[c][1])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            } else if (KIND == INT8_S32) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(acci[c][0]), "+r"(acci[c][1]), "+r"(acci[c][2]), "+r"(acci[c][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            } else {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(accf[c][0]), "+f"(accf[c][1]), "+f"(accf[c][2]), "+f"(accf[c][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long long c1 = clock64(), t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        clk[0] = c1 - c0;
        clk[1] = t1 - t0;
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kChains; ++c) s += accf[c][0] + accf[c][1] + accf[c][2] + accf[c][3] + (float)(acch[c][0] + acch[c][1]) + (float)(acci[c][0] + acci[c][3]);
    if (s == 12345.f) sink[threadIdx.x] = s;
}

static float time_ms(const std::function<void()>& fn, cudaStream_t st) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, st);
    fn();
    cudaEventRecord(b, st);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

template <int KIND>
void run(const char* name, double flops_per_mma, int blocks, float* sink, long long* clk, cudaStream_t st, double target_ms) {
    auto launch = [&](int n) { peak_kernel<KIND><<<blocks, kThreads, 0, st>>>(n, sink, clk); };
    // clock ramp
    for (int i = 0; i < 5; i++) launch(20000);
    cudaStreamSynchronize(st);
    int n = 1000;
    float t = time_ms([&] { launch(n); }, st);
    n = std::max(100, (int)(n * target_ms / std::max(0.01f, t)));
    float best = 1e30f; long long h[2] = {0, 0}; double bestmhz = 0;
    for (int r = 0; r < 5; r++) {
        float ms = time_ms([&] { launch(n); }, st);
        cudaMemcpy(h, clk, sizeof(h), cudaMemcpyDeviceToHost);
        if (ms < best) { best = ms; bestmhz = h[1] > 0 ? (double)h[0] / h[1] * 1e3 : 0; }
    }
    double warps = (double)blocks * (kThreads / 32);
    double tflops = warps * n * kChains * flops_per_mma / (best * 1e-3) / 1e12;
    printf("{\"name\": \"%s\", \"tflops\": %.1f, \"sm_mhz\": %.0f, \"iters\": %d, \"ms\": %.2f}\n", name, tflops, bestmhz, n, best);
    fflush(stdout);
}

int main(int argc, char** argv) {
    double target_ms = argc > 1 ? atof(argv[1]) : 200.0;
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    int blocks = p.multiProcessorCount * 4;
    cudaStream_t st; cudaStreamCreate(&st);
    float* sink; long long* clk;
    cudaMalloc(&sink, kThreads * sizeof(float));
    cudaMalloc(&clk, 2 * sizeof(long long));
    fprintf(stderr, "device %s, %d SMs\n", p.name, p.multiProcessorCount);
    run<BF16_F32>("bf16_f32acc_m16n8k16", 16.0 * 8 * 16 * 2, blocks, sink, clk, st, target_ms);
    run<FP16_F32>("fp16_f32acc_m16n8k16", 16.0 * 8 * 16 * 2, blocks, sink, clk, st, target_ms);
    run<FP16_F16>("fp16_f16acc_m16n8k16", 16.0 * 8 * 16 * 2, blocks, sink, clk, st, target_ms);
    run<INT8_S32>("int8_s32acc_m16n8k32", 16.0 * 8 * 32 * 2, blocks, sink, clk, st, target_ms);
    run<E4M3_PLAIN_F32>("e4m3_plain_f32acc_m16n8k32", 16.0 * 8 * 32 * 2, blocks, sink, clk, st, target_ms);
    // repeat the bf16 one last: drift check (clock/power)
    run<BF16_F32>("bf16_f32acc_m16n8k16_repeat", 16.0 * 8 * 16 * 2, blocks, sink, clk, st, target_ms);
    return 0;
}
