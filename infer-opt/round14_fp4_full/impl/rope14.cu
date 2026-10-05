// Round 14 (step 1): stand-alone RoPE + head-layout pass for a GEMM that already added the bias.
// Derived from round10_lowp_probe/impl/epi_kernels.cu::epi_rope_k (bias / scale arguments dropped, bf16 in, FP16 OUT for the r12 attention kernel).
//   out_f16[3,B,Ntok,H,64] = RoPE(y[m, which*d + h*64 + c])   y = bf16 [M, 3d] (bias already added), q/k rows with token index >= prefix get RoPE (fp32 math)
// One thread per (row, which, head, pair of columns c, c+1): partners c and c+32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__global__ void rope14_k(const __nv_bfloat16* __restrict__ y, const float* __restrict__ sin_t, const float* __restrict__ cos_t, __half* __restrict__ out,
                         int M, int Ntok, int prefix, int d, int H) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)M * 3 * H * 16;
    if (t >= total) return;
    int c = (t % 16) * 2; long r = t / 16;
    int h = r % H; r /= H;
    int which = r % 3; int m = r / 3;
    int col = which * d + h * 64 + c;
    long base = (long)m * 3 * d;
    __nv_bfloat162 lo2 = *reinterpret_cast<const __nv_bfloat162*>(y + base + col);
    __nv_bfloat162 hi2 = *reinterpret_cast<const __nv_bfloat162*>(y + base + col + 32);
    float lo0 = __bfloat162float(lo2.x), lo1 = __bfloat162float(lo2.y), hi0 = __bfloat162float(hi2.x), hi1 = __bfloat162float(hi2.y);
    int n = m % Ntok, pp = n - prefix;
    if (which < 2 && pp >= 0) {
        float2 cl = *reinterpret_cast<const float2*>(cos_t + (long)pp * 64 + c), sl = *reinterpret_cast<const float2*>(sin_t + (long)pp * 64 + c);
        float2 ch = *reinterpret_cast<const float2*>(cos_t + (long)pp * 64 + c + 32), sh = *reinterpret_cast<const float2*>(sin_t + (long)pp * 64 + c + 32);
        float n0 = lo0 * cl.x - hi0 * sl.x, n1 = lo1 * cl.y - hi1 * sl.y;
        float h0 = hi0 * ch.x + lo0 * sh.x, h1 = hi1 * ch.y + lo1 * sh.y;
        lo0 = n0; lo1 = n1; hi0 = h0; hi1 = h1;
    }
    int B = M / Ntok, b = m / Ntok;
    long o = ((((long)which * B + b) * Ntok + n) * H + h) * 64 + c;
    *reinterpret_cast<__half2*>(out + o) = __floats2half2_rn(lo0, lo1);
    *reinterpret_cast<__half2*>(out + o + 32) = __floats2half2_rn(hi0, hi1);
}

at::Tensor rope14(at::Tensor y, at::Tensor sin_t, at::Tensor cos_t, int64_t Ntok, int64_t prefix, int64_t d, int64_t H) {
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.size(1) == 3 * d, "y: bf16 [M, 3d]");
    int M = y.size(0);
    at::Tensor out = at::empty({3, M / Ntok, Ntok, H, 64}, y.options().dtype(at::kHalf));
    long total = (long)M * 3 * H * 16;
    rope14_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)y.data_ptr(), sin_t.data_ptr<float>(), cos_t.data_ptr<float>(),
                                                                                  (__half*)out.data_ptr(), M, Ntok, prefix, d, H);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rope14", &rope14); }
