// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// Private interface between hgemm.cu and the dedicated decode-shape path in hgemm_decode.cu.
// Not part of the public API (include/spark/kernels.h): hgemm variant 3 dispatches here from
// launch_auto for small M, and nothing else should.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "spark/kernels.h"

namespace spark::hgemm_decode {

// True if the decode kernel takes this shape: 1 <= M <= 64, N % 64 == 0, K % 64 == 0.
bool supports(int M, int N, int K);

// C[M,N] = A[M,K] * B[K,N], bf16 in and out, fp32 accumulation, B streamed from DRAM once,
// with the fused epilogue of kernels.h applied by the storing thread (the last K-slice to
// arrive, when K is split). Asynchronous on `stream`. The caller has checked supports().
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream, const HgemmEpilogue& ep = HgemmEpilogue());

}  // namespace spark::hgemm_decode
