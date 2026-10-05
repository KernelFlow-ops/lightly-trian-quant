// Declarations of the round12 attention extension (implemented in attn_fwd.cu / attn_bwd.cu / attn_misc.cu).
#pragma once
#include <torch/extension.h>
#include <optional>
#include <tuple>
#include <vector>

namespace a12 {
using at::Tensor;

// qkv [B,N,3,H,64] (or [B,N,3*H*64]) bf16 contiguous.  sinT/cosT fp32 [1024,64] (rope off when absent or prefix < 0).
// bits u32/i32 [B,mask_rows,32] (mask off when absent or mask_rows == 0): row r sees key mask_key0 + p iff bit p set.
// Returns (O [B,N,H*64] bf16, lse2 [B,H,N] fp32, log2 domain).
// With RoPE and save_rot the third output is qk_rot [B,N,2,H,64] (rotated q|k for the backward), else an empty tensor.
std::tuple<Tensor, Tensor, Tensor> attn_fwd(const Tensor& qkv, int64_t heads, const std::optional<Tensor>& sinT, const std::optional<Tensor>& cosT,
                                            const std::optional<Tensor>& bits, int64_t prefix, int64_t mask_rows, int64_t mask_key0, bool save_rot);

// Backward of attn_fwd: dO [B,N,H*64] bf16; returns dqkv [B,N,3,H,64] bf16 (inverse RoPE applied, scale included).
Tensor attn_bwd(const Tensor& qkv, const std::optional<Tensor>& qk_rot, const Tensor& O, const Tensor& lse2, const Tensor& dO, int64_t heads,
                const std::optional<Tensor>& sinT, const std::optional<Tensor>& cosT, const std::optional<Tensor>& bits, int64_t prefix, int64_t mask_rows,
                int64_t mask_key0);

// QueryBitmask builder: logits fp32 [B,Q,128,128] -> u32 [B,Q,32]; optional rnd [B,Q] / prob (fp32 device scalar) re-enable rows with rnd > prob.
Tensor build_bits(const Tensor& logits, const std::optional<Tensor>& rnd, const std::optional<Tensor>& prob);

}  // namespace a12
