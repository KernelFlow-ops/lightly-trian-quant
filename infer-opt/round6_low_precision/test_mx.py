"""Numerical checks for real MXFP8 quantization, swizzled scales, and GEMM."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "impl"))
import mxlinear as mx  # noqa: E402


def reference(x):
    M, K = x.shape
    blocks = x.float().reshape(M, K // 32, 32)
    scales = torch.exp2(torch.ceil(torch.log2((blocks.abs().amax(-1) / 448).clamp_min(1e-30))))
    q = (blocks / scales[..., None]).clamp(-448, 448).to(mx.FP8).reshape(M, K)
    nr, nc = (M + 127) // 128, K // 128
    padded = torch.ones((nr * 128, nc * 4), device=x.device)
    padded[:M] = scales
    swizzled = padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    swizzled = swizzled.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().to(mx.SCALE)
    return q, swizzled, scales


def check_quant(x, kernel):
    q, s = mx.empty_quant(*x.shape, x.device)
    kernel(x, q, s)
    qr, sr, plain = reference(x)
    q_mismatch = (q.view(torch.uint8) != qr.view(torch.uint8)).float().mean().item()
    s_mismatch = (s.view(torch.uint8) != sr.view(torch.uint8)).float().mean().item()
    assert q_mismatch == 0, q_mismatch
    assert s_mismatch == 0, s_mismatch
    return q, s, plain


def check_fused(x, mode):
    q, s = mx.empty_quant(*x.shape, x.device)
    if mode == "ln":
        g = torch.randn(x.shape[1], device=x.device)
        b = torch.randn(x.shape[1], device=x.device)
        mx.ext().ln_quant(x, g, b, q, s, 1e-5)
        ref_input = F.layer_norm(x, (x.shape[1],), g, b, 1e-5).bfloat16()
    else:
        mx.ext().gelu_quant(x, q, s)
        ref_input = F.gelu(x.float())
    qr, sr, _ = reference(ref_input)
    q_mismatch = (q.view(torch.uint8) != qr.view(torch.uint8)).float().mean().item()
    s_mismatch = (s.view(torch.uint8) != sr.view(torch.uint8)).float().mean().item()
    print(mode, x.shape, "q code mismatch", q_mismatch, "scale mismatch", s_mismatch)
    assert q_mismatch < (0.004 if mode == "ln" else 0.0001)
    assert s_mismatch < (0.001 if mode == "ln" else 0.0001)


def main():
    torch.manual_seed(1)
    for M, K in ((160, 768), (2058, 1024), (2258, 4096)):
        x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
        x[:, 5] *= 50
        q, s, plain = check_quant(x, mx.ext().quant_rows)
        print("quant", x.shape, "exact FP8/scales")
        if K <= 1024:
            check_fused(x.float(), "ln")
        check_fused(x, "gelu")
        if K == 1024:
            w = torch.randn(3072, K, device="cuda").bfloat16()
            qw, sw = mx.quant_weight(w)
            y = torch._scaled_mm(q, qw.t(), scale_a=s, scale_b=sw, out_dtype=torch.bfloat16)
            _, _, ws = reference(w)
            dq = (q.float().reshape(M, K // 32, 32) * plain[..., None]).reshape(M, K)
            dw = (qw.float().reshape(w.shape[0], K // 32, 32) * ws[..., None]).reshape(w.shape)
            ref = dq @ dw.t()
            rel_mae = ((y.float() - ref).abs().mean() / ref.abs().mean()).item()
            print("_scaled_mm vs dequant fp32 GEMM rel MAE", rel_mae)
            assert rel_mae < 0.003


if __name__ == "__main__":
    main()
