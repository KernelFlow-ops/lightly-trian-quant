"""Can block-scaled FP8 (MXFP8, 1x32 E8M0 scales) / NVFP4 GEMMs run on sm_120 through torch._scaled_mm?"""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from bench import time_fn

def ceil_div(a, b): return (a + b - 1) // b
def to_blocked(x):
    rows, cols = x.shape
    nr, nc = ceil_div(rows, 128), ceil_div(cols, 4)
    padded = torch.zeros(nr * 128, nc * 4, dtype=x.dtype, device=x.device)
    padded[:rows, :cols] = x
    return padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()

M, N, K = 2048, 4096, 1024
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda"); w = torch.randn(N, K, device="cuda")

def mx_quant(t, bs=32):
    tb = t.reshape(t.shape[0], -1, bs)
    amax = tb.abs().amax(-1)
    s = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-30))))
    q = (tb / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(t.shape)
    return q, s.to(torch.float8_e8m0fnu), (q.float().reshape(tb.shape) * s[..., None]).reshape(t.shape)

qx, sx, dx = mx_quant(x); qw, sw, dw = mx_quant(w)
ref = dx @ dw.t()
try:
    out = torch._scaled_mm(qx, qw.t(), scale_a=to_blocked(sx), scale_b=to_blocked(sw), out_dtype=torch.bfloat16)
    err = ((out.float() - ref).abs().mean() / ref.abs().mean()).item()
    print("MXFP8 _scaled_mm OK, rel err vs dequant ref:", err)
    t = time_fn(lambda: torch._scaled_mm(qx, qw.t(), scale_a=to_blocked(sx), scale_b=to_blocked(sw), out_dtype=torch.bfloat16), warmup=10, iters=50)
    sa, sb = to_blocked(sx), to_blocked(sw)
    t2 = time_fn(lambda: torch._scaled_mm(qx, qw.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), warmup=10, iters=50)
    print(f"MXFP8 GEMM {M}x{N}x{K}: {t2['gpu_ms_p50']*1e3:.1f} us  ({2*M*N*K/t2['gpu_ms_p50']/1e9:.0f} TFLOPS)")
    row = torch._scaled_mm(qx, qw.t(), scale_a=torch.ones(M,1,device='cuda'), scale_b=torch.ones(1,N,device='cuda'), out_dtype=torch.bfloat16)
    t3 = time_fn(lambda: torch._scaled_mm(qx, qw.t(), scale_a=torch.ones(M,1,device='cuda'), scale_b=torch.ones(1,N,device='cuda'), out_dtype=torch.bfloat16), warmup=10, iters=50)
    print(f"rowwise FP8 (same shape, incl. 2 tiny ones() allocs): {t3['gpu_ms_p50']*1e3:.1f} us")
except Exception as e:  # noqa: BLE001
    print("MXFP8 _scaled_mm FAILED:", str(e)[:400])

try:
    if hasattr(torch, "float4_e2m1fn_x2"):
        a4 = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
        b4 = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
        s_a = torch.ones(M, K // 16, device="cuda").to(torch.float8_e4m3fn)
        s_b = torch.ones(N, K // 16, device="cuda").to(torch.float8_e4m3fn)
        o = torch._scaled_mm(a4, b4.t(), scale_a=to_blocked(s_a), scale_b=to_blocked(s_b), out_dtype=torch.bfloat16)
        print("NVFP4 _scaled_mm OK", o.shape)
        sa4, sb4 = to_blocked(s_a), to_blocked(s_b)
        t4 = time_fn(lambda: torch._scaled_mm(a4, b4.t(), scale_a=sa4, scale_b=sb4, out_dtype=torch.bfloat16), warmup=10, iters=50)
        print(f"NVFP4 GEMM {M}x{N}x{K}: {t4['gpu_ms_p50']*1e3:.1f} us ({2*M*N*K/t4['gpu_ms_p50']/1e9:.0f} TFLOPS)")
except Exception as e:  # noqa: BLE001
    print("NVFP4 _scaled_mm FAILED:", str(e)[:400])
