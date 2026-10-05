"""torch.profiler kernel table of the three stage kernels (stage 2, ViT-L shapes)."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.bench import gpu_lock
from round14_head_sm120.impl import head_triton as ht
from torch.profiler import profile, ProfilerActivity

C, B, h = (int(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (1024, 7, 64)
PW = int(sys.argv[4]) if len(sys.argv) > 4 else None
NW = int(sys.argv[5]) if len(sys.argv) > 5 else None
w = h
with gpu_lock():
    Z = (torch.randn(B*h*w, 4*C, device="cuda")*2).bfloat16()
    bias = torch.randn(C, device="cuda")*0.1
    dwt = torch.randn(9, C, device="cuda")*0.3
    gam = 1+0.2*torch.randn(C, device="cuda"); bet = 0.1*torch.randn(C, device="cuda")
    dy = torch.randn(B*4*h*w, C, device="cuda").bfloat16()
    for _ in range(3):
        Y, mean, rstd = ht.fwd(Z, bias, dwt, gam, bet, B, h, w, 1e-5, PW, NW)
        du, a, b_ = ht.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, PW, NW)
        dZ, c_, d_ = ht.bwd2(Z, bias, dwt, du, B, h, w, PW, NW)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            Y, mean, rstd = ht.fwd(Z, bias, dwt, gam, bet, B, h, w, 1e-5, PW, NW)
            du, a, b_ = ht.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, PW, NW)
            dZ, c_, d_ = ht.bwd2(Z, bias, dwt, du, B, h, w, PW, NW)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
