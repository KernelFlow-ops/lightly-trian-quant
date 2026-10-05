import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
import torch, ops
torch.manual_seed(0)
B, H = 2, 16
sin = torch.randn(1024, 64, device="cuda"); sin[:, 32:] = sin[:, :32]
cos = torch.randn(1024, 64, device="cuda"); cos[:, 32:] = cos[:, :32]
for N in (1029, 1129):
    a = torch.randn(B, N, 3, H, 64, device="cuda", dtype=torch.bfloat16) * 2
    b = a.permute(2, 0, 1, 3, 4).contiguous()  # [3,B,N,H,64]
    for name, r1, r2 in (("plain", ops.attention_qkv(a), ops.attention_3bnhd(b)),
                         ("rope", ops.attention_rope_qkv(a, sin, cos, N - 1024), ops.attention_3bnhd(b, (sin, cos, N - 1024)))):
        print(N, name, "identical to [B,N,3,H,D] kernel:", torch.equal(r1, r2))
