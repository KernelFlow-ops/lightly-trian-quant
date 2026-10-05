"""Per-backend numerical error of SDPA on the REAL q/k/v of ViT-L frame 7 (the gate-sensitive frame)."""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round1_torch_cleanup"); sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch, torch.nn.functional as F, model_factory as mf
from torch.nn.attention import SDPBackend, sdpa_kernel
import opt_model
from opt_model import OptEoMT
m = mf.load_model("vitl16")
o = OptEoMT(m, sdpa_backend="cudnn")
cap = []
orig = F.scaled_dot_product_attention
def spy(q, k, v, *a, **kw):
    cap.append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
    return orig(q, k, v, *a, **kw)
opt_model.F.scaled_dot_product_attention = spy
fr = mf.load_frames()[7]
with torch.inference_mode():
    b, _ = mf.base_prep(m, fr)
    o.forward_lowres(b)
opt_model.F.scaled_dot_product_attention = orig
print("captured", len(cap), "attention calls")
be = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
rel = {k: [] for k in be}
with torch.inference_mode():
    for (q, k, v) in cap:
        ref = orig(q.double(), k.double(), v.double()).float()
        for n, b_ in be.items():
            with sdpa_kernel(b_):
                out = orig(q, k, v).float()
            rel[n].append(((out - ref).norm() / ref.norm()).item())
for n in be:
    r = torch.tensor(rel[n])
    print(f"{n:10s} rel-L2 vs fp64 SDPA  mean {r.mean():.3e}  max {r.max():.3e}  last-3-blocks {[f'{x:.2e}' for x in r[-3:].tolist()]}")
