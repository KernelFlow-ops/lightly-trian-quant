"""Correctness of the FP8 kernels and real-FP8 model vs the fake-quant simulation."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import fp8linear as fl, fp8sim
from check import tensor_diff

e = fl.ext()
torch.manual_seed(0)
for M, K in [(2058, 1024), (2258, 768)]:
    x = (torch.randn(M, K, device="cuda") * 3)
    x[:, 5] *= 50  # outlier channel
    g = torch.randn(K, device="cuda"); b = torch.randn(K, device="cuda")
    q = torch.empty(M, K, dtype=fl.FP8, device="cuda"); s = torch.empty(M, 1, device="cuda")
    e.ln_quant(x, g, b, q, s, 1e-5)
    ln = F.layer_norm(x, (K,), g, b, 1e-5).bfloat16().float()
    ref = fp8sim.fake_quant(ln, "row")
    print("ln_quant", M, K, tensor_diff(q.float() * s, ref), "code mismatch frac", ((q.float() * s - ref).abs() > 1e-6 * ref.abs().clamp_min(1e-3)).float().mean().item())
    xb = torch.randn(M, 4096, device="cuda").bfloat16()
    q4 = torch.empty(M, 4096, dtype=fl.FP8, device="cuda"); s4 = torch.empty(M, 1, device="cuda")
    e.gelu_quant(xb, q4, s4)
    ref = fp8sim.fake_quant(F.gelu(xb.float()), "row")
    print("gelu_quant", tensor_diff(q4.float() * s4, ref))
    e.quant_rows(xb, q4, s4)
    print("quant_rows", tensor_diff(q4.float() * s4, fp8sim.fake_quant(xb, "row")))

import sens, model_factory as mf, pipeline as pl
for name in ("vitb16", "vitl16"):
    ev = sens.Evaluator(name)
    L = ev.L
    units = [(i, t) for i in range(L) for t in sens.TYPES if t in ("proj", "fc2", "fc1", "qkv")][: 4 * 6]
    for tag, u in [("first6-all4", units), ("all-mlp", [(i, t) for i in range(L) for t in ("fc1", "fc2")]), ("all-proj", [(i, "proj") for i in range(L)])]:
        sim = ev.eval_cfg(u, "row")
        fl.enable(ev.model, u)
        real = ev.run()
        fl.disable(ev.model)
        print(name, tag, "SIM ", {k: round(v, 5) if isinstance(v, float) else v for k, v in sim.items()})
        print(name, tag, "REAL", {k: round(v, 5) if isinstance(v, float) else v for k, v in real.items()}, flush=True)
