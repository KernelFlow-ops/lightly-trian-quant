"""Gate A for the mg_i8 prototype: int8 GEMM + dequant + R5 epilogues vs an fp64 reference of the same int8 operands / scales.
  python tests/test_mg8.py [--model vitl16] [--M 2058]
The int32 accumulation is exact, so the only differences are the fp32 epilogue arithmetic and the bf16 output rounding."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE / "tests"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import bench10 as b  # noqa: E402
import cands_base as cb  # noqa: E402
import mg8  # noqa: E402
from test_mg16 import err, rope_ref  # noqa: E402

BF = torch.bfloat16
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitl16")
ap.add_argument("--M", type=int, default=2058)
ap.add_argument("--kinds", default="qkv,proj,fc1,fc2")
a = ap.parse_args()
torch.manual_seed(0)
B, Nt = cb._batch(a.M)
d, H = b.MODELS[a.model]["d"], b.MODELS[a.model]["H"]
prefix = Nt - 1024
out = {}
for kind in a.kinds.split(","):
    N, K = b.kind_nk(a.model, kind)
    X = torch.randn(a.M, K, device="cuda") * 0.7
    X[:, :4] *= 8
    W = torch.randn(N, K, device="cuda") * 0.03
    X8, sx = mg8.quant_rows(X)
    W8, sw = mg8.quant_rows(W)
    bias = (torch.randn(N, device="cuda") * 0.1).to(BF)
    res = torch.randn(B, Nt, N, device="cuda")
    gamma = torch.rand(N, device="cuda") + 0.1
    sin = torch.randn(1024, 64, device="cuda")
    cos = torch.randn(1024, 64, device="cuda")
    y = (X8.double() @ W8.double().t()) * sx.double()[:, None] * sw.double()[None, :] + bias.double()    # exact dequantised product + bias
    aux = dict(sin=sin, cos=cos, prefix=prefix, res=res, gamma=gamma)
    o = mg8.run_kind(kind, X8, sx, W8, sw, bias, aux=aux, B=B, heads=H)
    if kind == "fc1":
        ref = F.gelu(y)
    elif kind in ("proj", "fc2"):
        ref = res.view(-1, N).double() + gamma.double() * y
        o = o.view(-1, N)
    else:
        ref = rope_ref(y.float(), sin, cos, B, Nt, d, H, prefix).double()
        o = o.transpose(2, 3)
    e = err(o.float().view(ref.shape), ref)
    out[kind] = e
    print(f"{a.model} {kind:5s} M={a.M}: vs exact dequantised int8 product: rel_l2={e['rel_l2']:.2e} max_abs={e['max_abs']:.2e} cos={e['cos']:.9f}", flush=True)
b.merge_results(HERE / "results.json", {"mg8_test": {f"{a.model}|M{a.M}": out}})
