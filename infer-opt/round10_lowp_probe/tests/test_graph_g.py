"""Gate G for the prototypes: CUDA-graph replay of mg_f16acc / mg_i8 GEMMs (all epilogues) is bitwise equal to eager.   python tests/test_graph_g.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "impl"))
import torch
import bench10 as b, cands_base as cb, mg16, mg8
BF = torch.bfloat16
ok = True
res = {}
for model in ("vitl16", "vitb16"):
    for kind in ("qkv", "proj", "fc1", "fc2"):
        N, K = b.kind_nk(model, kind)
        M = 2058
        B, Nt = cb._batch(M); H = b.MODELS[model]["H"]
        aux = dict(sin=torch.randn(1024, 64, device="cuda"), cos=torch.randn(1024, 64, device="cuda"), prefix=Nt - 1024, res=torch.randn(B, Nt, N, device="cuda"), gamma=torch.rand(N, device="cuda"))
        bias = (torch.randn(N, device="cuda") * 0.1).to(BF)
        X16 = (torch.randn(M, K, device="cuda") * 0.5).half(); W16 = (torch.randn(N, K, device="cuda") * 0.03).half()
        X8, sx = mg8.quant_rows(torch.randn(M, K, device="cuda")); W8, sw = mg8.quant_rows(torch.randn(N, K, device="cuda"))
        for tag, fn in (("mg16", lambda: mg16.run_kind(kind, X16, W16, bias, aux=aux, B=B, heads=H)), ("mg8", lambda: mg8.run_kind(kind, X8, sx, W8, sw, bias, aux=aux, B=B, heads=H))):
            ref = fn().clone(); torch.cuda.synchronize()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2): fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fn()
            g.replay(); torch.cuda.synchronize()
            eq = bool(torch.equal(out, ref))
            ok &= eq
            res[f"{tag}|{model}|{kind}"] = eq
print("G (graph == eager, bitwise):", "ALL EQUAL" if ok else "MISMATCH", res)
b.merge_results(HERE / "results.json", {"gate_G_prototypes": {"all_equal": bool(ok), "cases": res}})
