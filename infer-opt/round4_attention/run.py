"""Round 4: whole-model replacement of SDPA by the TileLang attention (plain / RoPE-fused), eager + CUDA graph.

  python round4_attention/run.py --model vitl16 --variants sdpa-cudnn,tl,tl_rope --modes eager,graph
"""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "round1_torch_cleanup"; R2 = HERE.parent / "round2_cuda_graph"
for p in (HERE / "impl", R1, R2):
    sys.path.insert(0, str(p))
import torch
import harness, model_factory as mf
from graphed import GraphedModel
from opt_model import OptEoMT

OUT = HERE / "results.json"
VAR = {
    "sdpa-cudnn": dict(sdpa_backend="cudnn", attn_impl="sdpa"),
    "sdpa-flash": dict(sdpa_backend="flash", attn_impl="sdpa"),
    "tl": dict(attn_impl="tl"),
    "tl_rope": dict(attn_impl="tl_rope"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *mf.WEIGHTS])
    ap.add_argument("--variants", default="sdpa-cudnn,tl,tl_rope")
    ap.add_argument("--modes", default="eager,graph")
    ap.add_argument("--iters", type=int, default=100); ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    models = list(mf.WEIGHTS) if a.model == "all" else [a.model]
    for name in models:
        ref = mf.load_model(name, torch.float32)
        for v in a.variants.split(","):
            opt = OptEoMT(ref, **VAR[v])
            for mode in a.modes.split(","):
                fn = opt.forward_backend if mode == "eager" else GraphedModel(opt.forward_backend, (2, 3, 512, 512))
                harness.bench_variant(name, f"opt-{mode}-attn-{v}", ref, fn, OUT, iters=a.iters, warmup=a.warmup,
                                      extra={"config": {"attn": v, "mode": mode}})
        del ref; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
