"""Correctness of OptEoMT vs fp32 eager forward_backend / bf16-autocast baseline on real frames."""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import model_factory as mf
from check import tensor_diff
from opt_model import OptEoMT

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="vitl16"); a = ap.parse_args()
m = mf.load_model(a.model)
frame = mf.load_frames()[0]
with torch.inference_mode():
    batch, meta = mf.base_prep(m, frame)
    ref_m, ref_c = mf.base_model(m, batch)              # fp32 eager
    with mf.autocast_ctx("bf16-mixed"):
        ac_m, ac_c = mf.base_model(m, batch)            # baseline bf16-autocast
for cfg in [dict(), dict(head_channels_last=True), dict(head_convt_as_gemm=True)] + \
           [dict(sdpa_backend=b) for b in ("flash", "efficient", "cudnn", "math")]:
    o = OptEoMT(m, **cfg)
    with torch.inference_mode():
        om, oc = o.forward_backend(batch)
    print(cfg)
    print("  vs fp32   mask", {k: round(v, 5) for k, v in tensor_diff(om, ref_m).items() if k != 'nan'}, " cls", {k: round(v, 5) for k, v in tensor_diff(oc, ref_c).items() if k != 'nan'})
    print("  autocast vs fp32 mask", {k: round(v, 5) for k, v in tensor_diff(ac_m, ref_m).items() if k != 'nan'}, " cls", {k: round(v, 5) for k, v in tensor_diff(ac_c, ref_c).items() if k != 'nan'})
