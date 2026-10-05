"""Round 14 smoke: build one r14 variant on one model, run model_fn eagerly layer by layer with syncs (finds the failing op), report finite / cos vs r5_v1.
  python t_model_smoke.py vitb16 r14_nv4a"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R11 = HERE.parent.parent / "round11_attn_tier0"
R12 = HERE.parent.parent / "round12_lowp_stack"
for p in (R11, R11 / "impl", R12 / "impl", R12, HERE.parent, HERE.parent / "impl"):
    sys.path.insert(0, str(p))
import torch
import r11_common as c
import plug14
import stack14

model_name, variant = sys.argv[1], sys.argv[2]
with torch.inference_mode():
    frame = c.mf.load_frames()[0]
    model = c.mf.load_model(model_name)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xb, _ = c.mf.base_prep(model, frame)
    x = xb.to(torch.bfloat16).contiguous()
    fn, _ = plug14.make_variant(model_name, model, variant)
    # layer-by-layer sync check of the first call
    orig = stack14.block14a

    def checked(*a, **k):
        r = orig(*a, **k)
        torch.cuda.synchronize()
        return r

    stack14.block14a = checked
    out = fn(x)
    torch.cuda.synchronize()
    out2 = fn(x)
    torch.cuda.synchronize()
    from r5_model_impls import make_model_fn_r5
    ref_fn = make_model_fn_r5(model_name, model, impl="r5_v1")
    ref = ref_fn(x)
    torch.cuda.synchronize()
    ml, cl = out2
    mr, cr = ref
    cos = torch.nn.functional.cosine_similarity(ml.float().flatten().double(), mr.float().flatten().double(), dim=0).item()
    print(variant, "finite", bool(torch.isfinite(ml.float()).all()), "cos_vs_r5", cos, "mean|diff|", (ml.float() - mr.float()).abs().mean().item(),
          "mean|ref|", mr.float().abs().mean().item(), "first==second call", bool((ml == out[0]).all()))
