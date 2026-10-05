"""P3 correctness + microbench: fused head vs model._predict (bf16 autocast)."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, str(HERE.parent / "common"))
import model_factory as mf
from check import tensor_diff
from bench import time_fn, write_results
import head as hd

for name in ("vitl16", "vitb16"):
    m = mf.load_model(name)
    d = m.backbone.embed_dim
    fh = hd.FastHead(m)
    torch.manual_seed(0)
    # realistic tokens: run the real backbone up to the final norm on frame 0
    fr = mf.load_frames()[0]
    with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
        batch, meta = mf.base_prep(m, fr)
        grab = {}
        h = m.backbone.norm.register_forward_hook(lambda mod, a, o: grab.__setitem__("x", o))
        m.forward_train(batch, return_logits_per_layer=False); h.remove()
        x = grab["x"]
        ref_ml, ref_cl = m._predict(x, (32, 32))
        got_ml, got_cl = fh.predict(x, (32, 32))
    print(name, "mask_logits", {k: round(v, 6) if isinstance(v, float) else v for k, v in tensor_diff(got_ml, ref_ml).items()})
    print(name, "class_logits", tensor_diff(got_cl, ref_cl)["max_abs"])
    print(name, "logit>0 agreement", float(((got_ml > 0) == (ref_ml > 0)).float().mean()), "ref |logit| mean", float(ref_ml.float().abs().mean()))
    def f_ref():
        with mf.autocast_ctx("bf16-mixed"): return m._predict(x, (32, 32))
    def f_new():
        with mf.autocast_ctx("bf16-mixed"): return fh.predict(x, (32, 32))
    tr, tn = time_fn(f_ref, warmup=20, iters=100), time_fn(f_new, warmup=20, iters=100)
    print(name, f"_predict head: torch {tr['gpu_ms_p50']:.3f} ms gpu / {tr['wall_ms_p50']:.3f} wall | fused {tn['gpu_ms_p50']:.3f} ms gpu / {tn['wall_ms_p50']:.3f} wall", flush=True)
    write_results(HERE / "results_head.json", {name: {"predict_torch_gpu_ms": tr["gpu_ms_p50"], "predict_fused_gpu_ms": tn["gpu_ms_p50"],
        "predict_torch_wall_ms": tr["wall_ms_p50"], "predict_fused_wall_ms": tn["wall_ms_p50"],
        "mask_logits_diff": tensor_diff(got_ml, ref_ml)}})
    del m
