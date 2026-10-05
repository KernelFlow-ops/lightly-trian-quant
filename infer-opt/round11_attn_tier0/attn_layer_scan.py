"""Per-layer operator error of every attention candidate on the REAL qkv of all layers (flash trajectory), vs fp64 attention of the same bf16 q/k/v.

  python attn_layer_scan.py --model vitl16 --frames 0,7 --cands flash,sage2_fp8,sage2_pv16,spark_fp8,a16k3,a16k4,sage3
Output: profile/attn_layer_scan_<model>.json  { "L<i>_f<frame>": {cand: {rel_l2, max_abs, finite, S_absmax}} }
"""
import argparse, json
import torch
import r11_common as c
import r11_model as rm
import attn_cands as ac
from lock2 import gpu_lock2

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--frames", default="0,7")
ap.add_argument("--cands", default="flash,sage2_fp8,sage2_pv16,spark_fp8,a16k3,a16k4,sage3")
a = ap.parse_args()
cands = a.cands.split(",")
splits = {n: ac.get_split(n) for n in cands}
frames = c.mf.load_frames()


def ref_attn(qkv):
    q, k, v = [t.double().permute(0, 2, 1, 3) for t in (qkv[0], qkv[1], qkv[2])]
    out, smax = [], 0.0
    for b in range(q.shape[0]):
        s = q[b] @ k[b].transpose(-1, -2) * 0.125
        smax = max(smax, s.abs().max().item())
        out.append(torch.softmax(s, -1) @ v[b])
    return torch.stack(out).permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1), smax


res = {}
cur = {"f": None}


def hook(qkv, B, N, d, layer=None):
    qc = qkv.contiguous()
    ref, smax = ref_attn(qc)
    row = {"S_scaled_absmax_x8": smax / 8 * 8, "N": N}
    for n, (prep, kern) in splits.items():
        try:
            o = kern(prep(qc))
            dlt = o.double() - ref
            row[n] = dict(rel_l2=(dlt.norm() / ref.norm()).item(), max_abs=dlt.abs().max().item(), finite=bool(torch.isfinite(o.float()).all()))
        except Exception as e:  # noqa: BLE001
            row[n] = dict(error=f"{type(e).__name__}: {str(e)[:80]}")
    res[f"L{layer}_f{cur['f']}"] = row
    return rm.attn_flash(qkv, B, N, d)


with gpu_lock2(), torch.inference_mode():
    model = c.mf.load_model(a.model)
    on, flags = rm.parse("r11")
    flags["attn"] = hook
    fn = rm.make_model_fn_r11(a.model, model, on, flags)
    pipe = c.make_pipeline(model, rm.make_model_fn_r11(a.model, model, on, dict(flags, attn="flash")), rm.Pipeline11, slots=1)
    for fi in [int(x) for x in a.frames.split(",")]:
        cur["f"] = fi
        pipe.pinned_np[0][...] = frames[fi]
        pipe.frame_gpu[0].copy_(pipe.pinned[0])
        pipe._prep_launch(0)
        fn(pipe.batch)
        print("frame", fi, "done", flush=True)
json.dump(res, open(str(c.HERE / "profile" / f"attn_layer_scan_{a.model}.json"), "w"), indent=1)
bad = {n: [(k, v[n].get("rel_l2")) for k, v in res.items() if n in v and (not v[n].get("finite", True) or v[n].get("rel_l2", 0) > 0.03)] for n in cands}
print({n: b[:6] for n, b in bad.items()})
