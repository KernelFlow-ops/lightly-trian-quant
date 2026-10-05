"""Round 10 accuracy harness for the precision emulation (gate M = 8 eval frames, gate H = the other video frames, minus calibration frames).

  python emu_eval.py refs   --model vitl16                       # fp32 eager references for all 126 frames (orig path masks/conf for the gates; fast path lr logits for MAE)
  python emu_eval.py sweep  --model vitl16 --cfg cfgs/x.json --out sweeps/x.jsonl [--mode screen|full] [--time-limit 540]
  python emu_eval.py check  --model vitl16                       # P0 emulation == make_model_fn_r5 (bitwise) and its gates

Definitions
  eval frames   EVAL_IDX = round(i*125/7), i = 0..7                          (gate M: pooled head IoU >= 0.95, every frame conf rel diff <= 5%, vs cached fp32 refs `ref/`)
  calib frames  CAL_IDX  = 16 frames not in EVAL_IDX                          (SmoothQuant / GPTQ / static scales)
  gate H        all 126 frames minus CAL_IDX (110 frames), same two criteria  vs fp32 eager (orig prep/post path); also min-frame IoU, #frames conf > 3 %, logit MAE
  logit MAE     mean |lr_cand - lr_ref| over the (2, 512, 675) untiled head logits of the fast post path (R7 FastPost), fp32 eager reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/lightly-train/infer-opt")
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe")
for p in (HERE / "impl", ROOT, ROOT / "common", ROOT / "round7_head_postprocess", ROOT / "round7_head_postprocess" / "impl", ROOT / "round5_gemm_sm120"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2  # noqa: E402

import model_factory as mf  # noqa: E402
import postfuse as pf  # noqa: E402

NFR = 126
EVAL_IDX = [round(i * (NFR - 1) / (mf.NUM_EVAL_FRAMES - 1)) for i in range(mf.NUM_EVAL_FRAMES)]
_cal = [round(x) for x in np.linspace(3, NFR - 4, 16)]
CAL_IDX = []
for c in _cal:
    while c in EVAL_IDX or c in CAL_IDX:
        c += 1
    CAL_IDX.append(c)
H_IDX = [i for i in range(NFR) if i not in CAL_IDX]
CAL_ALT = [min(i + 2, NFR - 1) if (i + 2) not in EVAL_IDX else i + 1 for i in CAL_IDX]      # second calibration set (robustness check of SmoothQuant / GPTQ plans)


def read_frames():
    cap = cv2.VideoCapture(mf.VIDEO_PATH)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    assert len(out) == NFR, len(out)
    return out


def ref_file(name):
    return TMP / f"ref126_{name}.pt"


# ------------------------------------------------------------------------------------------------------ references
def make_refs(name: str):
    frames = read_frames()
    m32 = mf.load_model(name, torch.float32)
    crop = min(m32.image_size)
    norm = m32.image_normalize
    prep32 = pf.FastPrep(norm["mean"], norm["std"], crop, torch.float32)
    post = pf.FastPost(m32.internal_class_to_class.tolist(), mf.HEAD_CLASS_ID, crop)
    bits_o, conf_o, bits_f, conf_f, lrs = [], [], [], [], []
    t0 = time.time()
    for idx, fr in enumerate(frames):
        with torch.inference_mode():
            batch, meta = mf.base_prep(m32, fr)
            raw = m32.forward_backend(batch)
            logits = mf.base_post_logits(m32, raw, meta).float()
            mask, _, conf = mf.host_finalize(m32, logits.cpu().numpy())
            bits_o.append(np.packbits((mask == 1).reshape(-1)))
            conf_o.append(conf)
            b32, meta2 = prep32(fr)
            ml, cl = m32.forward_train(b32, return_logits_per_layer=False)
            mask_f, conf_f_ = post(ml[-1], cl[-1], meta2)
            bits_f.append(np.packbits((mask_f == 1).reshape(-1)))
            conf_f.append(conf_f_)
            lrs.append(post._state(meta2)["lr"].clone().cpu())
        if idx % 25 == 0:
            print(name, "ref frame", idx, f"{time.time() - t0:.0f}s", flush=True)
    # consistency with the cached 8-frame refs
    cache_ok = []
    for k, idx in enumerate(EVAL_IDX):
        r = torch.load(mf.ref_path(name, k))
        rm = (r["mask"].numpy() == 1).reshape(-1)
        mine = np.unpackbits(bits_o[idx])[: rm.size].astype(bool)
        cache_ok.append((bool((mine == rm).all()), abs(conf_o[idx] - r["confidence"])))
    print("orig-path refs vs cached refs/ (mask identical, |conf diff|):", cache_ok, flush=True)
    torch.save({"bits_o": np.stack(bits_o), "conf_o": conf_o, "bits_f": np.stack(bits_f), "conf_f": conf_f, "lr": torch.stack(lrs),
                "hw": tuple(mask.shape), "cache_check": cache_ok}, ref_file(name))
    print("saved", ref_file(name), flush=True)


# ------------------------------------------------------------------------------------------------------ metrics
def summarize(rows: dict, idxs: list[int]) -> dict:
    r = [rows[i] for i in idxs]
    inter = sum(x["inter"] for x in r)
    union = sum(x["union"] for x in r)
    cr = np.array([x["conf_rel"] for x in r])
    pooled = inter / union if union else 1.0
    return {"n": len(r), "pooled_iou": pooled, "min_iou": min(x["iou"] for x in r), "conf_max": float(cr.max()), "conf_p99": float(np.percentile(cr, 99)),
            "n_conf_gt3": int((cr > 0.03).sum()), "n_conf_gt5": int((cr > 0.05).sum()), "logit_mae": float(np.mean([x["mae"] for x in r])),
            "lr_cos": float(np.mean([x["cos"] for x in r])), "pass": bool(pooled >= mf.GATE_HEAD_IOU and cr.max() <= mf.GATE_CONF_REL)}


class Harness:
    def __init__(self, name: str, need_refs: bool = True):
        self.name = name
        self.frames = read_frames()
        self.model = mf.load_model(name, torch.float32)
        crop = min(self.model.image_size)
        norm = self.model.image_normalize
        self.prep = pf.FastPrep(norm["mean"], norm["std"], crop, torch.bfloat16)
        self.post = pf.FastPost(self.model.internal_class_to_class.tolist(), mf.HEAD_CLASS_ID, crop)
        self.ref = None
        if need_refs:
            R = torch.load(ref_file(name), weights_only=False)
            hw = R["hw"]
            n = hw[0] * hw[1]
            self.ref_head = np.stack([np.unpackbits(b)[:n].astype(bool) for b in R["bits_o"]])
            self.ref_conf = np.array(R["conf_o"])
            self.ref_lr = R["lr"].cuda()
            self.ref = R
        from emu import EmuNet
        self.net = EmuNet(name, self.model)

    def run(self, idxs: list[int]) -> dict:
        rows = {}
        with torch.inference_mode():
            for idx in idxs:
                b, meta = self.prep(self.frames[idx])
                ml, cl = self.net.forward(b)
                mask, conf = self.post(ml, cl, meta)
                lr = self.post._state(meta)["lr"]
                head = (mask.reshape(-1) == 1)
                rh = self.ref_head[idx]
                inter, union = int((head & rh).sum()), int((head | rh).sum())
                d = (lr - self.ref_lr[idx]).abs().mean().item()
                cs = torch.nn.functional.cosine_similarity(lr.flatten().double(), self.ref_lr[idx].flatten().double(), dim=0).item()
                rows[idx] = {"inter": inter, "union": union, "iou": inter / union if union else 1.0,
                             "conf_rel": abs(conf - self.ref_conf[idx]) / abs(self.ref_conf[idx]), "conf": conf, "mae": d, "cos": cs}
        return rows

    def eval_plan(self, plan: dict, mode: str = "screen") -> dict:
        self.net.set_plan(plan)
        t0 = time.time()
        idxs = EVAL_IDX if mode == "screen" else list(range(NFR))
        rows = self.run(idxs)
        out = {"M": summarize(rows, EVAL_IDX)}
        if mode == "full":
            out["H"] = summarize(rows, H_IDX)
            out["all126"] = summarize(rows, list(range(NFR)))
            out["per_frame"] = {"iou": [rows[i]["iou"] for i in range(NFR)], "conf_rel": [rows[i]["conf_rel"] for i in range(NFR)],
                                "mae": [rows[i]["mae"] for i in range(NFR)]}
        out["sec"] = round(time.time() - t0, 1)
        return out

    def calibrate(self, hess_kinds=(), idxs=None):
        from emu import Calib, P0
        calib = Calib()
        net = self.net
        net.set_plan({})

        def hook(i, kind, X):
            B, N, K = X.shape
            pm = net.prefix_mask(B, N, X.device)
            calib.update((i, kind), X.reshape(B * N, K), pm, hess=kind in hess_kinds)
        net.collect = hook
        self.run(CAL_IDX if idxs is None else idxs)
        net.collect = None
        net.calib = calib
        net._wq.clear(); net._wbytes = 0; net._lay.clear()      # cached quantised weights / SmoothQuant folds depend on the calibration
        return calib


# ------------------------------------------------------------------------------------------------------ sweep runner
def run_sweep(args):
    from emu import make_plan
    h = Harness(args.model)
    cfgs = json.loads(Path(args.cfg).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"] + "|" + json.loads(line)["mode"])
    need_cal = any(s["spec"].get("sq", 0) or s["spec"].get("gptq") for c in cfgs for s in c["entries"])
    need_h = any(s["spec"].get("gptq") for c in cfgs for s in c["entries"])
    if need_cal:
        t0 = time.time()
        gk = tuple(("qkv", "proj", "fc1", "fc2")) if need_h else ()
        h.calibrate(hess_kinds=gk)
        if need_h:
            import gptq
            gptq.install(h.net)
        print(f"calibrated on {len(CAL_IDX)} frames in {time.time() - t0:.0f}s", flush=True)
    t_start = time.time()
    for c in cfgs:
        mode = c.get("mode", args.mode)
        if c["name"] + "|" + mode in done:
            continue
        if time.time() - t_start > args.time_limit:
            print("time limit reached; resume later (finished configs are skipped)", flush=True)
            break
        plan = make_plan(c["entries"], h.net.nb)
        try:
            res = h.eval_plan(plan, mode)
        except Exception as ex:  # noqa: BLE001
            import traceback
            res = {"error": traceback.format_exc()[-600:]}
            print("FAILED", c["name"], res["error"], flush=True)
        rec = {"name": c["name"], "model": args.model, "mode": mode, "entries": c["entries"], "meta": c.get("meta", {}), **res}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        m = res.get("M", {})
        hh = res.get("H", {})
        print(f"{c['name']:48s} [{mode}] M: iou={m.get('pooled_iou', float('nan')):.4f} conf={m.get('conf_max', float('nan')) * 100:5.2f}% mae={m.get('logit_mae', float('nan')):.2e} pass={m.get('pass')}"
              + (f" | H: iou={hh['pooled_iou']:.4f} min={hh['min_iou']:.3f} conf={hh['conf_max'] * 100:5.2f}% >3%:{hh['n_conf_gt3']} mae={hh['logit_mae']:.2e} pass={hh['pass']}" if hh else "")
              + f"  ({res.get('sec')}s)", flush=True)


def check_p0(args):
    """P0 emulation vs make_model_fn_r5 (same kernels): bitwise equal outputs and the gates."""
    from emu import make_plan
    from round5_gemm_sm120.r5_model_impls import make_model_fn_r5
    h = Harness(args.model)
    h.net.set_plan({})
    x = torch.randn(2, 3, 512, 512, device="cuda").bfloat16()
    with torch.inference_mode():
        a = h.net.forward(x)
        fn = make_model_fn_r5(args.model, h.model, "r5_v1")        # re-patches model._predict with the R7 head (hd.enable)
        b = fn(x)
        h.model._predict = h.net._head_predict                     # restore the emulation head
    for k, (u, v) in enumerate(zip(a, b)):
        print("output", k, "max_abs diff emu vs r5_v1:", (u.float() - v.float()).abs().max().item(), "identical:", bool(torch.equal(u, v)))
    res = h.eval_plan({}, "full")
    print("P0 emulation  M:", res["M"], "\n              H:", res["H"], "\n              all126:", res["all126"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refs", "sweep", "check"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg")
    ap.add_argument("--out")
    ap.add_argument("--mode", default="screen")
    ap.add_argument("--time-limit", type=float, default=540)
    a = ap.parse_args()
    if a.cmd == "refs":
        make_refs(a.model)
    elif a.cmd == "sweep":
        run_sweep(a)
    else:
        check_p0(a)
