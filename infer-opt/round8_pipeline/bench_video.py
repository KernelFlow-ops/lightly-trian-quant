"""Round 8 measurements on the full test video (/root/autodl-tmp/1.mp4, 126 frames).

  python bench_video.py --model vitl16
  ../common/profile_nsys.sh <prefix> bench_video.py --model vitl16 --profile
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import model_factory as mf  # noqa: E402
from bench import env_info, gpu_lock, write_results  # noqa: E402
import video_infer as vi  # noqa: E402

VIDEO = mf.VIDEO_PATH
OUT = HERE / "results.json"
TMP = Path("/root/autodl-tmp/infer-opt-tmp/postfuse/r8_out")
NFRAMES = 126
EVAL_IDX = [round(i * (NFRAMES - 1) / (mf.NUM_EVAL_FRAMES - 1)) for i in range(mf.NUM_EVAL_FRAMES)]


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def baseline_loop(orig, model, loops, collect):
    """decode + predict_head sequentially, exactly like infer_video (no overlay)."""
    dev = torch.device("cuda")
    lat, masks, confs = [], [], []
    n = 0
    t0 = time.perf_counter()
    for _ in range(loops):
        cap = cv2.VideoCapture(VIDEO)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            mask, score, conf = orig.predict_head(model, frame, dev, "bf16-mixed")
            lat.append((time.perf_counter() - t) * 1e3)
            if collect and n < NFRAMES:
                masks.append(mask.copy())
                confs.append(conf)
            n += 1
        cap.release()
    total = time.perf_counter() - t0
    return {"frames": n, "total_s": total, "fps": n / total, "predict_ms_p50": pct(lat, 50),
            "predict_ms_p90": pct(lat, 90)}, masks, confs


def pipe_run(pipe, loops, collect=False):
    masks, confs = {}, {}

    def consumer(idx, mask, conf, frame, slot, release):
        if collect and idx < NFRAMES:
            masks[idx] = mask.copy()
            confs[idx] = conf
        return False

    st = pipe.run([VIDEO], consumer, loops=loops)
    t = st["t_res"]
    k = NFRAMES if loops > 1 else 0
    lat = st["lat_ms"][k:]
    svc = st["svc_ms"][k:]
    res = {"frames": st["frames"], "total_s": st["t_total"], "fps_overall": st["frames"] / st["t_total"],
           "fps_steady": (len(t) - 1 - k) / (t[-1] - t[k]) if len(t) - 1 > k else None,
           "latency_ms_p50": pct(lat, 50), "latency_ms_p90": pct(lat, 90), "latency_ms_mean": float(np.mean(lat)),
           "service_ms_p50": pct(svc, 50), "service_ms_p90": pct(svc, 90)}
    return res, masks, confs


def compare(bmasks, bconfs, pmasks, pconfs):
    agree, ious, cd = [], [], []
    inter = union = 0
    for i in range(NFRAMES):
        a, b = bmasks[i], pmasks[i]
        agree.append(float((a == b).mean()))
        ha, hb = a == 1, b == 1
        it, un = int((ha & hb).sum()), int((ha | hb).sum())
        inter += it
        union += un
        ious.append(it / un if un else 1.0)
        cd.append(abs(bconfs[i] - pconfs[i]))
    return {"mask_agree_min": min(agree), "mask_agree_mean": float(np.mean(agree)),
            "head_iou_min": min(ious), "head_iou_pooled": inter / union if union else 1.0,
            "conf_abs_diff_max": max(cd), "conf_abs_diff_mean": float(np.mean(cd)), "frames": NFRAMES}


def fp32_gate(model_name, pmasks, pconfs):
    acc = {}
    for i, idx in enumerate(EVAL_IDX):
        ref = torch.load(mf.ref_path(model_name, i))
        m = pmasks[idx]
        rm = ref["mask"].numpy() == 1
        pm = m == 1
        un = int((rm | pm).sum())
        it = int((rm & pm).sum())
        acc[f"frame{i}"] = {"video_frame": idx, "head_inter": it, "head_union": un,
                            "head_iou": it / un if un else 1.0,
                            "pixel_agree": float((m == ref["mask"].numpy()).mean()),
                            "confidence_rel_diff": abs(pconfs[idx] - ref["confidence"]) / abs(ref["confidence"])}
    acc["gate"] = mf.gate(acc)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mf.WEIGHTS))
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--skip-overlay", action="store_true")
    args = ap.parse_args()
    orig = vi.load_orig_script()
    model = mf.load_model(args.model)
    mfn = vi.make_eager_model_fn(model)
    variant = "bf16-autocast+pipeline"
    if args.profile:
        pipe = vi.VideoPipeline(model, mfn, 1440, 1900, slots=4, inflight=2)
        with gpu_lock(), torch.inference_mode():
            pipe.run([VIDEO], lambda *a: False, loops=1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            pipe.run([VIDEO], lambda *a: False, loops=1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return
    res = {}
    with gpu_lock():
        print("== baseline (decode + predict_head)", flush=True)
        baseline_loop(orig, model, 1, False)  # warmup
        bres, bmasks, bconfs = baseline_loop(orig, model, 2, True)
        res["baseline"] = bres
        print(bres, flush=True)
        pipe = vi.VideoPipeline(model, mfn, 1440, 1900, slots=4, inflight=2)
        print("== pipeline inflight=2 (graphs)", flush=True)
        pipe_run(pipe, 1)
        r2, pm, pc = pipe_run(pipe, args.loops, collect=True)
        res["pipeline_inflight2"] = r2
        print(r2, flush=True)
        pipe.inflight = 3
        r3, _, _ = pipe_run(pipe, args.loops)
        res["pipeline_inflight3"] = r3
        print(r3, flush=True)
        pipe.inflight = 1
        r1, _, _ = pipe_run(pipe, args.loops)
        res["pipeline_inflight1"] = r1
        print(r1, flush=True)
        pipe.inflight = 2
        pipe.use_graph = False
        rng, _, _ = pipe_run(pipe, args.loops)
        res["pipeline_inflight2_nograph"] = rng
        print(rng, flush=True)
        pipe.use_graph = True
        if not args.skip_overlay:
            print("== overlay + video writer", flush=True)
            if TMP.exists():
                shutil.rmtree(TMP)
            (TMP / "base").mkdir(parents=True)
            (TMP / "fast").mkdir(parents=True)
            pipe6 = vi.VideoPipeline(model, mfn, 1440, 1900, slots=8, inflight=3)
            t = time.perf_counter()
            fr = vi.infer_video_fast(pipe6, orig, Path(VIDEO), TMP / "fast", 0.45)
            tf = time.perf_counter() - t
            res["overlay_fast"] = {"frames": fr["frames"], "total_s": tf, "fps": fr["frames"] / tf}
            t = time.perf_counter()
            orig.infer_video(model, Path(VIDEO), TMP / "base", torch.device("cuda"), "bf16-mixed", 0.45)
            tb = time.perf_counter() - t
            res["overlay_baseline"] = {"frames": NFRAMES, "total_s": tb, "fps": NFRAMES / tb}
            print(res["overlay_fast"], res["overlay_baseline"], flush=True)
            del pipe6
    acc_full = compare(bmasks, bconfs, pm, pc)
    gate = fp32_gate(args.model, pm, pc)
    print("full-video vs baseline:", acc_full, "\nfp32 gate:", gate["gate"], flush=True)
    fps = r2["fps_steady"] or r2["fps_overall"]
    payload = {
        "latency_ms": {"e2e": {"wall_p50": r1["service_ms_p50"], "wall_p90": r1["service_ms_p90"]}},
        "fps_e2e": fps,
        "accuracy_vs_fp32": gate,
        "video": {**res, "accuracy_vs_baseline_126f": acc_full,
                  "speedup_fps_vs_baseline": fps / res["baseline"]["fps"],
                  "note": "latency_ms.e2e = submit->host result with inflight=1 (single-frame service latency); pipeline_*.latency_ms = ring-buffer -> result incl. queueing; throughput = steady state after 1st loop"},
    }
    write_results(OUT, {args.model: {variant: payload}, "env": env_info()})


if __name__ == "__main__":
    main()
