"""Real-FP8 (+fused head) accuracy of greedy prefixes on all 126 frames vs fp32 eager, plus the 8-frame gate.

  python heldout_real.py --model vitl16 --json greedy_vitl16_row_noproj.json --ns 0,6,10,14,18,24
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "impl"))
import sens, greedy, fp8linear as fl, head as hd  # noqa: E402
import model_factory as mf, pipeline as pl  # noqa: E402
from bench import write_results  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--json", required=True); ap.add_argument("--ns", required=True)
ap.add_argument("--no-head", action="store_true")
a = ap.parse_args()
ev = sens.Evaluator(a.model)
frames = greedy.frames_all()
m32 = mf.load_model(a.model); pl.init(m32)
ref = greedy.masks_conf(m32, "32-true", frames)
del m32
pl.init(ev.model)
order = [tuple(u) for u in json.loads(Path(a.json).read_text())["order"]]
if not a.no_head:
    hd.enable(ev.model)
rows = []
for n in [int(x) for x in a.ns.split(",")]:
    units = order[:n]
    if units: fl.enable(ev.model, units)
    g8 = ev.run()
    cand = greedy.masks_conf(ev.model, "bf16-mixed", frames)
    fl.disable(ev.model)
    inter = union = 0; cr = []
    for (rm, rc), (m, c) in zip(ref, cand):
        x, y = rm == 1, m == 1
        inter += int((x & y).sum()); union += int((x | y).sum()); cr.append(abs(c - rc) / abs(rc))
    cr = np.array(cr)
    row = {"n_units": n, "counts": {t: sum(1 for u in units if u[1] == t) for t in sens.TYPES},
           "gate8_pooled_iou": g8["pooled_iou"], "gate8_conf_max": g8["conf_rel_max"], "gate8_pass": g8["pass"],
           "all126_pooled_iou": inter / union, "all126_conf_max": float(cr.max()), "all126_conf_p99": float(np.percentile(cr, 99)),
           "all126_conf_p90": float(np.percentile(cr, 90)), "all126_frames_over_5pct": int((cr > 0.05).sum()),
           "all126_frames_over_3pct": int((cr > 0.03).sum())}
    rows.append(row)
    print(json.dumps(row), flush=True)
write_results(HERE / "results.json", {"heldout_real_fp8": {a.model: rows}})
