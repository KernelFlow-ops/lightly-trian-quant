"""rank sweep results: python rank_sweeps.py <model> [--stage G1,G2,..] [--fmt int8] [--kinds all4,mlp] [--top 6]"""
import argparse, glob, json
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--fmt"); ap.add_argument("--kinds"); ap.add_argument("--stage"); ap.add_argument("--top", type=int, default=5)
ap.add_argument("--all", action="store_true", help="also show failing")
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(f"sweeps/*_{a.model}.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" in r or "H" not in r: continue
        rows.append(r)
def key(r):
    n = r["name"].split("|"); return n
sel = []
for r in rows:
    n = key(r)
    if a.stage and n[0] not in a.stage.split(","): continue
    if len(n) < 6: continue
    st, fmt, ks, ls, pfx, tech = n[:6]
    if a.fmt and fmt != a.fmt: continue
    if a.kinds and ks not in a.kinds.split(","): continue
    sel.append((st, fmt, ks, ls, pfx, tech, r))
from collections import defaultdict
g = defaultdict(list)
for st, fmt, ks, ls, pfx, tech, r in sel:
    g[(fmt, ks)].append((st, ls, pfx, tech, r))
for (fmt, ks), L in sorted(g.items()):
    ok = [x for x in L if x[4]["M"]["pass"] and x[4]["H"]["pass"]]
    pool = ok if (ok or not a.all) else L
    pool = sorted(pool, key=lambda x: x[4]["H"]["logit_mae"])[: a.top]
    print(f"--- {fmt} {ks}: {len(ok)}/{len(L)} pass M+H")
    for st, ls, pfx, tech, r in pool:
        m, h = r["M"], r["H"]
        print(f"   {st}|{ls}|{pfx}|{tech:10s} M {m['pooled_iou']:.4f}/{m['conf_max']*100:5.2f}% | H iou {h['pooled_iou']:.4f} min {h['min_iou']:.3f} conf {h['conf_max']*100:5.2f}% >3%:{h['n_conf_gt3']:2d} mae {h['logit_mae']:.2e} {'P' if (m['pass'] and h['pass']) else 'f'}")
