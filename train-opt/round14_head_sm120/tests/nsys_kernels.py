"""Kernel table of one captured step (fwd/bwd NVTX ranges) from a profile_nsys.sh sqlite export.

Usage: python nsys_kernels.py <prefix> [--step 2]
Prints (a) the ordered kernel list of one step with fwd/bwd tag and (b) per-kernel-name totals averaged over steps.
"""
import argparse
import re
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--list", action="store_true")
args = ap.parse_args()
con = sqlite3.connect(f"{args.prefix}.sqlite")
names = {i: v for i, v in con.execute("SELECT id,value FROM StringIds")}
runtime = {c: s for c, s in con.execute("SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
nv = [(s, e, t if t is not None else names.get(tid, "")) for s, e, t, tid in con.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
steps = sorted((s, e) for s, e, n in nv if re.fullmatch(r"step_\d+", n))
tags = sorted((s, e, n) for s, e, n in nv if n in ("fwd", "bwd"))
ks = []
for s, e, c, n in con.execute("SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
    ks.append((s, e, runtime.get(c, s), names.get(n, "?")))
ks.sort()
def tag(t):
    for s, e, n in tags:
        if s <= t <= e:
            return n
    return "?"
per_step = defaultdict(list)
for s, e, launch, n in ks:
    for i, (ss, se) in enumerate(steps):
        if ss <= launch <= se:
            per_step[i].append((s, e, tag(launch), n))
nsteps = max(len(per_step), 1)
def short(n):
    return re.sub(r"\s+", " ", n)[:110]
if args.list:
    print(f"--- ordered kernels of step {args.step}")
    t0 = None
    for s, e, tg, n in per_step[args.step]:
        t0 = t0 or s
        print(f"{tg:3s} {(s-t0)/1e3:9.1f}us {(e-s)/1e3:8.1f}us  {short(n)}")
tot = defaultdict(lambda: [0, 0.0])
for i, lst in per_step.items():
    for s, e, tg, n in lst:
        key = (tg, short(n))
        tot[key][0] += 1
        tot[key][1] += (e - s) / 1e3
print(f"--- per-step totals (avg over {nsteps} steps): tag, count, total us, kernel")
rows = sorted(tot.items(), key=lambda kv: -kv[1][1])
sumall = sum(v[1] for _, v in rows) / nsteps
for (tg, n), (c, t) in rows[:40]:
    print(f"{tg:3s} {c/nsteps:5.1f} {t/nsteps:9.1f}us  {n}")
for tg in ("fwd", "bwd"):
    print(tg, "total GPU kernel us/step:", sum(v[1] for (t, _), v in tot.items() if t == tg) / nsteps)
for i, lst in per_step.items():
    if lst:
        print(f"step {i}: wall(first kernel start->last end) {(lst[-1][1]-lst[0][0])/1e3:.1f}us, kernel sum {sum(e-s for s,e,_,_ in lst)/1e3:.1f}us")
