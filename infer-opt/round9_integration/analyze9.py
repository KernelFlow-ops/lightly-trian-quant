"""nsys sqlite -> per-frame launch counts and GPU idle time.   python analyze9.py profile/<prefix>.sqlite [frames]"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 126
cur = db.cursor()
names = dict(cur.execute("select id, value from StringIds").fetchall())
rt = cur.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
cnt = {}
for s, e, n in rt:
    k = names.get(n, str(n)).split("_v")[0]
    cnt[k] = cnt.get(k, 0) + 1
kern = cur.execute("select start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
cp = cur.execute("select start,end,streamId,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
ms = cur.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMSET").fetchall() if cur.execute(
    "select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone() else []
print("runtime API calls in capture window (total / per frame):")
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {k:32s} {v:6d}  {v / nframes:7.2f}")
print(f"GPU kernels {len(kern)} ({len(kern) / nframes:.1f}/frame), memcpys {len(cp)} ({len(cp) / nframes:.2f}/frame)")
# compute-stream busy/idle over the steady window (first kernel .. last kernel)
iv = sorted((s, e) for s, e, _ in kern)
merged = []
for s, e in iv:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
span = merged[-1][1] - merged[0][0]
busy = sum(e - s for s, e in merged)
gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
print(f"kernel timeline: span {span / 1e6:.2f} ms over {nframes} frames = {span / 1e6 / nframes:.3f} ms/frame; "
      f"kernel-busy {busy / 1e6 / nframes:.3f} ms/frame; idle {(span - busy) / 1e6 / nframes:.3f} ms/frame "
      f"({100 * (span - busy) / span:.1f}% of span); busy fraction {100 * busy / span:.1f}%")
big = sorted(gaps, reverse=True)[:5]
print("largest 5 idle gaps (us):", [round(g / 1e3, 1) for g in big], " median gap us:", round(sorted(gaps)[len(gaps) // 2] / 1e3, 2) if gaps else None)
# frame boundaries: gaps > 20us split frames
fr = 1 + sum(1 for g in gaps if g > 20000)
print("groups separated by >20us:", fr)
kinds = {}
for s, e, st, k in cp:
    d = kinds.setdefault(k, [0, 0])
    d[0] += 1
    d[1] += e - s
for k, (n, d) in kinds.items():
    print(f"memcpy kind {k}: {n} ({n / nframes:.2f}/frame), {d / 1e6 / nframes:.3f} ms/frame")
