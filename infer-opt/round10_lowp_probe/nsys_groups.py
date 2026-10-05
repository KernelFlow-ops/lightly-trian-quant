import sqlite3, sys, collections, statistics
db = sqlite3.connect(sys.argv[1]); nframes = int(sys.argv[2])
cur = db.cursor()
names = dict(cur.execute("select id, value from StringIds").fetchall())
cols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
print(cols)
rows = cur.execute("select start,end,streamId,demangledName,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
g = collections.defaultdict(list)
for s,e,st,n,gx,gy,gz,bx in rows:
    g[(names[n][:70], gx,gy,gz,bx)].append(e-s)
tot = 0
for k,v in sorted(g.items(), key=lambda kv: -sum(kv[1])):
    t = sum(v)/nframes/1e3
    tot += t
    print(f"{t:8.1f} us/frame  n/frame={len(v)/nframes:6.2f}  avg={statistics.fmean(v)/1e3:8.1f}us  {k}")
print("total us/frame", tot)
