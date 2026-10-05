import sqlite3, sys
for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    rows = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    names = dict(c.execute("select id,value from StringIds").fetchall())
    n = len(rows)
    span = rows[-1][1] - rows[0][0]
    busy = sum(e - s for s, e, _ in rows)
    # overlap-aware busy time
    cur_end, cov = rows[0][0], 0
    for s, e, _ in rows:
        if s > cur_end: cur_end = s
        if e > cur_end: cov += e - cur_end; cur_end = e
    gaps = [rows[i + 1][0] - rows[i][1] for i in range(n - 1)]
    pos = [g for g in gaps if g > 0]
    print(p.split("/")[-1], "kernels", n, "span/iter(ms)", span / 20 / 1e6, "sum durations/iter", busy / 20 / 1e6, "union busy/iter", cov / 20 / 1e6,
          "mean gap us", sum(pos) / len(pos) / 1e3, "neg-gap(overlap) count", sum(1 for g in gaps if g < 0))
