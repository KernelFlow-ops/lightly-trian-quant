"""Split the own `gemm_kernel` launches of an nsys sqlite by grid shape: python gemm_split.py <sqlite> <iters>"""
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
it = int(sys.argv[2])
rows = c.execute("select k.gridX, k.gridY, k.blockX, count(*), sum(k.end - k.start), avg(k.end - k.start) from CUPTI_ACTIVITY_KIND_KERNEL k "
                 "join StringIds s on s.id = k.shortName where s.value like 'gemm_kernel%' group by 1, 2, 3 order by 5 desc").fetchall()
for gx, gy, bx, n, tot, avg in rows:
    print(f"grid {gx}x{gy} block {bx}: {n / it:.1f} launches/iter  {tot / it / 1e6:.3f} ms/iter  avg {avg / 1e3:.1f} us")
