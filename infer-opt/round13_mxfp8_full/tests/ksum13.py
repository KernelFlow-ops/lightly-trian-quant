"""Per-kernel table (ms per iteration) from an nsys cuda_gpu_kern_sum CSV:  python tests/ksum13.py <prefix> <iters> [top]"""
import csv, sys
p, it = sys.argv[1], int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = list(csv.DictReader(open(p + "_cuda_gpu_kern_sum.csv")))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"total kernel time {tot/it/1e6:.3f} ms/iter, launches {sum(int(r['Instances']) for r in rows)/it:.0f}/iter")
print("| kernel | ms/iter | % | launches/iter | avg us |")
print("|---|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda r: -float(r["Total Time (ns)"]))[:top]:
    t = float(r["Total Time (ns)"]); n = int(r["Instances"])
    nm = r["Name"].replace("|", "/")
    print(f"| {nm[:90]} | {t/it/1e6:.3f} | {100*t/tot:.1f} | {n/it:.1f} | {t/n/1e3:.1f} |")
