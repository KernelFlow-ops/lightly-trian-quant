"""Per-frame kernel table from an nsys cuda_gpu_kern_sum.csv.   python ksum.py <prefix> [iters] [top]"""
import csv, sys
p = sys.argv[1]; it = int(sys.argv[2]) if len(sys.argv) > 2 else 30; top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = list(csv.DictReader(open(p + '_cuda_gpu_kern_sum.csv')))
tot = sum(float(r['Total Time (ns)']) for r in rows)
n = sum(int(r['Instances']) for r in rows)
print(f'total GPU kernel time/frame {tot/it/1e6:.3f} ms, kernels/frame {n/it:.0f}')
for r in sorted(rows, key=lambda r: -float(r['Total Time (ns)']))[:top]:
    t = float(r['Total Time (ns)'])
    print(f"{t/it/1e3:9.1f} us/frame  {int(r['Instances'])/it:6.1f}x  avg {float(r['Avg (ns)'])/1e3:8.1f}  {r['Name'][:105]}")
