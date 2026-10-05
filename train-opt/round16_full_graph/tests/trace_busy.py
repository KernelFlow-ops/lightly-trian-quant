"""GPU busy ratio / gaps from the raw kernel trace of a 5-step nsys window, independent of NVTX step ranges (which are CPU-only for async graph
replays).  window = first kernel start .. last kernel end; busy = union of kernel intervals; gaps = idle spans > 20 us.
  python trace_busy.py <nsys prefix> [...]"""
import csv
import sys


def analyze(prefix: str, steps: int = 5, gap_us: float = 20.0):
    rows = list(csv.DictReader(open(prefix + "_cuda_gpu_trace.csv")))
    iv = []
    for r in rows:
        if r.get("GrdX", "") == "" or r.get("Name", "") == "":
            continue  # memcpy / memset rows have no grid
        iv.append((float(r["Start (ns)"]), float(r["Start (ns)"]) + float(r["Duration (ns)"])))
    iv.sort()
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    span = merged[-1][1] - merged[0][0]
    busy = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    big = [g for g in gaps if g > gap_us * 1e3]
    return {"kernels": len(iv) / steps, "span_ms_per_step": span / 1e6 / steps, "busy_ms_per_step": busy / 1e6 / steps, "busy_ratio": busy / span,
            "gaps_gt20us_per_step": len(big) / steps, "gap_ms_per_step": sum(big) / 1e6 / steps, "largest_gap_ms": max(gaps) / 1e6}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        a = analyze(p)
        print(f"{p.split('/')[-1]:44s} kernels/step {a['kernels']:6.0f}  window/step {a['span_ms_per_step']:7.2f} ms  busy {a['busy_ms_per_step']:7.2f} ms  ratio {a['busy_ratio']:.3f}  "
              f"gaps>20us {a['gaps_gt20us_per_step']:6.1f}/step = {a['gap_ms_per_step']:5.2f} ms  largest gap {a['largest_gap_ms']:.2f} ms")
