"""Print candidate-matrix JSON files as compact tables."""
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text())
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    print(f"== {path}")
    last = None
    for r in rows:
        key = (r.get("model"), r.get("kind"), r.get("mask"))
        if key != last:
            print(f"-- {key} B={r.get('B')} H={r.get('H')} N={r.get('N')}")
            last = key
        if "error" in r:
            print(f"  {r['variant']:<34} ERROR {r['error'][:90]}")
        else:
            def f(k, w=8):
                v = r.get(k)
                return f"{v:{w}.1f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
            print(f"  {r['variant']:<34} fwd {f('fwd_us')}us {f('fwd_tflops',6)}T | bwd {f('bwd_us')}us {f('bwd_tflops',6)}T | f+b {f('fwd_bwd_us')}us {f('fwd_bwd_tflops',6)}T")
