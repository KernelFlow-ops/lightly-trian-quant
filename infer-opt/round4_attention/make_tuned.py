"""Collect tune/*.json sweeps -> impl/tuned_cfg.json (best config per kind/H/N)."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
best = {}
for f in sorted((HERE / "tune").glob("*.json")):
    m = re.match(r"(plain_qkv|rope_qkv|plain|rope)_B2_H(\d+)_N(\d+)\.json", f.name)
    rows = json.loads(f.read_text())
    if not m or not rows: continue
    kind, H, N = m.group(1), int(m.group(2)), int(m.group(3))
    key = f"{kind}|{H}|{N}"
    r = min(rows, key=lambda r: r["us"])
    if key not in best or r["us"] < best[key]["us"]:
        best[key] = r
out = {k: {x: v[x] for x in ("bm", "bn", "stages", "threads")} for k, v in best.items()}
(HERE / "impl" / "tuned_cfg.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: {**out[k], "us": round(best[k]["us"], 1)} for k in out}, indent=1))
