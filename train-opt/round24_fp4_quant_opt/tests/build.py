"""Pre-build the extension variants on the CPU (no CUDA context, no GPU lock needed):  python tests/build.py [name:defs ...]"""
import sys, time
from pathlib import Path
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent)); sys.path.insert(0, str(ROUND / "tests"))
from bench_hot import load_mod
for c in (sys.argv[1:] or ["base"]):
    name, _, d = c.partition(":")
    t0 = time.time()
    load_mod(name, d).extension()
    print("built", c, f"{time.time() - t0:.0f}s", flush=True)
