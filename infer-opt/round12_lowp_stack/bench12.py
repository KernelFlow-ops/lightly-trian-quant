"""Round 12 paired whole-video bench + gates M / H / G: round-11 bench11.py with the r12 variants registered.

  python bench12.py --model vitl16 --variants r5_v1,r11+head+tail+post,r12+nopatch,r12 --loops 3 --rounds 2 --tag paired
Variants: r5_v1 | r11[+flag...] (round 11) | r12[+nopatch][+head16] (impl/stack12.py).  The first variant is the paired reference.
Everything runs under the dual GPU lock (bench11.main takes it).  Results go to this round's results.json (key "bench11").
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R11 = HERE.parent / "round11_attn_tier0"
for p in (R11, R11 / "impl", HERE / "impl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r11_common as c  # noqa: E402

_orig = c.make_variant


def make_variant(name, model, variant):
    if variant.split("+")[0] == "r12":
        import stack12
        return stack12.make_variant(name, model, variant)
    return _orig(name, model, variant)


c.make_variant = make_variant

import bench11  # noqa: E402

bench11.OUT = HERE / "results.json"

if __name__ == "__main__":
    bench11.main()
