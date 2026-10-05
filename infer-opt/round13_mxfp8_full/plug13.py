"""Round 13 plugin for round12_lowp_stack/speed12.py:  --plugin /root/lightly-train/infer-opt/round13_mxfp8_full:plug13
Registers the variants  r13_mx8a[+nohead]  (step 1, cuBLASLt MXFP8)  and  r13_mx8[...]  (step 2, own block-scaled kernels).
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "impl", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PREFIXES = ("r13",)


def make_variant(name, model, variant):
    import stack13
    return stack13.make_variant(name, model, variant)
