"""speed12.py plugin of round 14:  --plugin /root/lightly-train/infer-opt/round14_fp4_full:plug14
Variants r14_nv4a / r14_mx4a (step 1: QuACK + fp4q2 pieces), r14_nv4 / r14_mx4 (step 2: own block-scaled mma.sync kernel)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE / "impl") not in sys.path:
    sys.path.insert(0, str(HERE / "impl"))

PREFIXES = ("r14",)


def make_variant(name, model, variant):
    import stack14
    return stack14.make_variant(name, model, variant)
