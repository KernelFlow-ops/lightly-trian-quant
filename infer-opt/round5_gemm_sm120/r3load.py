"""Load round3's fused-block ops as package 'r3impl' (both rounds call their package 'impl')."""
import importlib.util
import sys

_R3 = "/root/lightly-train/infer-opt/round3_fusion/impl"


def load_r3():
    if "r3impl.ops" in sys.modules:
        return sys.modules["r3impl.ops"]
    spec = importlib.util.spec_from_file_location("r3impl", _R3 + "/__init__.py", submodule_search_locations=[_R3])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["r3impl"] = pkg
    spec.loader.exec_module(pkg)
    spec2 = importlib.util.spec_from_file_location("r3impl.ops", _R3 + "/ops.py")
    mod = importlib.util.module_from_spec(spec2)
    sys.modules["r3impl.ops"] = mod
    spec2.loader.exec_module(mod)
    return mod
