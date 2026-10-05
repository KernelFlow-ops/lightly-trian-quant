"""Run an existing round's test script WITHOUT letting it overwrite that round's result files: every Path.write_text into another round's directory
is redirected to round19_final/profile/gateA_<round>_<filename>.  Usage: python tests/ro_run.py <path/to/test.py> [args...]"""
import pathlib
import runpy
import sys

ROUND19 = pathlib.Path(__file__).resolve().parents[1]
_orig = pathlib.Path.write_text


def _patched(self, data, *a, **k):
    s = str(self)
    if "/train-opt/round" in s and "/round19_final/" not in s:
        rn = next(p for p in pathlib.Path(s).parts if p.startswith("round"))
        dest = ROUND19 / "profile" / f"gateA_{rn.split('_')[0]}_{pathlib.Path(s).name}"
        print(f"[ro_run] redirect write {s} -> {dest}", flush=True)
        return _orig(dest, data, *a, **k)
    return _orig(self, data, *a, **k)


pathlib.Path.write_text = _patched
test = sys.argv[1]
sys.argv = [test] + sys.argv[2:]
sys.path.insert(0, str(pathlib.Path(test).resolve().parent))
runpy.run_path(test, run_name="__main__")
