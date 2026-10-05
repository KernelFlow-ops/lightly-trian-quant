"""Run an existing round's test script WITHOUT letting it touch that round's result files or caches (adapted from round19_final/tests/ro_run.py):
  * every Path.write_text into another round's directory is redirected to round23_fp4_final/profile/gateA_<round>_<filename>,
  * Path.exists / read_text of those other-round profile files report "absent" (so the script starts a clean report instead of merging the old one),
  * QUACK_CACHE_DIR assignments are redirected to round23's own cache.
Usage: python tests/ro_run23.py <path/to/test.py> [args...]"""
import os
import pathlib
import runpy
import sys

ROUND23 = pathlib.Path(__file__).resolve().parents[1]
CACHE = "/root/autodl-tmp/train-opt-tmp/round23_fp4_final/quack_cache"
_orig_write, _orig_exists, _orig_read = pathlib.Path.write_text, pathlib.Path.exists, pathlib.Path.read_text
_orig_set = os._Environ.__setitem__


def _foreign(path) -> bool:
    s = str(path)
    return "/train-opt/round" in s and "/round23_fp4_final/" not in s and "/profile/" in s


def _dest(path):
    s = str(path)
    rn = next(p for p in pathlib.Path(s).parts if p.startswith("round"))
    return ROUND23 / "profile" / f"gateA_{rn.split('_')[0]}_{pathlib.Path(s).name}"


def _write(self, data, *a, **k):
    if _foreign(self):
        d = _dest(self)
        print(f"[ro_run23] redirect write {self} -> {d}", flush=True)
        return _orig_write(d, data, *a, **k)
    return _orig_write(self, data, *a, **k)


def _exists(self, *a, **k):
    if _foreign(self):
        return _orig_exists(_dest(self))
    return _orig_exists(self, *a, **k)


def _read(self, *a, **k):
    if _foreign(self):
        return _orig_read(_dest(self), *a, **k)
    return _orig_read(self, *a, **k)


def _setenv(self, key, value):
    if key == "QUACK_CACHE_DIR":
        value = CACHE
    return _orig_set(self, key, value)


pathlib.Path.write_text, pathlib.Path.exists, pathlib.Path.read_text = _write, _exists, _read
os._Environ.__setitem__ = _setenv
os.environ["QUACK_CACHE_DIR"] = CACHE
test = sys.argv[1]
sys.argv = [test] + sys.argv[2:]
sys.path.insert(0, str(pathlib.Path(test).resolve().parent))
runpy.run_path(test, run_name="__main__")
