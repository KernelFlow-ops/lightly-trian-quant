"""round20_fp4_probe (R20a) driver: runs the whole probe sequentially (every GPU step takes the dual lock inside its own script).

  python run.py [--skip-matrix] [--skip-error]    # ~25 min on infer-5090; outputs go to profile/
Individual steps (see NOTES section 3) can be run directly from tests/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
T = ROUND / "tests"
STEPS = {
    "selftest": [["python", str(T / "test_fp4_ref.py")]],
    "matrix": [["python", str(T / "bench_fp4_gemm.py"), "--model", m, "--what", w] for m, w in (("vitl16", "bb:7203"), ("vitl16", "bb:7903"), ("vitl16", "head"), ("vitb16", "bb:14406"), ("vitb16", "bb:15806"), ("vitb16", "head"))]
    + [["python", str(T / "bench_qutlass_gemm.py")]],
    "quant": [["python", str(T / "bench_fp4_quant.py")]],
    "error": [["python", str(T / "fp4_error_table.py"), "vitl16", "--save"], ["python", str(T / "fp4_error_table.py"), "vitb16"], ["python", str(T / "debug_wgrad.py")], ["python", str(T / "debug_prefix.py")]],
    "summary": [["python", str(T / n)] for n in ("summarize_r20a.py", "summarize_error.py", "make_tables.py", "make_results20a.py")],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--skip-error", action="store_true")
    a = ap.parse_args()
    for name, cmds in STEPS.items():
        if (name == "matrix" and a.skip_matrix) or (name == "error" and a.skip_error):
            continue
        for c in cmds:
            print("==", " ".join(c), flush=True)
            subprocess.run([sys.executable] + c[1:], check=True)
