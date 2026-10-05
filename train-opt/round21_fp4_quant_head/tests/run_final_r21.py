"""Run round19's run_final.py (real loader + StepGraphRunner + optional validation / checkpoint / export) with the round21 presets registered:
    python tests/run_final_r21.py --model vitb16 --stack P3H --steps 20 --val --save-ckpt DIR --json profile/run_final_vitb16_P3H.json
"""
import sys
from pathlib import Path

R21 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R21.parent))
sys.path.insert(0, str(R21))
import run as r21run  # noqa: E402,F401  (registers P3H*, redirects round16 / round19 outputs to round21)

sys.path.insert(0, str(R21.parent / "round19_final"))
import run_final as rf  # noqa: E402

if __name__ == "__main__":
    rf.main()
