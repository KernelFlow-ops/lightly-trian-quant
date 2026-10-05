"""round24 nsys category report (round22 classifier + actf_kernel of fp4q2): round19's classifier (imported, unmodified) + the FP4 categories (round21 quantizer kernels, QuACK nv4 GEMMs = MMA atom m16n8k64).
  python tests/nsys_report24.py --json profile/nsys_report24.json --label L_FINAL=<prefix> --label L_P3B=<prefix> ...
"""
import sys

sys.path.insert(0, "/root/lightly-train/train-opt/round19_final/tests")
import nsys_final_report as nfr  # noqa: E402

nfr.CATS = [
    ("FP4 quantize (round21 act_kernel, round24 actf_kernel / w_kernel, amax roll)", r"actf?_kernel|w_kernel|amax_roll|amax_abs"),
    ("GEMM NVFP4 (QuACK nv4 m16n8k64, backbone dgrad / wgrad)", r"ShapeMNK16864|GemmMod_(dgelu|bias|res|gelu|plain)_ts"),
] + nfr.CATS

if __name__ == "__main__":
    nfr.main()
