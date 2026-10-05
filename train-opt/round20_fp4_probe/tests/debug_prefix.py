"""Does excluding the cls+register (+query) prefix tokens from the FP4 wgrad token reduction remove the block0 blow-up?  ViT-L cached tensors."""
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND / "tests"))
import torch

import fp4_error_table as E
import fp4_ref as R
from common.bench import gpu_lock

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = "cuda"
out = {}
with gpu_lock():
    store = torch.load("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/cap_vitl16.pt")
    signs = R.rht_signs(device=dev)
    for key in sorted(k for k in store if k.startswith("block")):
        X, dY = store[key]["X"].to(dev).float(), store[key]["dY"].to(dev).float()
        n = 1129 if key.startswith("block22") else 1029
        pref = 105 if n == 1129 else 5
        pos = torch.arange(X.shape[0], device=dev) % n
        keep = pos >= pref
        ref = dY.t() @ X
        ref_pre = dY[~keep].t() @ X[~keep]
        ref_pat = dY[keep].t() @ X[keep]
        rec = {"prefix_tokens_per_image": pref, "frac_of_dW_norm_from_prefix": float(ref_pre.norm() / ref.norm()), "X_prefix_absmax": float(X[~keep].abs().max()), "X_patch_absmax": float(X[keep].abs().max()),
               "dY_prefix_absmax": float(dY[~keep].abs().max()), "dY_patch_absmax": float(dY[keep].abs().max())}
        for fmt in ("nv4", "mx4", "mx8"):
            for rht in ((False, True) if fmt != "mx8" else (False,)):
                A = E.operand_tokens(dY[keep], fmt, rht=rht, signs=signs)
                B = E.operand_tokens(X[keep], fmt, rht=rht, signs=signs)
                approx = A @ B.t() + ref_pre  # prefix tokens contribution kept exact (bf16/fp32 side GEMM)
                A0 = E.operand_tokens(dY, fmt, rht=rht, signs=signs)
                B0 = E.operand_tokens(X, fmt, rht=rht, signs=signs)
                rec[f"{fmt}{'_rht' if rht else ''}_all_tokens"] = E.rel(A0 @ B0.t(), ref)
                rec[f"{fmt}{'_rht' if rht else ''}_prefix_excluded_exact_side_gemm"] = E.rel(approx, ref)
        out[key] = rec
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if "all_tokens" in k or "excluded" in k or k.startswith("frac")}, flush=True)
(ROUND / "profile" / "debug_prefix_tokens.json").write_text(json.dumps(out, indent=1))
