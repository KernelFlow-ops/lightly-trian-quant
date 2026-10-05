"""Debug the block0.fc2 wgrad blow-up (nv4+RHT relL2 25) on the cached real tensors."""
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
    for key in ("block0.fc2", "block0.fc1", "block11.fc2"):
        X, dY = store[key]["X"].to(dev).float(), store[key]["dY"].to(dev).float()
        ref = dY.t() @ X
        rec = {"ref_norm": float(ref.norm()), "X_absmax": float(X.abs().max()), "X_rms": float(X.pow(2).mean().sqrt()), "dY_absmax": float(dY.abs().max()), "dY_rms": float(dY.pow(2).mean().sqrt())}
        # outlier structure: tokens / channels with the largest magnitude
        tok = X.abs().amax(1)
        rec["X_top_tokens"] = [(int(i), float(tok[i])) for i in tok.topk(5).indices]
        rec["X_frac_tokens_gt_10x_rms"] = float((tok > 10 * rec["X_rms"]).float().mean())
        tokd = dY.abs().amax(1)
        rec["dY_top_tokens"] = [(int(i), float(tokd[i])) for i in tokd.topk(5).indices]
        rec["dY_frac_tokens_gt_100x_rms"] = float((tokd > 100 * rec["dY_rms"]).float().mean())
        for name, rht in (("rn", False), ("rht", True)):
            A = E.operand_tokens(dY, "nv4", rht=rht, signs=signs)
            B = E.operand_tokens(X, "nv4", rht=rht, signs=signs)
            Aexact = E.operand_tokens(dY, "mx8", rht=False) if False else None
            rec[f"nv4_{name}_both"] = E.rel(A @ B.t(), ref)
            # quantize only one operand (the other exact, rotated if rht)
            Mp = A.shape[1]
            Xp = torch.nn.functional.pad(X, (0, 0, 0, Mp - X.shape[0])).t().contiguous()
            dYp = torch.nn.functional.pad(dY, (0, 0, 0, Mp - dY.shape[0])).t().contiguous()
            if rht:
                Xp, dYp = R.rht_apply(Xp, signs), R.rht_apply(dYp, signs)
            rec[f"nv4_{name}_only_dY"] = E.rel(A @ Xp.t(), ref)
            rec[f"nv4_{name}_only_X"] = E.rel(dYp @ B.t(), ref)
            rec[f"nv4_{name}_operand_relL2_dY"] = E.rel(A, dYp)
            rec[f"nv4_{name}_operand_relL2_X"] = E.rel(B, Xp)
            # fraction of dY blocks whose e4m3 scale underflows to 0 (all-zero data)
            q, sf, ts = R.quantize_matrix(dYp if rht is False else dYp, "nv4")
            s2 = R.from_blocked(sf, dYp.shape[0], dYp.shape[1] // 16).view(torch.float8_e4m3fn).float()
            rec[f"nv4_{name}_dY_ts"] = float(ts)
            rec[f"nv4_{name}_dY_frac_scale0"] = float((s2 == 0).float().mean())
            rec[f"nv4_{name}_dY_frac_scale_subnormal(<2^-6)"] = float(((s2 > 0) & (s2 < 2 ** -6)).float().mean())
            q, sf, ts = R.quantize_matrix(Xp, "nv4")
            s2 = R.from_blocked(sf, Xp.shape[0], Xp.shape[1] // 16).view(torch.float8_e4m3fn).float()
            rec[f"nv4_{name}_X_frac_scale0"] = float((s2 == 0).float().mean())
            rec[f"nv4_{name}_X_frac_scale_subnormal(<2^-6)"] = float(((s2 > 0) & (s2 < 2 ** -6)).float().mean())
        # mx8 for reference
        rec["mx8_both"] = E.rel(E.operand_tokens(dY, "mx8") @ E.operand_tokens(X, "mx8").t(), ref)
        # excluding the top outlier tokens from X and dY (what would remain)
        out[key] = rec
        print(key, json.dumps(rec, default=lambda v: round(v, 4))[:2500], flush=True)
(ROUND / "profile" / "debug_wgrad_block0.json").write_text(json.dumps(out, indent=1))
