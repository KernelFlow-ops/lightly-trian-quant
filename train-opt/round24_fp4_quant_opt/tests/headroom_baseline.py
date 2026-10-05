"""R21a follow-up: bit-identity guard for the `headroom` change.  `--save` (run on the PRE-change build) stores sha256 of the outputs of a fixed set of delayed-mode quantizer calls
(default arguments only); `--check` (post-change build) recomputes them with the default call and with an explicit headroom=2.0 and compares.   -> profile/headroom_baseline_hashes.json
"""
import hashlib
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND.parent / "round20_fp4_probe" / "impl"))
import torch

import fp4q2 as fp4q
from common.bench import gpu_lock

OUT = ROUND / "profile" / "headroom_baseline_hashes.json"
dev = "cuda"


def h(*ts):
    m = hashlib.sha256()
    for t in ts:
        if t is not None:
            m.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return m.hexdigest()[:20]


def run(extra):
    res = {}
    g = torch.Generator(device=dev).manual_seed(5)
    for fmt in ("nv4", "mx4"):
        for (R, C, pre) in ((7203, 1024, (7, 1029, 5)), (1029, 768, None), (14406, 768, None)):
            x = (torch.randn(R, C, device=dev, generator=g) * 3).bfloat16()
            x[::97] *= 8
            for cfg in ("dual", "rhtcol_sr", "prefix_rhtcol_sr"):
                if cfg == "prefix_rhtcol_sr" and pre is None:
                    continue
                st = fp4q.FP4State(4, seed=99)
                kw = dict(prefix=pre) if "prefix" in cfg else {}
                if "sr" in cfg:
                    kw.update(rht_col=True, sr_row=True, sr_col=True, sr_offset=3, seed=st.seed)
                dk = dict(amax="delayed", state=st, site=1, **extra) if fmt == "nv4" else {}
                # step 0 (exact init), step 1 (delayed from prev, input scaled), step 2
                for step, scale in enumerate((1.0, 0.5, 3.0)):
                    st.roll()
                    k = fp4q.quant_act((x.float() * scale).bfloat16(), fmt, **kw, **dk)
                    res[f"act/{fmt}/{R}x{C}/{cfg}/step{step}"] = h(k.q, k.sf, k.ts, k.qt, k.sft, k.tst, k.xp)
                getattr(st, "fold", lambda: None)()  # round24: the fast kernels publish the amax spread over NPART sectors; fold() moves it into amax[1] (roll() does it inside the graph)
                res[f"act/{fmt}/{R}x{C}/{cfg}/amax_state"] = h(st.amax, st.seed, st.step)
    for fmt in ("nv4", "mx4"):
        for (N, K) in ((3072, 1024), (768, 768)):
            w = torch.randn(N, K, device=dev, generator=g) * 0.02
            st = fp4q.FP4State(3, seed=1)
            dk = dict(amax="delayed", state=st, site=2, **extra) if fmt == "nv4" else {}
            for step, scale in enumerate((1.0, 1.01, 1.0)):
                st.roll()
                k = fp4q.quant_weight(w * scale, fmt, **dk)
                res[f"weight/{fmt}/{N}x{K}/step{step}"] = h(k.q, k.sf, k.ts, k.qt, k.sft)
            getattr(st, "fold", lambda: None)()
            res[f"weight/{fmt}/{N}x{K}/amax_state"] = h(st.amax, st.seed, st.step)
    return res


def main():
    mode = sys.argv[1]
    with gpu_lock():
        if mode == "--save":
            r = run({})
            OUT.write_text(json.dumps({"note": "sha256[:20] of outputs of the PRE-headroom build (default args)", "hashes": r}, indent=1))
            print("saved", len(r))
        else:
            base = json.loads(OUT.read_text())["hashes"]
            for tag, extra in (("default", {}), ("headroom=2.0", {"headroom": 2.0})):
                r = run(extra)
                bad = [k for k in base if r.get(k) != base[k]]
                print(f"{tag}: {len(base) - len(bad)}/{len(base)} identical", bad[:5])
                if bad:
                    sys.exit(1)
            print("DONE")


if __name__ == "__main__":
    main()
