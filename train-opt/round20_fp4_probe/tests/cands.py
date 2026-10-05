"""R20b candidate policies (PLAN_v3 R20b / 6a 12.3).  Names -> policy strings (``impl/fp4_policy.py``)."""
from __future__ import annotations

H3 = "head.f=mx8,head.dw=nv4+sr"
B3 = H3 + ",all.d=nv4+sr,all.w=nv4+sr+rht"

SCREEN = {
    "H1": "head.dw=nv4+sr",
    "H2": "head.dw=mx8",
    "H3": H3,
    "B1": H3 + ",all.w=nv4+sr+rht",
    "B2": H3 + ",all.w=nv4+sr+rht,prefix",
    "B3": B3,
    "B4": B3 + ",nofirst",
    "B5": B3 + ",prefix",
}


def controls(base: str, tag: str) -> dict:
    """Control arms / record-only items around a backbone base policy (spelled out, so each name maps to one string)."""
    b = SCREEN[base]
    out = {}
    out[f"{base}+mx4"] = b.replace("nv4", "mx4")                         # nv4 -> mx4 (head and backbone)
    out[f"{base}+rn"] = b.replace("+sr", "+rn")                           # SR -> RN for the gradient operands
    out[f"{base}+w1d"] = b.replace("all.d=nv4+sr", "all.d=nv4+sr+w1d")      # 1D instead of 2D weight scales
    out[f"{base}+norht"] = b.replace("all.w=nv4+sr+rht", "all.w=nv4+sr")      # wgrad without RHT
    out[f"{base}+fcF4"] = b + ",fc.f=nv4"                              # record only
    out[f"{base}+qkvF8"] = b + ",qkv.f=mx8"                            # record only
    return out


def all_screen_cands() -> dict:
    d = dict(SCREEN)
    for b in ("B3", "B4", "B5"):
        d.update(controls(b, b))
    d["B3x"] = SCREEN["B3"] + ",amax=exact"   # same as B3 but the nv4 per-tensor amax is exact (S gate: delayed vs exact)
    d["B4x"] = SCREEN["B4"] + ",amax=exact"
    return d


if __name__ == "__main__":
    for k, v in all_screen_cands().items():
        print(f"{k:14s} {v}")
