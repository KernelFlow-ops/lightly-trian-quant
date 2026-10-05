"""Compact tables of the round14 measurements from profile/*.json (step benches, gate B)."""
import json
import statistics
from pathlib import Path

P = Path("/root/lightly-train/train-opt/round14_head_sm120/profile")
print("== step benches (50x3, warmup 20, ms per step; weighted img/s 0.2/0.6/0.2)")
W = {}
for m in ("vitl16", "vitb16"):
    for s in ("TB1_R13", "TB1_R13_R14", "P2", "P2_R14", "MXH_P2_bwd"):
        f = P / f"bench_{m}_{s}_all.json"
        if not f.exists():
            print(m, s, "MISSING")
            continue
        d = json.loads(f.read_text())["summary"]
        W[(m, s)] = d["weighted_img_s"]
        bd = {k: round(v, 1) for k, v in d["breakdown_ms"]["mid"].items()}
        print(f"{m} {s:12s} e/m/l {[round(d['step_ms_mean'][p], 1) for p in ('early', 'mid', 'late')]} w_img/s {d['weighted_img_s']:.2f} peak {max(d['peak_allocated_gb'].values()):.2f} GB mid {bd}")
print("== ratios")
for m in ("vitl16", "vitb16"):
    for a, b in (("TB1_R13", "TB1_R13_R14"), ("P2", "P2_R14"), ("P2_R14", "MXH_P2_bwd"), ("TB1_R13", "P2_R14")):
        if (m, a) in W and (m, b) in W:
            print(f"{m} {b}/{a} = {W[(m, b)] / W[(m, a)]:.3f}x  ({W[(m, a)]:.2f} -> {W[(m, b)]:.2f} img/s)")
print()
for f in ("gateB_vitl16_r14", "gateB_vitb16_r14", "gateB_vitl16_r14ctrl", "gateB_vitb16_r14ctrl", "gateB_vitl16_r14mx8"):
    p = P / f"{f}.json"
    if not p.exists():
        print(f, "MISSING")
        continue
    d = json.loads(p.read_text())
    print("==", f, d["model"], "seeds", list(d["seeds"].keys()))
    agg = {}
    for sd, rep in d["seeds"].items():
        for c, x in rep["cands"].items():
            agg.setdefault(c, []).append(x)
            print(f"  seed {sd} {c:14s} {x['kind']:4s} PASS={x['pass']!s:5s} vsref: loss_rel {x['vs_ref']['loss_rel']:.1e} gcos {x['vs_ref']['grad_cos_global']:.5f} nrel {x['vs_ref']['grad_norm_rel']:.4f} "
                  f"ucos {x['vs_ref']['update_cos_global']:.4f} pmin {x['vs_ref']['grad_cos_min']:.3f} | vsR0 loss_rel {x['vs_r0']['loss_rel']:.1e} gcos {x['vs_r0']['grad_cos_global']:.5f} | offenders {len(x['per_param_offenders_vs_r0'])}")
    for c, xs in agg.items():
        print(f"  -> {c}: {sum(bool(x['pass']) for x in xs)}/{len(xs)} pass; gcos_vs_ref min {min(x['vs_ref']['grad_cos_global'] for x in xs):.5f} median {statistics.median(x['vs_ref']['grad_cos_global'] for x in xs):.5f}; "
              f"loss_rel max {max(x['vs_ref']['loss_rel'] for x in xs):.1e}; offender counts {[len(x['per_param_offenders_vs_r0']) for x in xs]}")
