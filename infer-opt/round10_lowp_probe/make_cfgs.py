"""Write sweep config files (cfgs/<name>.json) for emu_eval.py sweep.  Each config = {"name", "entries": [{"kinds", "layers", "spec"}], "mode"}.

  python make_cfgs.py A            # stage A: one GEMM kind (or head kind) at a time, all layers, plain round-to-nearest, per format
  python make_cfgs.py B <model>    # stage B: layer sets / prefix rows for the (format, kind) pairs that survived stage A (reads sweeps/A_<model>.jsonl)
  python make_cfgs.py C <model>    # stage C: techniques (SmoothQuant, Hadamard, GPTQ) on the stage B candidates
  python make_cfgs.py D <model> <fmt> <kinds,...>   # stage D: single-unit sensitivities (layer, kind) for the greedy ordering
  python make_cfgs.py E <model> <fmt> <kinds,...> <steps>  # stage E: greedy prefixes from sweeps/D_<model>_<fmt>.jsonl (full 126-frame runs)
  python make_cfgs.py F <model>    # fp16 tier configs from the overflow scan (profile/overflow_scan_<model>.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "cfgs"
SW = HERE / "sweeps"
NB = {"vitl16": 24, "vitb16": 12}
KINDS = ["qkv", "proj", "fc1", "fc2"]
HEAD = ["ct1", "ct2", "einsum"]
FMTS = ["mx8", "int8", "nv4", "mx4"]


def ent(kinds, layers, **spec):
    return {"kinds": list(kinds), "layers": layers, "spec": spec}


def save(name, cfgs):
    CFG.mkdir(exist_ok=True)
    (CFG / f"{name}.json").write_text(json.dumps(cfgs, indent=0))
    print(name, len(cfgs), "configs")


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def stage_a():
    cfgs = [{"name": "A|p0", "entries": [], "mode": "screen"}]
    for fmt in FMTS:
        for k in KINDS + HEAD:
            cfgs.append({"name": f"A|{fmt}|{k}|all", "entries": [ent([k], "all", fmt=fmt)], "mode": "screen"})
        cfgs.append({"name": f"A|{fmt}|mlp|all", "entries": [ent(["fc1", "fc2"], "all", fmt=fmt)], "mode": "screen"})
        cfgs.append({"name": f"A|{fmt}|backbone4|all", "entries": [ent(KINDS, "all", fmt=fmt)], "mode": "screen"})
    return cfgs


def stage_b(model):
    rows = load(SW / f"A_{model}.jsonl")
    base = next((r["M"]["logit_mae"] for r in rows if r["name"] == "A|p0"), 0.0)
    cfgs = []
    for r in rows:
        if not r["name"].startswith("A|") or "error" in r or r["name"] == "A|p0":
            continue
        _, fmt, k, _ = r["name"].split("|")
        # keep pairs whose logit MAE increase is below a generous bound (these may pass the gate once layers are excluded)
        if r["M"]["logit_mae"] > 40 * max(base, 1e-6) and not r["M"]["pass"]:
            continue
        kinds = {"mlp": ["fc1", "fc2"], "backbone4": KINDS}.get(k, [k])
        for ls in ("nofirst", "nojoint", "nofirst_nojoint"):
            for pfx in (False, True):
                cfgs.append({"name": f"B|{fmt}|{k}|{ls}|pfx{int(pfx)}", "entries": [ent(kinds, ls, fmt=fmt, pfx=pfx)], "mode": "screen"})
        cfgs.append({"name": f"B|{fmt}|{k}|all|pfx1", "entries": [ent(kinds, "all", fmt=fmt, pfx=True)], "mode": "screen"})
    return cfgs


def stage_c(model):
    """techniques on top of the best stage A/B format x kind sets: pick the (fmt, kinds) with the lowest logit MAE per fmt and try SQ / Hadamard / GPTQ on them"""
    cfgs = []
    for fmt in FMTS:
        for k, kinds in (("mlp", ["fc1", "fc2"]), ("backbone4", KINDS), ("qkv", ["qkv"]), ("fc1", ["fc1"]), ("proj", ["proj"]), ("fc2", ["fc2"])):
            for tag, spec in (("sq50", dict(sq=0.5)), ("sq70", dict(sq=0.7)), ("sq85", dict(sq=0.85)), ("had16", dict(had=16)), ("had32", dict(had=32)),
                              ("sq70had16", dict(sq=0.7, had=16)), ("gptq", dict(gptq=True)), ("gptqhad16", dict(gptq=True, had=16))):
                if "sq" in spec and k == "fc2":       # no SmoothQuant fold for the GELU-fed fc2 input
                    continue
                if spec.get("sq") and spec.get("gptq"):
                    continue
                cfgs.append({"name": f"C|{fmt}|{k}|{tag}|all", "entries": [ent(kinds, "all", fmt=fmt, **spec)], "mode": "screen"})
    return cfgs


def stage_d(model, fmt, kinds, extra=None):
    extra = extra or {}
    cfgs = []
    for k in kinds:
        for i in range(NB[model]):
            cfgs.append({"name": f"D|{fmt}|{k}|L{i}" + ("|" + "_".join(f"{a}{b}" for a, b in extra.items()) if extra else ""),
                         "entries": [ent([k], [i], fmt=fmt, **extra)], "mode": "screen", "meta": {"layer": i, "kind": k}})
    return cfgs


def stage_e(model, fmt, kinds, steps, extra=None, tag=""):
    """greedy prefixes: units ranked by single-unit logit MAE increase (stage D), cumulative sets of the first n units, full 126-frame runs."""
    extra = extra or {}
    rows = [r for r in load(SW / f"D_{model}_{fmt}{tag}.jsonl") if r["name"].startswith(f"D|{fmt}|") and "error" not in r]
    rows.sort(key=lambda r: r["M"]["logit_mae"])
    order = [(r["meta"]["kind"], r["meta"]["layer"]) for r in rows if r["meta"]["kind"] in kinds]
    cfgs = []
    for n in steps:
        units = order[:n]
        ents = []
        for k in kinds:
            ls = [l for kk, l in units if kk == k]
            if ls:
                ents.append(ent([k], ls, fmt=fmt, **extra))
        cfgs.append({"name": f"E|{fmt}{tag}|{'+'.join(kinds)}|top{n}", "entries": ents, "mode": "full", "meta": {"units": units}})
    return cfgs


def stage_f(model):
    """fp16 tier: every layer of one kind (or all kinds) with the pre-scale p from the overflow scan (W-side scale, chunk 32)."""
    scan = json.loads((HERE / "profile" / f"overflow_scan_{model}.json").read_text())["per_gemm"]
    def spec_for(i, k, excl):
        c = scan[f"{i}|{k}"]["c32"]
        return dict(fmt="fp16", pw=c["p_excl_prefix" if excl else "p_all"], pfx=excl)
    cfgs = []
    for excl in (False, True):
        tag = "pfx" if excl else "all"
        for kinds in ([k] for k in KINDS):
            ents = [ent(kinds, [i], **spec_for(i, kinds[0], excl)) for i in range(NB[model])]
            cfgs.append({"name": f"F|fp16|{kinds[0]}|{tag}", "entries": ents, "mode": "full"})
        ents = [ent([k], [i], **spec_for(i, k, excl)) for k in KINDS for i in range(NB[model])]
        cfgs.append({"name": f"F|fp16|backbone4|{tag}", "entries": ents, "mode": "full"})
        for hk in HEAD[:2]:
            cfgs.append({"name": f"F|fp16|{hk}|{tag}", "entries": [ent([hk], "all", fmt="fp16")], "mode": "full"})
    cfgs.append({"name": "F|p0", "entries": [], "mode": "full"})
    return cfgs


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "A":
        save("A", stage_a())
    elif st == "B":
        save(f"B_{sys.argv[2]}", stage_b(sys.argv[2]))
    elif st == "C":
        save(f"C_{sys.argv[2]}", stage_c(sys.argv[2]))
    elif st == "D":
        save(f"D_{sys.argv[2]}_{sys.argv[3]}", stage_d(sys.argv[2], sys.argv[3], sys.argv[4].split(",")))
    elif st == "E":
        save(f"E_{sys.argv[2]}_{sys.argv[3]}", stage_e(sys.argv[2], sys.argv[3], sys.argv[4].split(","), [int(x) for x in sys.argv[5].split(",")]))
    elif st == "F":
        save(f"F_{sys.argv[2]}", stage_f(sys.argv[2]))


# ---------------------------------------------------------------------------------------------------- generic grids (full 126-frame mode: M from the 8 eval frames, H from the 110 non-calibration frames)
KSETS = {"qkv": ["qkv"], "proj": ["proj"], "fc1": ["fc1"], "fc2": ["fc2"], "mlp": ["fc1", "fc2"], "attn": ["qkv", "proj"], "all4": KINDS}
TECH = {"rn": {}, "sq50": dict(sq=0.5), "sq70": dict(sq=0.7), "sq85": dict(sq=0.85), "had16": dict(had=16), "had32": dict(had=32),
        "sq70had16": dict(sq=0.7, had=16), "gptq": dict(gptq=True), "gptqhad16": dict(gptq=True, had=16), "gptqhad32": dict(gptq=True, had=32)}


def grid(prefix, fmts, ksets, layersets, pfxs, techs, mode="full"):
    cfgs = []
    for fmt in fmts:
        for ks in ksets:
            kinds = KSETS[ks]
            for ls in layersets:
                for pfx in pfxs:
                    for t in techs:
                        spec = dict(TECH[t])
                        if "sq" in spec and ks == "fc2":
                            continue
                        if spec.get("sq") and spec.get("gptq"):
                            continue
                        e = ent(kinds, ls, fmt=fmt, pfx=pfx, **spec)
                        if "sq" in spec and "fc2" in kinds:     # fc2 has no SmoothQuant fold: split it out
                            e1 = ent([k for k in kinds if k != "fc2"], ls, fmt=fmt, pfx=pfx, **spec)
                            e2 = ent(["fc2"], ls, fmt=fmt, pfx=pfx, **{k: v for k, v in spec.items() if k != "sq"})
                            entries = [e1, e2]
                        else:
                            entries = [e]
                        cfgs.append({"name": f"{prefix}|{fmt}|{ks}|{ls}|pfx{int(pfx)}|{t}", "entries": entries, "mode": mode})
    return cfgs


def head_combos(fmts, mode="full"):
    out = []
    for fmt in fmts:
        for hs in (["ct1"], ["ct2"], ["einsum"], ["ct1", "ct2"], ["ct1", "ct2", "einsum"]):
            out.append({"name": f"H|{fmt}|{'+'.join(hs)}", "entries": [ent(hs, "all", fmt=fmt)], "mode": mode})
    return out


def sets_for(name):
    if name == "G1":     # plain RN, int8 + mx8, every kind-set x layer-set x prefix handling
        return grid("G1", ["int8", "mx8"], ["qkv", "proj", "fc1", "fc2", "mlp", "attn", "all4"], ["all", "nofirst", "nojoint", "nofirst_nojoint"], [False, True], ["rn"]) \
            + [{"name": "G1|p0", "entries": [], "mode": "full"}]
    if name == "G2":     # techniques for int8 / mx8 on the broader sets
        return grid("G2", ["int8", "mx8"], ["qkv", "fc1", "proj", "all4", "attn"], ["all", "nofirst_nojoint"], [False, True], ["sq50", "sq70", "sq85", "had16", "had32", "sq70had16"])
    if name == "G3":     # 4-bit formats: techniques on the MLP / single kinds (the only place they could pass)
        return grid("G3", ["nv4", "mx4"], ["fc1", "fc2", "mlp", "proj", "qkv"], ["all", "nofirst_nojoint"], [False, True], ["rn", "had16", "had32", "sq50", "sq70"])
    if name == "G4":     # GPTQ (needs Hessians: calibration pass computes them)
        return grid("G4", ["int8", "mx8", "nv4", "mx4"], ["fc1", "fc2", "mlp", "all4"], ["all", "nofirst_nojoint"], [False, True], ["gptq", "gptqhad16"])
    if name == "G5":     # heads
        return head_combos(["int8", "mx8", "nv4", "mx4", "fp16"])
    raise ValueError(name)


if __name__ == "__main__" and sys.argv[1] in ("G1", "G2", "G3", "G4", "G5"):
    save(sys.argv[1], sets_for(sys.argv[1]))


def sets_for2(name):
    if name == "G6":     # block-scaled int8 (one fp32 scale per 32/64/128 along K) -- accuracy of the 'int8 with k-group scales' kernel option vs per-token int8
        return grid("G6", ["i8b32", "i8b64", "i8b128"], ["qkv", "proj", "fc1", "fc2", "mlp", "all4"], ["all"], [False, True], ["rn"])
    raise ValueError(name)


if __name__ == "__main__" and sys.argv[1] in ("G6",):
    save(sys.argv[1], sets_for2(sys.argv[1]))
