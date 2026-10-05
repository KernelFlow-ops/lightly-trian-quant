"""Assemble the R20b chapter of NOTES.md (appended after the untouched R20a content) and results.json["r20b"] (screen / finals / arms / gateS / decision).

    python tests/make_notes_r20b.py [--tpl PATH --fill PATH]      (server side; reads profile/*.json, writes NOTES.md and results.json)

The narrative lives in ``r20b_notes.tpl.md`` (placeholders ``{{TABLE:key}}``, ``{{NAME}}``) and ``r20b_fill.py`` (the strings for the placeholders); both are kept
next to this script so the chapter can be regenerated from the measured json files at any time.  Only the part of NOTES.md from the R20b marker on is replaced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "tests"))
import summarize_r20b as S  # noqa: E402

MARK = "\n---\n\n# Round20（R20b）"
R20A_BACKUP = Path("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/r20b/NOTES_R20a.md")


def savings():
    d = json.loads((ROUND / "profile" / "per_step_estimate.json").read_text())["models"]
    step = {"vitl16": 101.8, "vitb16": 86.6}
    out = {}
    for m, v in d.items():
        it = v["step_estimate"]["fused_nv4_amax_from_producer"]["per_item"]
        pre = v["prefix_side_gemm_ms"]["prefix5_plain+prefix105_joint_ms"]
        nblk = 24 if m == "vitl16" else 12
        h1, h2a, b3 = it["item1_head_dw"]["saving_ms"], it["item2a_head_f_mx8"]["saving_ms"], it["item3_bb_dw"]["saving_ms"]
        h2 = v["items"]["1b_head_dw_mx8_fallback"]["dgemm_ms"]
        est = {"H1": h1, "H2": h2, "H3": h1 + h2a, "B3": h1 + h2a + b3, "B4": h1 + h2a + b3 * (1 - 1 / nblk), "B5": h1 + h2a + b3 - pre}
        out[m] = {k: {"ms": round(x, 2), "pct_of_final_step": round(100 * x / step[m], 1)} for k, x in est.items()}
        out[m]["_prefix_side_gemm_ms"] = round(pre, 2)
        out[m]["_nofirst_loss_ms"] = round(b3 / nblk, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", default=str(ROUND / "tests" / "r20b_notes.tpl.md"))
    ap.add_argument("--fill", default=str(ROUND / "tests" / "r20b_fill.py"))
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r20b_fill", args.fill)
    fill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fill)
    tabs, res = S.build()
    sav = savings()
    tpl = Path(args.tpl).read_text(encoding="utf-8")
    tpl = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: tabs.get(m.group(1), f"（表 `{m.group(1)}` 尚未产生：not measured）\n"), tpl)
    for k, v in fill.TEXT(sav, res).items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(\w+)\}\}", tpl)
    assert not left, left
    notes = ROUND / "NOTES.md"
    R20A_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    cur = notes.read_text(encoding="utf-8")
    base = cur.split(MARK)[0]
    if not R20A_BACKUP.exists():
        R20A_BACKUP.write_text(base, encoding="utf-8")
    notes.write_text(base.rstrip("\n") + "\n" + tpl, encoding="utf-8")
    rp = ROUND / "results.json"
    d = json.loads(rp.read_text())
    d["r20b"] = {**d.get("r20b", {}), **res, "estimated_saving_ms_from_r20a_per_step_estimate": sav, "decision": fill.DECISION(sav, res),
                 "measured_on": "infer-5090 (RTX 5090), eager simulation (torch quantizers + real QuACK FP4 GEMMs), dual lock", "step_speed": None,
                 "not_measured": fill.NOT_MEASURED_LIST, "delayed_amax_diag": fill.DIAG_ROWS()}
    rp.write_text(json.dumps(d, indent=1))
    print("NOTES.md chapter R20b:", len(tpl), "chars; tables:", sorted(tabs), "; results.json updated")


if __name__ == "__main__":
    main()
