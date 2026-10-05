"""Markdown table generators for NOTES.md (build_notes.py replaces {{name}} / {{name:arg}} placeholders in NOTES_template.md).  All numbers come from the json / jsonl files."""
from __future__ import annotations

import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("vitl16", "vitb16")
KINDS = ("qkv", "proj", "fc1", "fc2")
SHORT = {"vitl16": "L", "vitb16": "B"}


def J(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else {}


def merged(files):
    out = {}
    for f in files:
        for k, v in J(f).items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            out.setdefault(k, {}).update({c: r for c, r in v.items() if isinstance(r, dict)})
    return out


def f1(v):
    return "-" if v is None else f"{v:.1f}"


def gemm_lowp1(_=None):
    d = merged(["results_gemm_lowp1.json"])
    ep = merged(["results_gemm_epi.json"])
    cols = ["r5_v1", "cublas_fp16acc", "cublaslt_mx8", "spark_mx8_v1", "spark_mx8_v2", "int8_mm"]
    out = ["| 形状 | " + " | ".join(cols) + " | epi_i32 | epi_bf16 |", "|---|" + "---:|" * (len(cols) + 2)]
    for m in MODELS:
        for k in KINDS:
            for M in (2058, 2258):
                r = d.get(f"{m}|{k}|{M}")
                if not r:
                    continue
                e = ep.get(f"{m}|{k}|{M}", {})
                cells = []
                for c in cols:
                    v = r.get(c)
                    cells.append("n/a" if v is None or "us" not in v else f"{v['us']:.1f} ({v['tflops']:.0f})")
                for c in ("epi_i32", "epi_bf16"):
                    v = e.get(c) or r.get(c)
                    cells.append("-" if v is None or "us" not in v else f"{v['us']:.1f}")
                out.append(f"| {SHORT[m]} {k} {M} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def mg_stage2(tag):
    """prototype tile sweep stage 2: best tile per shape vs same-run r5_v1 / cuBLAS fp16-acc / bf16."""
    files = {"mg16": "impl/tuned16.json", "mg16_pkt1": "impl/tuned16_pkt1.json", "mg8": "impl/tuned8.json"}
    t = J(files[tag])
    out = ["| 形状 | 原型 µs (TFLOPS) | r5_v1 (同轮) | 对 r5_v1 | " + ("cuBLAS fp16-acc | " if tag.startswith("mg16") else "cuBLAS fp16-acc | ") + "最优 tile |", "|---|---:|---:|---:|---:|---|"]
    from bench10 import kind_nk
    for m in MODELS:
        for k in KINDS:
            N, K = kind_nk(m, k)
            for M in (2058, 2258):
                e = t.get(f"{k}|{N}|{K}|{M}")
                if not e:
                    continue
                b = e.get("baseline_us", {})
                cfg = e["cfg"]
                tile = f"{cfg['bm']}x{cfg['bn']}x{cfg['bk']} st{cfg['st']} w{cfg['wm']}x{cfg['wn']} minb{cfg['minb']} sk{cfg.get('splitk', 1)}" + (f" pkt{cfg['pkt']}" if cfg.get("pkt", 1) != 1 else "")
                r5 = b.get("r5_v1")
                out.append(f"| {SHORT[m]} {k} {M} | {e['us']:.1f} ({e['tflops']:.0f}) | {f1(r5)} | {'-' if not r5 else f'{r5 / e['us']:.2f}x'} | {f1(b.get('cublas_fp16acc'))} | {tile} |")
    return "\n".join(out)


def quant_costs(_=None):
    q = J("results_quant.json")
    if not q:
        return "(not measured yet)"
    names = []
    for v in q.values():
        for c in v:
            if c not in names:
                names.append(c)
    out = ["| 位点 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in sorted(q):
        out.append(f"| {k} | " + " | ".join(f1(q[k].get(c, {}).get("us")) if "us" in q[k].get(c, {}) else "-" for c in names) + " |")
    return "\n".join(out)


def sweep_rows(model, prefix):
    rows = []
    for f in sorted(glob.glob(str(HERE / "sweeps" / f"{prefix}_{model}.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    rows.append(r)
    return rows


def _fmt(r):
    m = r["M"]
    s = f"{m['pooled_iou']:.4f} / {m['conf_max'] * 100:.2f}%"
    h = r.get("H")
    if h:
        s2 = f"{h['pooled_iou']:.4f} / {h['min_iou']:.3f} / {h['conf_max'] * 100:.2f}% / {h['n_conf_gt3']}"
        return s, s2, f"{h['logit_mae']:.2e}", ("通过" if (m["pass"] and h["pass"]) else "不过")
    return s, "-", f"{m['logit_mae']:.2e}", ("M 通过" if m["pass"] else "M 不过")


def sweep_A(model):
    rows = sweep_rows(model, "A")
    out = ["| 配置（全部层，RN，无技术） | 门 M：IoU / conf | logit MAE | 门 M |", "|---|---|---:|---|"]
    for r in rows:
        m = r["M"]
        out.append(f"| {r['name'].replace('A|', '').replace('|all', '')} | {m['pooled_iou']:.4f} / {m['conf_max'] * 100:.2f}% | {m['logit_mae']:.2e} | {'过' if m['pass'] else '不过'} |")
    return "\n".join(out)


def sweep_F(model):
    rows = sweep_rows(model, "F_tuned")
    out = ["| 配置（fp16 档，真实 mg_f16acc kernel） | 门 M：IoU / conf | 门 H（110 帧）：IoU / min-frame IoU / conf 最大 / conf>3% 帧数 | H logit MAE | 判定 |", "|---|---|---|---:|---|"]
    for r in rows:
        a, b, c, d = _fmt(r)
        out.append(f"| {r['name'].replace('F|', '')} | {a} | {b} | {c} | {d} |")
    return "\n".join(out)


def sweep_best(arg):
    """arg = 'model:stage:fmt' -> best M+H passing rows per kind-set (lowest H logit MAE)"""
    model, stage, fmt = arg.split(":")
    if stage == "ALL":
        rows = [r for st in ("G1", "G2", "G3", "G4", "G6") for r in sweep_rows(model, st)]
    else:
        rows = sweep_rows(model, stage)
    rows = [r for r in rows if len(r["name"].split("|")) == 6 and r["name"].split("|")[1] == fmt and "H" in r]
    best = {}
    for r in rows:
        n = r["name"].split("|")
        ks = n[2]
        ok = r["M"]["pass"] and r["H"]["pass"]
        key = (ks, ok)
        if key not in best or r["H"]["logit_mae"] < best[key]["H"]["logit_mae"]:
            best[key] = r
    out = ["| 层集合 | 种类 | 前缀行 | 技术 | 门 M | 门 H：IoU / min-frame / conf 最大 / >3% | H logit MAE | 通过 M+H 的配置数 / 总数 |", "|---|---|---|---|---|---|---:|---|"]
    for ks in ("qkv", "proj", "fc1", "fc2", "mlp", "attn", "all4"):
        tot = [r for r in rows if r["name"].split("|")[2] == ks]
        if not tot:
            continue
        npass = sum(1 for r in tot if r["M"]["pass"] and r["H"]["pass"])
        r = best.get((ks, True)) or best.get((ks, False))
        n = r["name"].split("|")
        a, b, c, d = _fmt(r)
        out.append(f"| {n[3]} | {ks} | {n[4]} | {n[5]} | {a} | {b} | {c} | {npass}/{len(tot)}（{'此行为通过者中 MAE 最低' if best.get((ks, True)) else '无通过者，此行为 MAE 最低'}） |")
    return "\n".join(out)


def gemm_extra(_=None):
    d = merged(["results_gemm_extra1.json", "results_gemm_extra2.json"])
    cols = ["cublas_bf16", "cublas_fp16acc", "mg16_tuned", "mg8_tuned", "cublaslt_mx8", "int8_mm", "spark_mx8_v1", "spark_mx8_v2", "quack_mx8", "cublaslt_nv4", "cublaslt_mx4"]
    cols = [c for c in cols if any(c in v for v in d.values())]
    out = ["| 形状（模型 / GEMM / M） | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in sorted(d):
        out.append(f"| {k} | " + " | ".join(f"{d[k][c]['us']:.1f}" if c in d[k] and 'us' in d[k][c] else "-" for c in cols) + " |")
    return "\n".join(out)


def gemm_b2(_=None):
    d = merged(["results_gemm_b2.json"])
    cols = ["r5_v1", "cublas_bf16", "cublas_fp16acc", "mg16_tuned", "mg8_tuned", "cublaslt_mx8", "int8_mm", "epi_i32"]
    out = ["| 形状 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in sorted(d):
        out.append(f"| {k} | " + " | ".join(f"{d[k][c]['us']:.1f}" if c in d[k] and 'us' in d[k][c] else "-" for c in cols) + " |")
    return "\n".join(out)


def gemm_quack_fp4(_=None):
    d = merged(["results_gemm_quackmx8.json", "results_gemm_fp4a.json", "results_gemm_fp4b.json"])
    cols = ["quack_mx8", "cublaslt_nv4", "cublaslt_mx4", "qutlass_nv4", "qutlass_mx4", "spark_nv4_v1", "spark_nv4_v2", "spark_mx4_v1", "spark_mx4_v2", "quack_nv4", "quack_mx4"]
    cols = [c for c in cols if any(c in v for v in d.values())]
    if not cols:
        return "(not measured yet)"
    out = ["| 形状 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in sorted(d):
        out.append(f"| {k} | " + " | ".join(f"{d[k][c]['us']:.1f} ({d[k][c]['tflops']:.0f})" if c in d[k] and 'us' in d[k][c] else "-" for c in cols) + " |")
    return "\n".join(out)


def perframe(_=None):
    """per-frame GEMM ms per tier (kernel + separately needed epilogue; activation quantisation listed separately), from speed_model."""
    from speed_model import Speed, LAYERS
    sp = Speed()
    out = []
    tiers = [("r5_v1", "bf16"), ("fp16", "fp16"), ("fp16_pkt1", "fp16"), ("fp16_cublas", "fp16"), ("int8", "int8"), ("int8_cublas", "int8"), ("mx8", "mx8"), ("nv4", "nv4"), ("mx4", "mx4")]
    for m in MODELS:
        out.append(f"**{m}**（ms / 帧；量化 = 激活量化，central = 融合进生产者的假设值，pess = 单独量化 kernel 实测）")
        out.append("")
        out.append("| 档位 / kernel | " + " | ".join(KINDS) + " | 合计（无量化） | 量化 central | 量化 pess | 合计 central | 合计 pess | 对 r5_v1 省（central） |")
        out.append("|---|" + "---:|" * (len(KINDS) + 7))
        base_tot = None
        for t, q in tiers:
            ks = [sp.frame_ms(m, k, t, "bf16", "central") if t == "r5_v1" else sp.frame_ms(m, k, t, "bf16", "central") for k in KINDS]
            if any(v is None for v in ks):
                continue
            qc = [sp.frame_ms(m, k, t, q, "central") - v for k, v in zip(KINDS, ks)] if q != "bf16" else [0.0] * 4
            qp = [sp.frame_ms(m, k, t, q, "pess") - v for k, v in zip(KINDS, ks)] if q != "bf16" else [0.0] * 4
            tot = sum(ks)
            if t == "r5_v1":
                base_tot = tot
            out.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in ks) + f" | {tot:.3f} | {sum(qc):.3f} | {sum(qp):.3f} | {tot + sum(qc):.3f} | {tot + sum(qp):.3f} | {(base_tot - tot - sum(qc)):.3f} |")
        out.append("")
    return "\n".join(out)



def sweep_head(model):
    rows = sweep_rows(model, "G5")
    out = ["| 头部 GEMM（全部输入） | 格式 | 门 M：IoU / conf | 门 H（110 帧）：IoU / min-frame / conf 最大 / conf>3% | H logit MAE | 判定 |", "|---|---|---|---|---:|---|"]
    for r in rows:
        n = r["name"].split("|")
        a, b, c, d = _fmt(r)
        out.append(f"| {n[2]} | {n[1]} | {a} | {b} | {c} | {d} |")
    return "\n".join(out)


def sweep_i8b(model):
    """block-scaled int8 (scale per 32/64/128 along K) vs per-token int8: accuracy only (no kernel was written for k-group scales)."""
    rows = [r for r in sweep_rows(model, "G6") if "H" in r]
    out = ["| 格式 | 种类 | 前缀行 | 门 M：IoU / conf | 门 H：IoU / min-frame / conf 最大 / >3% | H logit MAE | 判定 |", "|---|---|---|---|---|---:|---|"]
    for r in rows:
        n = r["name"].split("|")
        a, b, c, d = _fmt(r)
        out.append(f"| {n[1]} | {n[2]} | {n[4]} | {a} | {b} | {c} | {d} |")
    return "\n".join(out)


def plan_table(arg):
    """arg = 'name' -> backbone groups + head of profile/final_<name>.json with the go rule, plus the verification numbers."""
    d = J(f"profile/final_{arg}.json")
    if not d:
        return "(not available)"
    out = ["| GEMM 种类 | 档位 | 格式 / 技术 | 层数 | 每帧节省 ms（central / pess） | go 规则（≥0.15 ms） |", "|---|---|---|---:|---|---|"]
    for r in d["backbone_groups"]:
        sp = r["spec"]
        tech = "+".join(x for x in (f"SQ{sp['sq']}" if sp.get("sq") else "", f"Had{sp['had']}" if sp.get("had") else "", "前缀行bf16" if sp.get("pfx") else "", "GPTQ" if sp.get("gptq") else "") if x) or "RN"
        sc, spp = r["saving_ms_central"], r["saving_ms_pess"]
        out.append(f"| {r['kind']} | {r['tier']} | {sp['fmt']} / {tech} | {r['n_layers']} | {'-' if sc is None else f'{sc:.3f}'} / {'-' if spp is None else f'{spp:.3f}'} | {'go' if r['go'] else 'no-go（回到 P0）'} |")
    for h in d["head"]:
        sc, spp = h["saving_ms_central"], h["saving_ms_pess"]
        out.append(f"| head {h['kind']} | {h['tier']} | {h['fmt']} / RN | - | {'-' if sc is None else f'{sc:.3f}'} / {'-' if spp is None else f'{spp:.3f}'} | {'go' if h['go'] else 'no-go（回到 P0）'} |")
    out.append(f"| **合计（go 项）** | | | | **{d['saving_ms_central_after_go_rule']:.3f} / {d['saving_ms_pess_after_go_rule']:.3f}** | |")
    return "\n".join(out)


def verify_table(_=None):
    out = ["| 方案（go 规则之后） | 仿真方式 | 门 M：IoU / conf 最大 | 门 H（110 帧）：IoU / min-frame IoU / conf 最大 / conf>3% 帧数 / logit MAE | 换校准集（SmoothQuant 方案）H IoU / conf | M+H |", "|---|---|---|---|---|---|"]
    for f in sorted((HERE / "profile").glob("final_*.json")):
        name = f.stem[len("final_"):]
        for suffix, label in (("", "bf16 舍入的反量化操作数"), ("_realkernel", "真实 mg_i8 kernel（int8 精确累加）")):
            v = J(f"profile/verify_{name}{suffix}.json")
            if not v:
                continue
            m, h = v["M"], v["H"]
            alt = v.get("alt_calibration")
            altc = "-" if not alt else f"{alt['H']['pooled_iou']:.4f} / {alt['H']['conf_max'] * 100:.2f}% (M {alt['M']['pooled_iou']:.4f})"
            out.append(f"| {name} | {label} | {m['pooled_iou']:.4f} / {m['conf_max'] * 100:.2f}% | {h['pooled_iou']:.4f} / {h['min_iou']:.3f} / {h['conf_max'] * 100:.2f}% / {h['n_conf_gt3']} / {h['logit_mae']:.2e} | {altc} | {'通过' if (m['pass'] and h['pass']) else '不过'} |")
    return "\n".join(out)


GEN = {"sweep_head": sweep_head, "sweep_i8b": sweep_i8b, "plan_table": plan_table, "verify_table": verify_table, "gemm_extra": gemm_extra, "gemm_b2": gemm_b2, "gemm_quack_fp4": gemm_quack_fp4, "perframe": perframe, "gemm_lowp1": gemm_lowp1, "mg_stage2": mg_stage2, "quant_costs": quant_costs, "sweep_A": sweep_A, "sweep_F": sweep_F, "sweep_best": sweep_best}


import tables2  # noqa: E402
GEN.update({'dec_groups': tables2.dec_groups, 'dec_summary': tables2.dec_summary, 'robust_table': tables2.robust_table, 'verify2': tables2.verify2})
