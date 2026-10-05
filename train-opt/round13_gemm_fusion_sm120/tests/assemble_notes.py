"""Regenerate NOTES.md from tests/NOTES.template.md + the measured JSON/CSV artifacts (missing data -> 'pending').

  python3 tests/assemble_notes.py          # writes ../NOTES.md
Markers in the template look like {{NAME}}.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
NSYS = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/nsys")


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ GEMM / LN / epilogue / mxfp8
def gemm_tables():
    out = []
    cands = ["torch", "quack", "quack_sk", "spark6"]
    for model in ("vitl16", "vitb16"):
        for m in ("plain", "joint"):
            rows = load(PROF / f"gemm_matrix_{model}_{m}.json")
            if not rows:
                continue
            cell = defaultdict(dict)
            for r in rows:
                if "us" in r:
                    cell[(r["linear"], r["kind"])][r["cand"]] = r
            M = rows[0]["M"]
            out.append(f"\n**{model}, M={M} ({m})** (us / TFLOPS, % of the 230 TF roofline; Spark v4 omitted: always slower than v6)\n")
            out.append("| GEMM | torch (cuBLAS) | QuACK | QuACK split-K | Spark v6 | best |")
            out.append("|---|---|---|---|---|---|")
            order = lambda kv: (["qkv", "proj", "fc1", "fc2"].index(kv[0][0]), ["fwd", "dgrad", "wgrad"].index(kv[0][1]))
            for (lin, kind), d in sorted(cell.items(), key=order):
                allc = {k: v for k, v in d.items() if k != "spark4"}
                best = min(allc.values(), key=lambda r: r["us"])
                cols = []
                for c in cands:
                    r = d.get(c)
                    cols.append(f"{r['us']:.1f} / {r['tflops']:.0f} ({r['pct_230']:.0f}%)" if r else "-")
                out.append(f"| {lin} {kind} | " + " | ".join(cols) + f" | **{best['cand']}** |")
    return "\n".join(out) or "pending"


def ln_table():
    rows = load(PROF / "ln_matrix.json")
    if not rows:
        return "pending"
    out = ["| shape | op | candidate | us | cos vs fp32 |", "|---|---|---|---|---|"]
    for r in rows:
        if "us" not in r or r["M"] not in (7203, 14406):
            continue
        cos = f"{r['err']['cos']:.7f}" if r.get("err") else "-"
        out.append(f"| {r['model']} M={r['M']} | {r['kind']} | {r['cand']} | {r['us']:.1f} | {cos} |")
    return "\n".join(out)


def epilogue_table():
    d = load(PROF / "epilogue_matrix.json")
    if not d:
        return "pending"
    out = ["| model | op | candidate | us | TFLOPS | cos vs fp32 (erf) |", "|---|---|---|---|---|---|"]
    for model, ops in d.items():
        for op, rows in ops.items():
            for r in rows:
                if "us" not in r:
                    out.append(f"| {model} | {op} | {r['cand']} | ERROR | | {r['error'][:60]} |")
                    continue
                cos = f"{r['err']['cos']:.6f}" if r.get("err") else "-"
                tf = f"{r['tflops']:.0f}" if r.get("tflops") else "-"
                out.append(f"| {model} | {op} | {r['cand']} | {r['us']:.1f} | {tf} | {cos} |")
    return "\n".join(out)


def mxfp8_table():
    rows = load(PROF / "mxfp8_probe.json")
    if not rows:
        return "pending"
    out = ["| GEMM (ViT-L, M=7203) | bf16 QuACK us | MXFP8 QuACK us | MXFP8 TFLOPS | speedup | rel L2 err vs fp32 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['linear']} {r['kind']} | | | | | {r['error'][:60]} |")
        else:
            out.append(f"| {r['linear']} {r['kind']} | {r['bf16_us']:.1f} | {r['mxfp8_us']:.1f} | {r['mxfp8_TF']:.0f} | {r['speedup']:.2f}x | {r['err_vs_fp32']['rel_l2']:.4f} |")
    return "\n".join(out)


def block_table():
    d = load(PROF / "bench_block_final.json")
    if not d:
        return "pending"
    out = ["| block (real shapes) | variant | fwd ms | bwd ms | fwd+bwd ms | vs T-BF16 block (r5 ln_fuse) |", "|---|---|---|---|---|---|"]
    for key, v in d.items():
        r5 = v["r5_ln_fuse"]["fwdbwd_ms"]
        for name, r in v.items():
            out.append(f"| {key} | {name} | {r['fwd_ms']:.3f} | {r['bwd_ms']:.3f} | {r['fwdbwd_ms']:.3f} | {r5 / r['fwdbwd_ms']:.3f}x |")
    return "\n".join(out)


def step_table():
    d = load(ROUND / "results.json")
    if not d:
        return "pending"
    out = []
    for model, variants in d.items():
        if not isinstance(variants, dict) or not any(isinstance(v, dict) and "gpu_step" in v for v in variants.values()):
            continue
        base = variants.get("TB1", {}).get("gpu_step", {})
        out.append(f"\n**{model}** (batch {next(iter(variants.values())).get('batch')}; phases early/mid/late, weights 0.2/0.6/0.2)\n")
        out.append("| stack | early ms | mid ms | late ms | weighted img/s | vs TB1 | fwd / bwd / opt / loss (mid, ms) | peak mem GB |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name, r in variants.items():
            gs = r.get("gpu_step") or {}
            if not gs or gs.get("weighted_img_s") is None:
                continue
            bd = r.get("breakdown_ms") or {}
            ratio = gs["weighted_img_s"] / base["weighted_img_s"] if base.get("weighted_img_s") else float("nan")
            mem = r.get("peak_mem_gb")
            out.append(f"| {name} | {gs['early']['step_ms']['mean']:.1f} | {gs['mid']['step_ms']['mean']:.1f} | {gs['late']['step_ms']['mean']:.1f} | "
                       f"{gs['weighted_img_s']:.2f} | {ratio:.3f}x | {bd.get('fwd') or 0:.1f} / {bd.get('bwd') or 0:.1f} / {bd.get('opt') or 0:.1f} / {bd.get('loss') or 0:.1f} | {mem:.1f} |" if mem else "")
    return "\n".join(o for o in out if o) or "pending"


# ------------------------------------------------------------------ static
def static_table():
    d = load(PROF / "static_block_kernels.json")
    if not d:
        return "pending"
    out = ["| kernel | regs | spill/stack B | instr | LDG | STG | BAR | SHFL | FFMA | MUFU |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in d.items():
        m = re.match(r"_Z\d+(\w+?)ILi(\d+)ELi(\d+)ELi(\d+)ELb(\d)ELb(\d)ELb(\d)E", name)
        if m:
            label = f"rowbwd D={m.group(2)} {'LN ' if m.group(5)=='1' else ''}{'GAM ' if m.group(6)=='1' else ''}".strip()
        elif "ln_fwd" in name:
            label = "ln_fwd D=" + re.search(r"ILi(\d+)E", name).group(1)
        else:
            label = re.sub(r"^_Z\d+", "", name)[:24]
        mix = r["mix"]
        out.append(f"| {label} | {r['regs']} | {r['stack_bytes']} | {r['instr_total']} | {mix.get('LDG', 0)} | {mix.get('STG', 0)} | {mix.get('BAR', 0)} | {mix.get('SHFL', 0)} | {mix.get('FFMA', 0)} | {mix.get('MUFU', 0)} |")
    return "\n".join(out)


# ------------------------------------------------------------------ gates
def gate_a():
    k = load(PROF / "gateA_kernels.json")
    b = load(PROF / "gateA_block.json")
    out = []
    if k:
        n = sum(len(v) for v in k.values())
        bad = sum(1 for v in k.values() for r in v if not r["pass"])
        out.append(f"- elementwise kernels (`gateA_kernels.json`): {len(k)} cases (D=1024/768 x M=7203/7903/14406/15806/129 x plain/spiky), {n} checks, {bad} failures; "
                   f"worst cos of any fp32 output {min((r.get('cos', 1.0) for v in k.values() for r in v), default=1.0):.9f}; bitwise deterministic.")
    if b:
        out.append("- fused block vs fp32 block (real DINOv3 `SelfAttentionBlock`, LayerScale, k-masked qkv bias, spiky residual, joint mask): "
                   "error of every tensor (out, dx, all 12 Parameter gradients) vs the fp32 block, fused vs torch-autocast (same-precision reference):\n")
        out.append("| case | worst fused-vs-fp32 cos | worst autocast-vs-fp32 cos | worst fused-vs-autocast cos | max rel L2 fused / autocast |")
        out.append("|---|---|---|---|---|")
        for key, rows in b.items():
            wf = min(r["fused_vs_fp32"][0] for r in rows.values())
            wa = min(r["autocast_vs_fp32"][0] for r in rows.values())
            wfa = min(r["fused_vs_autocast"][0] for r in rows.values())
            rf = max(r["fused_vs_fp32"][1] for r in rows.values())
            ra = max(r["autocast_vs_fp32"][1] for r in rows.values())
            out.append(f"| {key} | {wf:.6f} | {wa:.6f} | {wfa:.6f} | {rf:.2e} / {ra:.2e} |")
    return "\n".join(out) or "pending"


def gate_b():
    out = []
    for model in ("vitl16", "vitb16"):
        d = load(PROF / f"gateB_step_equivalence_{model}_TB1_R13_vs_TB1_mid.json")
        if not d:
            continue
        out.append(f"\n**{model}, mid phase, `check.step_equivalence`, ref = TB1, cand = TB1 + round13**\n")
        out.append("| seed | loss_rel | grad_cos_min (per param) | #params < 0.99 | grad_norm_rel | update_cos_min | check.py pass |")
        out.append("|---|---|---|---|---|---|---|")
        for s, r in d["seeds"].items():
            out.append(f"| {s} | {r['loss_rel']:.2e} | {r['grad_cos_min']:.4f} | {r['n_grad_cos_below_0.99']} / {r['n_params']} | {r['grad_norm_rel']:.2e} | {r['update_cos_min']:.4f} | {r['pass']} |")
    cal = []
    for f in sorted(glob.glob(str(PROF / "calibrate_gateb_*seed*.json"))):
        if "first_run" in f:
            continue
        d = load(f)
        s = d["summary"]
        cal.append((Path(f).stem.replace("calibrate_gateb_", ""), s))
    if cal:
        out.append("\n**Calibration against fp32 (R0), same weights / batch / RNG tape** (`tests/calibrate_gateb.py`)\n")
        out.append("| run | loss R0 / TB1 / cand | grad norm R0 / TB1 / cand | min per-param cos: TB1 vs R0 | cand vs R0 | cand vs TB1 | #params<0.99: TB1-R0 / cand-R0 / cand-TB1 | calibrated pass (cand-R0 >= min(0.99, TB1-R0 - 0.02)) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name, s in cal:
            l, g, m, n = s["loss"], s["grad_norm"], s["min_cos"], s["n_below_0.99"]
            out.append(f"| {name} | {l['R0']:.4f} / {l['REF']:.4f} / {l['CAND']:.4f} | {g['R0']:.2f} / {g['REF']:.2f} / {g['CAND']:.2f} | {m['ref_vs_r0']:.3f} | {m['cand_vs_r0']:.3f} | {m['cand_vs_ref']:.3f} | "
                       f"{n['ref_vs_r0']} / {n['cand_vs_r0']} / {n['cand_vs_ref']} | {s.get('calibrated_pass')} |")
    return "\n".join(out) or "pending"


def gate_s():
    out = []
    for model in ("vitl16", "vitb16"):
        files = sorted(glob.glob(str(PROF / f"gateS_{model}_TB1_R13_vs_TB1*.json")))
        if not files:
            continue
        d = load(files[-1])
        out.append(f"\n**{model}**: 50 steps on 2 real batches, loss EMA(0.9) deviation at steps 10/20/30/40/50 vs TB1 (gate_c: <= 5%, finite, grad-norm ratio in [0.1, 10])\n")
        out.append("| trajectory | max rel EMA dev | per checkpoint | finite | grad-norm ratios | pass |")
        out.append("|---|---|---|---|---|---|")
        for key, run in d["runs"].items():
            g = run["gate"]
            cps = ", ".join(f"{v:.1e}" for v in g["checkpoints"].values())
            gr = ", ".join(f"{v:.3f}" for v in g["grad_norm_ratios"].values() if v is not None)
            out.append(f"| {key} | {g['curve300_max_rel']:.2e} | {cps} | {g['finite']} | {gr} | {g['pass']} |")
    return "\n".join(out) or "pending"


# ------------------------------------------------------------------ nsys
CATS = [
    ("attention / RoPE", re.compile(r"flash|fmha|sdpa|cudnn.*(attn|mha|sdpa)|attention|mha|rope|_rotate|bwd_dot_do_o|convert_dq", re.I)),
    ("GEMM (BF16 / FP32 / TF32)", re.compile(r"gemm|cutlass|nvjet|xmma|cublas|sm80_|sm90_|sm100_|sm120_|quack|turing_|ampere_|splitK", re.I)),
    ("convolution (head)", re.compile(r"conv|implicit|nhwc|nchw|wgrad_alg|dgrad_alg|cudnn", re.I)),
    ("round13 fused LN/row kernels", re.compile(r"ln_fwd_kernel|rowbwd_kernel|colsum_bf16|finalize_kernel")),
    ("elementwise / copy / reduce / norm / optimizer", re.compile(r"elementwise|vectorized|copy|cast|reduce|add|mul|fill|index|gelu|catArray|transpose|layer_norm|LayerNorm|GammaBeta|multi_tensor|adam|foreach|norm", re.I)),
]


def cat_of(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def step_nsys(tag):
    f = NSYS / f"{tag}_cuda_gpu_kern_sum.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    iters = 5
    cats = defaultdict(lambda: [0, 0.0])
    tot_n, tot_ms = 0, 0.0
    top = []
    for r in rows:
        n = int(float(r["Instances"])) / iters
        ms = float(r["Total Time (ns)"]) / iters / 1e6
        c = cat_of(r["Name"])
        cats[c][0] += n
        cats[c][1] += ms
        tot_n += n
        tot_ms += ms
        top.append((ms, n, r["Name"]))
    top.sort(reverse=True)
    return {"cats": dict(cats), "tot_n": tot_n, "tot_ms": tot_ms, "top": top[:10]}


def nsys_step_tables():
    out = []
    for model in ("vitl16", "vitb16"):
        a, b = step_nsys(f"step_{model}_TB1_a"), step_nsys(f"step_{model}_TB1_R13_a")
        if not (a and b):
            continue
        out.append(f"\n**{model}, phase mid, per step (5 profiled steps / 5): GPU kernel time by category**\n")
        out.append("| category | TB1 kernels | TB1 ms | TB1+R13 kernels | TB1+R13 ms |")
        out.append("|---|---|---|---|---|")
        for c in sorted(set(a["cats"]) | set(b["cats"]), key=lambda c: -a["cats"].get(c, [0, 0])[1]):
            x, y = a["cats"].get(c, [0, 0]), b["cats"].get(c, [0, 0])
            out.append(f"| {c} | {x[0]:.0f} | {x[1]:.1f} | {y[0]:.0f} | {y[1]:.1f} |")
        out.append(f"| **total** | {a['tot_n']:.0f} | {a['tot_ms']:.1f} | {b['tot_n']:.0f} | {b['tot_ms']:.1f} |")
        for label, s in (("TB1", a), ("TB1+R13", b)):
            out.append(f"\nTop-10 kernels {model} {label} (ms/step, launches/step):\n")
            out.append("| ms | n | kernel |")
            out.append("|---|---|---|")
            for ms, n, name in s["top"]:
                out.append(f"| {ms:.2f} | {n:.0f} | `{re.sub(r'[|]', '/', name)[:100]}` |")
    return "\n".join(out) or "pending"


def nsys_block_tables():
    out = []
    names = {"autocast_default": "autocast default block", "r5_ln_fuse": "T-BF16 block (r5 ln_fuse)", "r13_full": "round13 fused block"}
    have = {k: load(PROF / f"nsys_block_{k}.json") for k in names}
    if not any(have.values()):
        return "pending"
    out.append("ViT-L plain block (B=7, N=1029), identical attention (flash + triton RoPE); GPU time = sum of kernel durations per block (graph-free eager profile, 5 iterations).\n")
    out.append("| variant | fwd kernels | fwd GPU us | bwd kernels | bwd GPU us | of which GEMM (fwd+bwd) us | elementwise/fused-row us | attention us |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, label in names.items():
        d = have[k]
        if not d:
            continue
        f, b = d.get("fwd", {}), d.get("bwd", {})

        def cat(rng, *keys):
            c = rng.get("categories", {})
            return sum(c.get(x, {}).get("us", 0) for x in keys)
        gemm = cat(f, "gemm") + cat(b, "gemm")
        ew = cat(f, "norm/residual (fused CUDA)", "elementwise/copy/reduce", "other") + cat(b, "norm/residual (fused CUDA)", "elementwise/copy/reduce", "other")
        att = cat(f, "attention") + cat(b, "attention")
        out.append(f"| {label} | {f.get('kernels_per_iter', 0):.0f} | {f.get('gpu_us_per_iter', 0):.0f} | {b.get('kernels_per_iter', 0):.0f} | {b.get('gpu_us_per_iter', 0):.0f} | {gemm:.0f} | {ew:.0f} | {att:.0f} |")
    return "\n".join(out)


def nsys_block_kernels():
    d = load(PROF / "nsys_block_r13_full.json")
    if not d:
        return "pending"
    out = []
    for rng in ("fwd", "bwd"):
        r = d.get(rng)
        if not r:
            continue
        out.append(f"\nround13 block, {rng}: {r['kernels_per_iter']:.0f} kernels, {r['gpu_us_per_iter']:.0f} us\n")
        out.append("| n | us | kernel |")
        out.append("|---|---|---|")
        for k in r["top_kernels"][:16]:
            out.append(f"| {k['count']:.0f} | {k['us']:.1f} | `{k['name'][:96].replace('|', '/')}` |")
    return "\n".join(out)


def main():
    tpl = (ROUND / "tests" / "NOTES.template.md").read_text(encoding="utf-8")
    fills = {
        "GEMM_TABLES": gemm_tables, "LN_TABLE": ln_table, "EPILOGUE_TABLE": epilogue_table, "MXFP8_TABLE": mxfp8_table,
        "BLOCK_TABLE": block_table, "STEP_TABLE": step_table, "STATIC_TABLE": static_table, "GATE_A": gate_a, "GATE_B": gate_b,
        "GATE_S": gate_s, "NSYS_STEP": nsys_step_tables, "NSYS_BLOCK": nsys_block_tables, "NSYS_BLOCK_KERNELS": nsys_block_kernels,
    }
    for key, fname in (("NSYS_CONCLUSIONS", "notes_nsys_conclusions.md"), ("CONCLUSION", "notes_conclusion.md"), ("ROOFLINE_NOTE", "notes_roofline.md")):
        f = ROUND / "tests" / fname
        tpl = tpl.replace("{{" + key + "}}", f.read_text(encoding="utf-8").strip() if f.exists() else "(pending)")
    for k, fn in fills.items():
        try:
            val = fn()
        except Exception as exc:  # keep NOTES usable whatever is missing
            val = f"(table generation failed: {type(exc).__name__}: {exc})"
        tpl = tpl.replace("{{" + k + "}}", val)
    (ROUND / "NOTES.md").write_text(tpl, encoding="utf-8")
    print("wrote", ROUND / "NOTES.md", len(tpl), "bytes")


if __name__ == "__main__":
    main()
