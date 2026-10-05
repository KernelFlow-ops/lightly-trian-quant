"""train-opt figures (7). Every number is copied from train-opt NOTES / results / CONTRACTS; sources are printed on each figure."""
from __future__ import annotations

import numpy as np  # noqa: F401
from matplotlib.patches import Rectangle  # noqa: F401

from .style import (  # noqa: F401
    ARROW, FMT, GRAY, INK, INK2, LABEL, MUTED, PAPER, RULE, SOFT, arrow, bar_label, barh_fmt, box, callout, canvas, fig,
    fmt_bar, legend_fmt, line, note, on, panel_title, ptext, render, source, star, star_marker, style_axes, text, textbox,
    tint, title,
)

# ----------------------------------------------------------------------------------------------------
# train_01  one training step
# ----------------------------------------------------------------------------------------------------


@fig("train_01_overview", 12, 8.8)
def train_01(fig):
    title(fig, "One training step: 430 ms down to 86 ms",
          "DINOv3 + EoMT semantic segmentation, ViT-L/16, batch 7, RTX 5090 (mid-training phase)")

    # ---- (a) step time -------------------------------------------------------------------------
    a = fig.add_axes([0.165, 0.50, 0.335, 0.30])
    style_axes(a, grid="x")
    panel_title(a, "(a) Step time", "ms per step, GPU only, same harness")
    segs = [("fwd", 117.2, "#e7e2d3"), ("loss", 21.5, "#cfc8b2"), ("bwd", 268.8, "#b9b19b"), ("opt", 22.3, "#9b937e")]
    left = 0.0
    for name, v, c in segs:
        a.barh(2, v, left=left, height=0.55, color=c, edgecolor=INK, linewidth=1.6, zorder=3,
               hatch="//" if name == "bwd" else None)
        if name == "loss":   # narrow segment: caption below the bar
            a.text(left + v / 2, 1.62, f"{name} {v:.0f}", fontsize=11.5, ha="center", va="top")
        else:
            a.text(left + v / 2, 2.42, f"{name} {v:.0f}", fontsize=11.5, ha="center", va="bottom")
        left += v
    barh_fmt(a, 1, 102.4, "MXFP8", height=0.55)
    barh_fmt(a, 0, 85.6, "NVFP4", height=0.55)
    bar_label(a, 430.5 + 8, 2, "430 ms", ha="left", va="center", size=13.5)
    bar_label(a, 102.4 + 8, 1, "102 ms   4.2×", ha="left", va="center", size=13.5)
    bar_label(a, 85.6 + 8, 0, "86 ms   5.0×", ha="left", va="center", size=13.5)
    a.set_xlim(0, 560)
    a.set_ylim(-0.55, 3.05)
    a.set_yticks([2, 1, 0])
    a.set_yticklabels(["R0 original\n(fp32 / TF32)", "FINAL\n(MXFP8)", "P3\n(NVFP4)"], fontsize=12)
    a.set_xticks([0, 100, 200, 300, 400])
    callout(a, (330, 1.72), (290, 0.42), "fp32 attention backward\nalone = 86 ms of the\n269 ms backward", size=10.6, rad=-0.2)

    # ---- (b) inside the P3 step ------------------------------------------------------------------
    b = fig.add_axes([0.745, 0.50, 0.23, 0.30])
    style_axes(b, grid="x")
    panel_title(b, "(b) Inside the P3 step", "83.2 ms of kernels (nsys)")
    cats = [("attention (flash)", 16.06, GRAY[3], None), ("GEMM NVFP4", 10.56, FMT["NVFP4"], None),
            ("GEMM MXFP8", 9.65, FMT["MXFP8"], None), ("FP4 quantize", 8.50, FMT["NVFP4"], "//"),
            ("optimizer (flat)", 7.72, GRAY[2], None), ("fused head kernels", 6.73, GRAY[2], None),
            ("elementwise, other", 6.64, GRAY[1], None), ("LayerNorm / rows", 5.47, GRAY[1], None),
            ("GEMM BF16 (qkv fwd)", 4.79, FMT["BF16"], None), ("GEMM BF16 (head)", 3.10, FMT["BF16"], None),
            ("loss, patch conv", 2.21, GRAY[0], None), ("MXFP8 quantize", 1.73, FMT["MXFP8"], "//")]
    n = len(cats)
    for i, (lab, v, c, h) in enumerate(cats):
        y = n - 1 - i
        b.barh(y, v, height=0.62, color=c, edgecolor=INK, linewidth=1.4, hatch=h, zorder=3)
        b.text(v + 0.3, y, f"{v:.1f}", fontsize=11, fontweight="bold", va="center")
    b.set_yticks(range(n))
    b.set_yticklabels([c[0] for c in cats][::-1], fontsize=10.8)
    b.set_xlim(0, 21)
    b.set_ylim(-0.6, n - 0.3)
    b.set_xticks([0, 5, 10, 15, 20])

    # ---- tiles ---------------------------------------------------------------------------------
    ov = canvas(fig, 12, 8.8)
    tiles = [
        ("kernels per step", ["6,499", "2,475", "2,860"], "62% fewer than R0;\nFP4 adds quantize kernels"),
        ("GPU busy", ["88%", "98%", "97%"], "launch gaps closed by one\nwhole-step CUDA graph"),
        ("host syncs per step", ["132", "0", "0"], "matcher + loss no longer\nround-trip to the CPU"),
        ("peak reserved memory", ["30.0", "15.9", "15.0"], "GB; bf16 activations,\nfused head"),
    ]
    x0s = [0.45, 3.28, 6.11, 8.94]
    for x0, (ttl, vals, sub) in zip(x0s, tiles):
        box(ov, x0, 0.72, 2.62, 2.5, fc=PAPER, lw=1.7, r=0.12, z=2)
        ov.text(x0 + 0.16, 2.9, ttl, fontsize=13.5, fontweight="bold", va="center")
        for j, (v, k, lab) in enumerate(zip(vals, ["FP32", "MXFP8", "NVFP4"], ["R0", "FINAL", "P3"])):
            cx = x0 + 0.5 + j * 0.8
            ov.text(cx, 2.22, v, fontsize=16, fontweight="bold", ha="center", va="center")
            box(ov, cx - 0.32, 1.62, 0.64, 0.28, fc=FMT[k], lw=1.3, r=0.06, z=3)
            ptext(ov, cx, 1.76, lab, FMT[k], size=10.2, ha="center", va="center", fontweight="bold")
        ov.text(x0 + 0.16, 1.08, sub, fontsize=10, color=INK2, va="center", linespacing=1.2)
    source(fig, "round0 NOTES (R0 phases, kernels, syncs, busy; phase split = late phase), round19 NOTES (FINAL),\n"
           "round23 NOTES §4.2 / §4.9 (P3 and FINAL step times, kernel categories, memory).", y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_02  throughput waterfall
# ----------------------------------------------------------------------------------------------------

_STAGES = [
    ("R0  original Lightly step", "FP32"),
    ("P0  fp32 clean-up (r1,3,4,5,7)", "FP32"),
    ("TB1  BF16 autocast + flash attn", "BF16"),
    ("+ r13  fused blocks", "BF16"),
    ("+ r17  MXFP8 GEMMs", "MXFP8"),
    ("+ r15  flat optimizer", "MXFP8"),
    ("+ r16  whole-step CUDA graph", "MXFP8"),
    ("+ r14  fused head = FINAL", "MXFP8"),
    ("P3  NVFP4 head + dgrad/wgrad", "NVFP4"),
]


@fig("train_02_waterfall", 12, 8.2)
def train_02(fig):
    title(fig, "Training throughput, stage by stage",
          "weighted images per second (early / mid / late phases 0.2 / 0.6 / 0.2), batch fixed: ViT-L 7, ViT-B 14")
    data = {
        "L": ([16.31, 18.00, 41.40, 43.70, 51.70, 56.00, 59.50, 68.87, 81.87],
              ["1.00×", "1.10×", "2.54×", "2.68×", "3.17×", "3.43×", "3.65×", "4.22×", "5.04×"]),
        "B": ([37.77, 44.40, 96.00, 100.70, 109.90, 116.60, 123.10, 161.94, 191.40],
              ["1.00×", "1.18×", "2.54×", "2.67×", "2.91×", "3.09×", "3.26×", "4.29×", "5.06×"]),
    }
    axes = {"L": fig.add_axes([0.275, 0.14, 0.30, 0.64]), "B": fig.add_axes([0.665, 0.14, 0.31, 0.64])}
    xmax = {"L": 118, "B": 285}
    ttl = {"L": ("(a) ViT-L/16, batch 7", "img/s and × over R0"), "B": ("(b) ViT-B/16, batch 14", "img/s and × over R0")}
    n = len(_STAGES)
    for key, ax in axes.items():
        style_axes(ax, grid="x")
        panel_title(ax, *ttl[key])
        vals, mult = data[key]
        for i, ((name, fmt), v, m) in enumerate(zip(_STAGES, vals, mult)):
            y = n - 1 - i
            barh_fmt(ax, y, v, fmt, height=0.62)
            bar_label(ax, v + xmax[key] * 0.015, y, f"{v:.1f}   {m}", ha="left", va="center", size=12, weight="bold")
        ax.set_xlim(0, xmax[key])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(n))
        if key == "L":
            ax.set_yticklabels([s[0] for s in _STAGES][::-1], fontsize=11.5)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 25, 50, 75, 100] if key == "L" else [0, 50, 100, 150, 200])
    ov = canvas(fig, 12, 8.2)
    legend_fmt(ov, 3.3, 0.66, ["FP32", "BF16", "MXFP8", "NVFP4"], size=12.5)
    ov.text(7.55, 0.66, "colour = number format of the GEMMs at that stage", fontsize=11.5, color=INK2, va="center")
    source(fig, "round19_final/summary.md (R0 .. FINAL; earlier stages are the numbers each round published, ±2-3% drift between sessions);\n"
           "round23 NOTES §4.2 (P3, same-session × over R0).", y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_04  precision tiers vs the baseline
# ----------------------------------------------------------------------------------------------------


@fig("train_04_precision_tiers", 12, 8.8)
def train_04(fig):
    title(fig, "FP32 / BF16 / MXFP8 / NVFP4 / MXFP4 vs the baseline",
          "training throughput per number format, whole step under one CUDA graph; gradient cosine vs R0 below each bar")
    tiers = ["FP32", "BF16", "MXFP8", "NVFP4", "MXFP4"]
    cfg = {
        "L": dict(rect=[0.075, 0.355, 0.41, 0.425], vals=[16.3, 54.4, 68.9, 81.9], mult=["1.00×", "3.33×", "4.22×", "5.04×"],
                  med=["", "0.9997", "0.9960", "0.9914", "0.9921"], mn=["", "0.9928", "0.9854", "0.9823", "0.9796"],
                  ymax=118, title="(a) ViT-L/16, batch 7"),
        "B": dict(rect=[0.565, 0.355, 0.41, 0.425], vals=[37.8, 139.6, 161.9, 191.4], mult=["1.00×", "3.70×", "4.29×", "5.06×"],
                  med=["", "0.9993", "0.9947", "0.9895", "0.9874"], mn=["", "0.9954", "0.9804", "0.9749", "0.9758"],
                  ymax=275, title="(b) ViT-B/16, batch 14"),
    }
    for key, c in cfg.items():
        ax = fig.add_axes(c["rect"])
        style_axes(ax, grid="y")
        panel_title(ax, c["title"], "weighted img/s and × over R0")
        ym = c["ymax"]
        for i, v in enumerate(c["vals"]):
            fmt_bar(ax, i, v, tiers[i], width=0.64)
            ax.text(i, v + ym * 0.015, f"{v:.1f}", fontsize=14, fontweight="bold", ha="center", va="bottom")
            ax.text(i, v + ym * 0.085, c["mult"][i], fontsize=12.5, ha="center", va="bottom", color=INK2)
        # MXFP4: dashed empty bar at the NVFP4 height = "no end-to-end run"
        ax.bar(4, c["vals"][3], width=0.64, color=PAPER, edgecolor=INK, linewidth=1.6, linestyle=(0, (4, 3)), zorder=3)
        ax.text(4, c["vals"][3] * 0.5, "not\nmeasured", fontsize=12, ha="center", va="center", color=INK2, style="italic")
        ax.set_xlim(-0.55, 4.55)
        ax.set_ylim(0, ym)
        ax.set_xticks(range(5))
        ax.set_xticklabels(tiers, fontsize=12.5, fontweight="bold")
        # gradient cosine vs R0 under each bar: median row and min row
        for i in range(1, 5):
            ax.text(i, -ym * 0.155, c["med"][i], fontsize=11, ha="center", va="top", fontweight="bold", clip_on=False)
            ax.text(i, -ym * 0.235, c["mn"][i], fontsize=11, ha="center", va="top", clip_on=False, color=INK2)
        ax.text(-0.42, -ym * 0.155, "cos median", fontsize=10.2, ha="right", va="top", color=INK, clip_on=False)
        ax.text(-0.42, -ym * 0.235, "cos min", fontsize=10.2, ha="right", va="top", color=INK2, clip_on=False)
    textbox(fig.axes[0], -0.45, 106, "R0 is fp32 / TF32: bf16 autocast was configured\nbut never ran inside Lightly's training step", size=10.4, ha="left")

    # what each tier changes
    ov = canvas(fig, 12, 8.8)
    what = [("FP32", "R0", "fp32 / TF32 everywhere\nthe original Lightly step"),
            ("BF16", "FINAL_noMX", "autocast + flash attention\nfused blocks, flat optimizer,\nwhole-step CUDA graph"),
            ("MXFP8", "FINAL", "+ block-scaled GEMMs\nE4M3 values, UE8M0 scale\nper 1×32"),
            ("NVFP4", "P3", "+ dgrad / wgrad in e2m1\nE4M3 scale per 16, SR on dY\nFP4 head (ConvT)"),
            ("MXFP4", "R20b †", "e2m1 + UE8M0 per 32\nGEMMs only 3-7% faster\nthan NVFP4")]
    cw, gap, x = 2.1, 0.15, 0.45
    for k, a_, b_ in what:
        box(ov, x, 0.62, cw, 1.35, fc=PAPER, lw=1.5, r=0.1, z=2)
        box(ov, x, 0.62, 0.14, 1.35, fc=FMT[k], lw=1.5, r=0.06, z=3)
        ov.text(x + 0.26, 1.78, k, fontsize=12.5, fontweight="bold", va="center")
        ov.text(x + 0.26 + 0.118 * len(k) + 0.1, 1.78, a_, fontsize=9.8, va="center", color=INK2)
        ov.text(x + 0.26, 1.2, b_, fontsize=9.6, va="center", linespacing=1.2)
        x += cw + gap
    source(fig, "round19 summary (FINAL_noMX = BF16, FINAL = MXFP8); round23 §4.2 / §4.4 (P3 = NVFP4; cosine vs R0 over 24 batches, P1 / FINAL / P3).\n"
           "† MXFP4: round20 R20b arm B4+mx4, emulated (torch quantiser + real FP4 GEMM), 8 batches; no whole-step measurement.", y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_06  gradient fidelity
# ----------------------------------------------------------------------------------------------------


@fig("train_06_gradient_fidelity", 12, 8.8)
def train_06(fig):
    title(fig, "Does the gradient still point the same way?",
          "global gradient cosine vs the original FP32 step (R0): same weights, same batches; 24 batches per dot (MXFP4: 8, emulated)")
    tiers = [("BF16", "BF16\n(P1)"), ("MXFP8", "MXFP8\n(FINAL)"), ("NVFP4", "NVFP4\nhead\n(P3H)"), ("NVFP4", "NVFP4\nbackbone\n(P3B)"),
             ("NVFP4", "NVFP4\nall\n(P3)"), ("MXFP4", "MXFP4\nemulated")]
    data = {
        "L": dict(med=[0.9997, 0.9960, 0.9931, 0.9945, 0.9914, 0.9921], mn=[0.9928, 0.9854, 0.9850, 0.9846, 0.9823, 0.9796]),
        "B": dict(med=[0.9993, 0.9947, 0.9921, 0.9922, 0.9895, 0.9874], mn=[0.9954, 0.9804, 0.9770, 0.9769, 0.9749, 0.9758]),
    }
    layout = [("L", "med", [0.075, 0.585, 0.32, 0.215], "(a) ViT-L: median of 24 batches", (0.976, 1.0045), (0.99, 0.98), "median"),
              ("L", "mn", [0.575, 0.585, 0.31, 0.215], "(b) ViT-L: worst of 24 batches", (0.960, 1.0045), (0.98, 0.97), "min"),
              ("B", "med", [0.075, 0.205, 0.32, 0.215], "(c) ViT-B: median of 24 batches", (0.976, 1.0045), (0.99, 0.98), "median"),
              ("B", "mn", [0.575, 0.205, 0.31, 0.215], "(d) ViT-B: worst of 24 batches", (0.960, 1.0045), (0.98, 0.97), "min")]
    for model, which, rect, ttl, ylim, (strict, relaxed), word in layout:
        ax = fig.add_axes(rect)
        style_axes(ax, grid="y")
        panel_title(ax, ttl)
        vals = data[model][which]
        ax.axhspan(ylim[0], relaxed, color=tint(INK, 0.07), zorder=1)
        ax.axhline(strict, color=INK, lw=1.5, ls=(0, (5, 4)), zorder=2)
        ax.axhline(relaxed, color=INK, lw=1.5, ls=(0, (2, 3)), zorder=2)
        ax.text(5.62, strict, f"strict ≥ {strict:.2f}", fontsize=10.6, ha="left", va="center", clip_on=False)
        ax.text(5.62, relaxed, f"relaxed ≥ {relaxed:.2f}", fontsize=10.6, ha="left", va="center", clip_on=False)
        for i, ((k, lab), v) in enumerate(zip(tiers, vals)):
            if k == "MXFP4":
                ax.plot([i], [v], marker="D", ms=14, mfc=PAPER, mec=INK, mew=1.6, zorder=6, linestyle="none")
                ax.plot([i], [v], marker="D", ms=8, mfc=FMT[k], mec=FMT[k], zorder=7, linestyle="none")
            else:
                ax.plot([i], [v], marker="o", ms=16, mfc=FMT[k], mec=INK, mew=1.6, zorder=6, linestyle="none")
            ax.text(i, v + (ylim[1] - ylim[0]) * 0.06, f"{v:.4f}", fontsize=10.8, fontweight="bold", ha="center", va="bottom")
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(6))
        ax.set_xticklabels([t[1] for t in tiers], fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        if which == "mn" and model == "B":
            textbox(ax, 5.45, 0.9625, "40 batches: worst P3 = 0.9749,\nonly 0.0049 above the relaxed line", size=10, ha="right")
        if which == "mn" and model == "L":
            textbox(ax, 5.45, 0.9625, "40 batches: worst P3 0.9731 vs R0, 0.9693 vs P1\n(one outlier batch, FINAL itself 0.9643)", size=10, ha="right")
        if which == "med" and model == "B":
            textbox(ax, 5.45, 0.9785, "P3 median 0.9895: relaxed pass,\nstrict fail", size=10, ha="right")
    ov = canvas(fig, 12, 8.8)
    ov.plot([0.6], [0.78], marker="D", ms=11, mfc=PAPER, mec=INK, mew=1.4)
    ov.text(0.85, 0.78, "= emulated (round20 R20b: torch quantiser + real FP4 GEMM, 8 batches); filled circles are the real stacks",
            fontsize=10.8, va="center", color=INK2)
    ov.text(0.45, 0.5, "grey zone = below the relaxed gate.   dashed = strict gate, dotted = relaxed gate (CONTRACTS 11.5).",
            fontsize=10.8, va="center", color=INK2)
    source(fig, "round23 NOTES §4.4 (24 merged batches; P1 = BF16 stack, P3H / P3B = head only / backbone only) and gate_b_extra (40-batch worst cases); round20 R20b (mx4 arm).",
           y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_05  GEMM speed and error
# ----------------------------------------------------------------------------------------------------


@fig("train_05_gemm_speed_error", 12, 10.6)
def train_05(fig):
    title(fig, "GEMMs: what each number format buys, and what it costs",
          "ViT-L/16 training shapes on an RTX 5090; speed in CUDA-graph replay, error as relative L2 vs an fp32 product")
    # ---- (a) one block, 12 GEMMs, bf16 vs MXFP8 --------------------------------------------------
    a = fig.add_axes([0.075, 0.605, 0.40, 0.21])
    style_axes(a, grid="y")
    panel_title(a, "(a) One ViT-L block: 12 GEMMs, BF16 vs MXFP8", "µs per block (M = 7203); stacked fwd / dgrad / wgrad, dark = wgrad")
    kinds = ["qkv", "proj", "fc1", "fc2"]
    bf = {"qkv": [195.5, 217.5, 215.0], "proj": [73.9, 75.1, 86.6], "fc1": [266.8, 287.7, 307.1], "fc2": [289.5, 266.1, 304.0]}
    mx = {"qkv": [74.4, 82.0, 113.1], "proj": [26.8, 30.1, 48.1], "fc1": [109.6, 108.7, 118.1], "fc2": [94.7, 119.3, 117.1]}
    shade = [0.4, 0.7, 1.0]
    for i, kd in enumerate(kinds):
        for off, dd, fk in [(-0.2, bf, "BF16"), (0.2, mx, "MXFP8")]:
            bot = 0.0
            for j, v in enumerate(dd[kd]):
                a.bar(i + off, v, bottom=bot, width=0.34, color=tint(FMT[fk], shade[j]), edgecolor=INK, linewidth=1.4, zorder=3)
                bot += v
            a.text(i + off, bot + 14, f"{bot:.0f}", fontsize=11.5, fontweight="bold", ha="center", va="bottom")
    a.set_xticks(range(4))
    a.set_xticklabels(["qkv\n2.33×", "proj\n2.24×", "fc1\n2.56×", "fc2\n2.60×"], fontsize=11.5)
    a.set_ylim(0, 1200)
    a.set_xlim(-0.6, 3.6)
    a.set_yticks([0, 250, 500, 750, 1000])
    textbox(a, 1.5, 1105, "all 12: 2,585 → 1,042 µs (2.48×); with 203 µs of quantising: 2.08×", size=10.4, ha="center")
    # ---- (c) error ------------------------------------------------------------------------------
    c = fig.add_axes([0.575, 0.605, 0.40, 0.21])
    style_axes(c, grid="y")
    panel_title(c, "(c) Single-GEMM error", "relative L2 vs fp32, median over ViT-L backbone Linears")
    roles = ["forward", "dgrad", "wgrad"]
    err = {"MXFP8": [0.029, 0.034, 0.022], "NVFP4": [0.114, 0.135, 0.080], "MXFP4": [0.186, 0.199, 0.156]}
    for i, r in enumerate(roles):
        for j, k in enumerate(["MXFP8", "NVFP4", "MXFP4"]):
            x = i + (j - 1) * 0.26
            c.bar(x, err[k][i], width=0.24, color=FMT[k], edgecolor=INK, linewidth=1.5, zorder=3)
            c.text(x, err[k][i] + 0.004, f"{err[k][i]:.3f}", fontsize=10, fontweight="bold", ha="center", va="bottom")
    c.set_xticks(range(3))
    c.set_xticklabels(roles, fontsize=12)
    c.set_ylim(0, 0.31)
    c.set_xlim(-0.55, 2.55)
    c.set_yticks([0, 0.1, 0.2])
    textbox(c, 2.5, 0.285, "dgrad NVFP4 with SR: 0.162;  wgrad worst case (block 0): 0.60", size=10, ha="right")
    ov = canvas(fig, 12, 10.6)
    legend_fmt(ov, 1.3, 5.62, ["BF16", "MXFP8"], size=11.5, sw=0.2, gap=0.25)
    legend_fmt(ov, 7.3, 5.62, ["MXFP8", "NVFP4", "MXFP4"], size=11.5, sw=0.2, gap=0.25)
    # ---- (b) per-step GEMM time by family --------------------------------------------------------
    b = fig.add_axes([0.215, 0.075, 0.755, 0.385])
    style_axes(b, grid="x")
    panel_title(b, "(b) GEMM time per training step, by family", "ms per step (ViT-L, CUDA-graph replay); hatched = measured but not adopted")
    rows = [
        ("backbone dgrad", [("MXFP8", 8.42, 0), ("NVFP4", 4.60, 0)], "in P3"),
        ("backbone wgrad", [("MXFP8", 8.81, 0), ("NVFP4", 4.49, 0)], "in P3"),
        ("head dgrad", [("BF16", 5.35, 0), ("NVFP4", 1.06, 0)], "in P3"),
        ("head wgrad", [("BF16", 5.67, 0), ("NVFP4", 1.08, 0)], "in P3"),
        ("head forward", [("BF16", 5.37, 0), ("MXFP8", 1.98, 0), ("NVFP4", 1.07, 1)], "MXFP8 in P3"),
        ("qkv forward", [("BF16", 4.84, 0), ("MXFP8", 1.83, 0), ("NVFP4", 0.94, 1)], "MXFP8: optional P3_q8"),
        ("fc1 / fc2 forward", [("MXFP8", 6.53, 0), ("NVFP4", 3.80, 1)], "NVFP4 fails the gate"),
    ]
    n = len(rows)
    for i, (lab, bars, tag) in enumerate(rows):
        y = n - 1 - i
        m = len(bars)
        for j, (k, v, h) in enumerate(bars):
            yy = y - (j - (m - 1) / 2) * 0.3
            b.barh(yy, v, height=0.25, color=FMT[k], edgecolor=INK, linewidth=1.4, hatch="//" if h else None, zorder=3)
            b.text(v + 0.1, yy, f"{v:.2f}", fontsize=10.2, fontweight="bold" if not h else "normal", va="center")
        b.text(10.05, y, tag, fontsize=11, va="center", ha="left", fontweight="bold" if tag.startswith("in P3") else "normal")
    b.set_xlim(0, 12.6)
    b.set_ylim(-0.6, n - 0.35)
    b.set_yticks(range(n))
    b.set_yticklabels([r[0] for r in rows][::-1], fontsize=11.5)
    b.set_xticks([0, 2, 4, 6, 8])
    legend_fmt(ov, 8.2, 5.2, ["BF16", "MXFP8", "NVFP4"], size=11.5, sw=0.2, gap=0.25)
    source(fig, "round17 §4.1 (12 GEMMs, +203 µs quantisers), round20 §4.3 (ms per step), §4.5 (errors; block-0 wgrad 0.604 with NVFP4, 0.042 with cls / register tokens in bf16).",
           y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_03  which GEMM runs in which format
# ----------------------------------------------------------------------------------------------------


@fig("train_03_precision_map", 12, 7.8)
def train_03(fig):
    W, H = 12, 7.8
    ax = canvas(fig, W, H)
    title(fig, "Which GEMM runs in which number format",
          "per Linear layer: forward, dgrad, wgrad; the head's two ConvTranspose layers are run as GEMMs too")
    T, B_, M, N = ("FP32", "tf32"), ("BF16", "bf16"), ("MXFP8", "mx8"), ("NVFP4", "nv4+SR")
    cols = ["qkv", "proj", "fc1", "fc2", "head\nConvT"]
    rows = ["fwd", "dgrad", "wgrad"]
    stacks = [
        dict(t="R0 · original step", s=["fp32 / TF32 everywhere", "16.3 | 37.8 img/s"],
             g=[(T, T, T)] * 5),
        dict(t="FINAL · MXFP8", s=["68.9 | 161.9 img/s", "passes the strict gate"],
             g=[(B_, M, M), (M, M, M), (M, M, M), (M, M, M), (B_, B_, B_)]),
        dict(t="P3 · ViT-L default", s=["81.9 img/s, 1.20× over FINAL", "relaxed gate; block 0 stays MXFP8"],
             g=[(B_, N, N), (M, N, N), (M, N, N), (M, N, N), (M, N, N)]),
        dict(t="P3_H2 · ViT-B default", s=["177.4 img/s, 1.10× over FINAL", "relaxed gate; head dgrad / wgrad in MXFP8"],
             g=[(B_, N, N), (M, N, N), (M, N, N), (M, N, N), (B_, M, M)]),
        dict(t="P3 · ViT-B speed tier", s=["191.4 img/s, 1.20× over FINAL", "1 batch of 16 misses the relaxed minimum"],
             g=[(B_, N, N), (M, N, N), (M, N, N), (M, N, N), (M, N, N)]),
    ]
    xs = [0.45, 4.25, 8.05]
    tops = [6.45, 3.3]
    cw, ch, lw_ = 0.6, 0.4, 0.55
    for idx, st in enumerate(stacks):
        x0, top = xs[idx % 3], tops[idx // 3]
        ax.text(x0, top, st["t"], fontsize=14.5, fontweight="bold", va="center")
        ax.text(x0, top - 0.32, st["s"][0], fontsize=10.6, va="center", color=INK2)
        ax.text(x0, top - 0.56, st["s"][1], fontsize=10.6, va="center", color=INK2)
        for j, cname in enumerate(cols):
            ax.text(x0 + lw_ + j * cw + cw / 2 - 0.02, top - 0.98, cname, fontsize=10.2, ha="center", va="center", linespacing=1.0)
        for i, r in enumerate(rows):
            cy = top - 1.42 - i * (ch + 0.06)
            ax.text(x0 + lw_ - 0.08, cy, r, fontsize=10.8, ha="right", va="center", color=INK2)
            for j in range(5):
                fk, tag = st["g"][j][i]
                bx = x0 + lw_ + j * cw
                box(ax, bx, cy - ch / 2, cw - 0.06, ch, fc=FMT[fk], lw=1.4, r=0.05, z=3)
                ptext(ax, bx + (cw - 0.06) / 2, cy, tag, FMT[fk], size=9, fontweight="bold", ha="center", va="center", z=6)
    # notes cell (sixth)
    x0, top = xs[2], tops[1]
    ax.text(x0, top, "Reading guide", fontsize=14.5, fontweight="bold", va="center")
    lx = x0
    for k, lab in [("FP32", "TF32"), ("BF16", "bf16"), ("MXFP8", "MXFP8"), ("NVFP4", "NVFP4")]:
        box(ax, lx, top - 0.5, 0.2, 0.2, fc=FMT[k], lw=1.3, r=0.04, z=3)
        ax.text(lx + 0.27, top - 0.4, lab, fontsize=10.8, va="center")
        lx += 0.27 + 0.1 * len(lab) + 0.22
    notes = ["SR = stochastic rounding of the gradient dY",
             "weights are re-quantised every step from the fp32",
             "master copy (NVFP4: 2-D 16×16 block scales)",
             "",
             "bf16 / fp32 in every stack: attention (flash /",
             "cuDNN), LayerNorm, RoPE, loss + matcher, and the",
             "optimizer (flat fp32 master weights, AdamW)",
             "",
             "ConvT = the mask head's two ConvTranspose2d"]
    for k, s in enumerate(notes):
        ax.text(x0, top - 0.85 - k * 0.27, s, fontsize=10.6, va="center", color=INK)
    source(fig, "CONTRACTS 11.4 / 13.2 / 14.2 / 15.1 (policies), round17 / 21 / 22 / 23 NOTES;\nthroughput: round19 summary, round23 §4.2, CONTRACTS Amendment 6d §15.3.",
           y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_07  the FP4 quantiser
# ----------------------------------------------------------------------------------------------------


@fig("train_07_fp4_quantizer", 12, 9.9)
def train_07(fig):
    W, H = 12, 9.9
    ax = canvas(fig, W, H)
    title(fig, "How the FP4 path works",
          "block formats, one fused quantiser that writes both copies, and the delayed amax that keeps it free of extra passes")

    # ---- 1. block formats -----------------------------------------------------------------------
    text(ax, 0.45, 8.52, "1  Block formats", size=15.5, weight="bold")
    fmts = [("MXFP8", 7.9, 7, 0.19, "UE8M0", "32 × E4M3  +  1 × UE8M0 scale  =  8.25 bit / value"),
            ("NVFP4", 7.3, 10, 0.095, "E4M3", "16 × e2m1  +  1 × E4M3 scale  (+ fp32 per tensor)  =  4.5 bit"),
            ("MXFP4", 6.7, 10, 0.095, "UE8M0", "32 × e2m1  +  1 × UE8M0 scale  =  4.25 bit")]
    for k, y, n, w, sc, desc in fmts:
        box(ax, 0.45, y - 0.2, 1.0, 0.4, fc=FMT[k], lw=1.5, r=0.07, z=3)
        ptext(ax, 0.95, y, k, FMT[k], size=11.5, fontweight="bold", ha="center", va="center")
        x = 1.6
        for i in range(n):
            box(ax, x, y - 0.17, w - 0.02, 0.34, fc=tint(FMT[k], 0.55), lw=1.1, r=0.02, z=3)
            x += w
        ax.text(x + 0.03, y, "···", fontsize=12, va="center")
        x += 0.28
        box(ax, x, y - 0.17, 0.22, 0.34, fc=GRAY[2], lw=1.3, r=0.03, z=3)
        ax.text(x + 0.11, y, "s", fontsize=10.5, ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(3.75, y, desc, fontsize=10.6, va="center")
    # bits per value
    text(ax, 8.35, 8.2, "bits per value", size=11.5, color=INK2)
    for k, y, bits in [("BF16", 7.85, 16.0), ("MXFP8", 7.5, 8.25), ("NVFP4", 7.15, 4.5), ("MXFP4", 6.8, 4.25)]:
        ax.text(8.9, y, k, fontsize=10.6, ha="right", va="center")
        box(ax, 9.0, y - 0.12, bits * 0.15, 0.24, fc=FMT[k], lw=1.3, r=0.03, z=3)
        ax.text(9.0 + bits * 0.15 + 0.08, y, f"{bits:g}", fontsize=10.6, va="center", fontweight="bold")
    ax.text(0.45, 6.15, "NVFP4:  ts = amax / (6·448) per tensor;  s_blk = E4M3( amax_blk / (6·ts) );  x_q = e2m1( x / (ts · s_blk) );  weights use 2-D 16×16 block scales",
            fontsize=10, va="center", color=INK2)

    # ---- 2. one read, two copies ----------------------------------------------------------------
    text(ax, 0.45, 5.62, "2  One read, two copies", size=15.5, weight="bold")
    yc = 4.45
    box(ax, 0.45, yc - 0.5, 1.9, 1.0, fc=GRAY[0], lw=1.6, r=0.1, z=3)
    ax.text(1.4, yc + 0.1, "X or dY", fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(1.4, yc - 0.18, "bf16  [R tokens × C]", fontsize=10.2, ha="center", va="center", zorder=6)
    qf = tint(FMT["NVFP4"], 0.5)
    box(ax, 3.0, yc - 0.5, 2.2, 1.0, fc=qf, lw=1.7, r=0.1, z=3)
    ptext(ax, 4.1, yc + 0.12, "fused FP4 quantiser", qf, size=12, fontweight="bold", ha="center", va="center")
    ptext(ax, 4.1, yc - 0.17, "one read, two writes", qf, size=10.4, ha="center", va="center")
    arrow(ax, 2.38, yc, 2.97, yc, lw=2.0)
    box(ax, 6.0, yc + 0.28, 2.75, 0.78, fc=PAPER, lw=1.6, r=0.08, z=3)
    ax.text(6.12, yc + 0.78, "row copy   q [R, C/2]", fontsize=10.8, fontweight="bold", va="center", zorder=6)
    ax.text(6.12, yc + 0.52, "+ scales, 128×4 swizzle (K = C)", fontsize=9.8, va="center", zorder=6)
    box(ax, 6.0, yc - 1.06, 2.75, 0.78, fc=PAPER, lw=1.6, r=0.08, z=3)
    ax.text(6.12, yc - 0.56, "transposed copy   qt [C, R_pad/2]", fontsize=10.8, fontweight="bold", va="center", zorder=6)
    ax.text(6.12, yc - 0.82, "+ scales; R padded to 128 (K = tokens)", fontsize=9.8, va="center", zorder=6)
    arrow(ax, 5.23, yc + 0.18, 5.97, yc + 0.64, lw=1.9, rad=-0.1)
    arrow(ax, 5.23, yc - 0.18, 5.97, yc - 0.66, lw=1.9, rad=0.1)
    # GEMM chips
    gx = 9.25
    gem = [("fwd", "Y = X · Wᵀ", "X.q × W.q", yc + 0.95), ("dgrad", "dX = dY · W", "dY.q × Wᵀ.q", yc + 0.32),
           ("wgrad", "dW = dYᵀ · X", "dYᵀ.qt × Xᵀ.qt  (fp32 out)", yc - 0.78)]
    for nm, f, u, y in gem:
        box(ax, gx, y - 0.25, 2.3, 0.5, fc=tint(FMT["NVFP4"], 0.28), lw=1.5, r=0.07, z=3)
        ax.text(gx + 0.1, y + 0.08, f"{nm}   {f}", fontsize=10.4, fontweight="bold", va="center", zorder=6)
        ax.text(gx + 0.1, y - 0.13, u, fontsize=9.2, va="center", zorder=6)
    arrow(ax, 8.78, yc + 0.62, 9.22, yc + 0.95, lw=1.6, ms=11)
    arrow(ax, 8.78, yc + 0.6, 9.22, yc + 0.34, lw=1.6, ms=11)
    arrow(ax, 8.78, yc - 0.66, 9.22, yc - 0.76, lw=1.6, ms=11)
    # amax by-product
    box(ax, 3.0, yc - 1.2, 2.2, 0.46, fc=PAPER, lw=1.4, r=0.07, z=3, ls=(0, (4, 3)))
    ax.text(4.1, yc - 0.97, "amax by-product (atomicMax)", fontsize=10, ha="center", va="center", zorder=6)
    arrow(ax, 4.1, yc - 0.52, 4.1, yc - 0.72, lw=1.5, ms=11)
    textbox(ax, 0.45, yc - 1.12, "SR on dY: software Philox\nrounding (no hardware\ncvt.rs on sm_120)", size=9.6, pad=0.3)
    textbox(ax, 9.25, yc - 1.4, "sm_120: 4-bit operands must\nbe K-major on both sides,\nso all three GEMMs are TN", size=9.6, pad=0.25)

    # ---- 3. delayed amax --------------------------------------------------------------------------
    text(ax, 0.45, 2.55, "3  Delayed amax, no extra pass", size=15.5, weight="bold")
    steps = [("step t−1", "quantiser writes\namax(t−1)"), ("step t", "ts = headroom × amax(t−1)\nwrites amax(t)"), ("step t+1", "ts = headroom × amax(t)\n...")]
    sx = 0.45
    for ttl, body in steps:
        box(ax, sx, 1.42, 1.95, 0.85, fc=PAPER, lw=1.5, r=0.08, z=3)
        ax.text(sx + 0.1, 2.08, ttl, fontsize=10.8, fontweight="bold", va="center", zorder=6)
        ax.text(sx + 0.1, 1.67, body, fontsize=9.2, va="center", zorder=6, linespacing=1.2)
        sx += 2.15
    arrow(ax, 2.42, 1.85, 2.58, 1.85, lw=1.6, ms=11)
    arrow(ax, 4.57, 1.85, 4.73, 1.85, lw=1.6, ms=11)
    ax.text(0.45, 1.12, "first call: exact amax, run eagerly before the graph is captured", fontsize=9.8, va="center", color=INK2)
    ax.text(0.45, 0.82, "dY calls that saturate (ViT-L, 36,800 calls):", fontsize=10, va="center", color=INK2)
    for i, (hd, pct, extra) in enumerate([("headroom 8", "0.63%", ""), ("16", "0.09%", ""), ("32", "0.00%", "  recommended")]):
        bx = 0.45 + i * 2.15
        box(ax, bx, 0.28, 1.95, 0.38, fc=tint(FMT["NVFP4"], 0.45 if i == 2 else 0.18), lw=1.4, r=0.07, z=3)
        ax.text(bx + 0.1, 0.47, f"{hd}:  {pct}{extra}", fontsize=10, va="center", fontweight="bold", zorder=6)

    # ---- 4. massive activations -------------------------------------------------------------------
    text(ax, 6.95, 2.55, "4  Why ViT-L block 0 stays MXFP8", size=15.5, weight="bold")
    chips = [("cls / register tokens carry massive activations:", "block 0 fc2, X max 1992 against rms 9"),
             ("they pin the per-tensor scale; NVFP4 wgrad error 0.60", "(0.04 if those tokens stay in bf16)"),
             ("decision: block 0 keeps MXFP8 (\"nofirst\")", "costs about 1/24 of the dgrad / wgrad saving")]
    for i, (a_, b_) in enumerate(chips):
        y = 1.93 - i * 0.66
        box(ax, 6.95, y - 0.27, 4.6, 0.56, fc=PAPER, lw=1.5, r=0.08, z=3)
        ax.text(7.07, y + 0.1, a_, fontsize=10, va="center", zorder=6, fontweight="bold" if i == 2 else "normal")
        ax.text(7.07, y - 0.13, b_, fontsize=9.6, va="center", zorder=6, color=INK2)
    source(fig, "CONTRACTS 11.2 / 12.2 / 12.3 / 13.3 / 15.2 (formats, delayed amax, headroom study); round20 §4.4 / §4.5 (quantiser, massive activations); round21 NOTES.", y=0.008)


# ----------------------------------------------------------------------------------------------------
# train_08  whole-network structure of one training step
# ----------------------------------------------------------------------------------------------------


def _band2(ax, x, y, w, h=0.09):
    """Two-colour band = forward GEMMs in MXFP8 (green), backward GEMMs in NVFP4 (yellow)."""
    for i, k in enumerate(["MXFP8", "NVFP4"]):
        ax.add_patch(Rectangle((x + i * w / 2, y), w / 2, h, fc=FMT[k], ec=INK, lw=1.0, zorder=4))


def _tbox(ax, x, y, w, h, ttl, sub="", fill=PAPER, ts=10.8, ss=9.0, band=False):
    box(ax, x, y, w, h, fc=fill, lw=1.6, r=0.07, z=3)
    cy = y + h / 2 + (0.04 if band else 0)
    ptext(ax, x + w / 2, cy + (0.1 if sub else 0), ttl, fill, size=ts, fontweight="bold", ha="center", va="center", z=6)
    if sub:
        ptext(ax, x + w / 2, cy - 0.12, sub, fill, size=ss, ha="center", va="center", z=6)
    if band:
        _band2(ax, x + 0.04, y + 0.03, w - 0.08)


def _chips(ax, x, ytop, w, roles, ch=0.2, gap=0.02, size=8.4):
    """Stack of format chips under a Linear: roles = [(label, format key), ...]."""
    for i, (lab, k) in enumerate(roles):
        y = ytop - (i + 1) * ch - i * gap
        box(ax, x, y, w, ch, fc=FMT[k], lw=1.2, r=0.04, z=3)
        ptext(ax, x + w / 2, y + ch / 2, lab, FMT[k], size=size, fontweight="bold", ha="center", va="center", z=6)


@fig("train_08_model_structure", 12, 10.2)
def train_08(fig):
    import textwrap

    W, H = 12, 10.2
    ax = canvas(fig, W, H)
    title(fig, "Training: where MXFP8 / NVFP4 sit in one step",
          "P3 stack, ViT-L/16, batch 7: forward, backward and update run as one CUDA graph; MXFP4 was only tried as an arm")
    GR = GRAY

    # ---- row A: the step -------------------------------------------------------------------------
    flow = [("Batch", "7 images 512x512", "+ COCO masks", GR[0], False),
            ("Patch embed", "cuDNN conv, bf16", "+ cls, 4 registers", tint(FMT["BF16"], 0.3), False),
            ("Blocks 1-21", "plain ViT blocks", "(ViT-B: 1-9)", PAPER, True),
            ("+ queries", "100 learned tokens", "join at block 22", GR[0], False),
            ("Joint blocks x3", "masked attention", "mask from predict", PAPER, True),
            ("EoMT head", "norm + head", "runs 4x per step", PAPER, True),
            ("Loss", "Hungarian matcher", "BCE + dice + class", GR[1], False)]
    bw, gap, ya, ha_ = 1.35, 0.275, 8.08, 0.88
    for i, (a, b, c, fill, band) in enumerate(flow):
        x = 0.45 + i * (bw + gap)
        _tbox(ax, x, ya, bw, ha_, a, b, fill=fill, ts=10.8, ss=8.8, band=band)
        ptext(ax, x + bw / 2, ya + 0.17 + (0.06 if band else 0), c, fill, size=8.8, ha="center", va="center", z=6)
        if i < len(flow) - 1:
            arrow(ax, x + bw + 0.02, ya + ha_ / 2, x + bw + gap - 0.02, ya + ha_ / 2, lw=1.8, ms=12)
    # backward path under the flow
    xl, xr = 0.45 + 1 * (bw + gap) + bw / 2, 0.45 + 6 * (bw + gap) + bw / 2
    yb = 7.55
    line(ax, [xr, xr], [ya - 0.02, yb], color=INK2, lw=1.8, z=2)
    line(ax, [xr, xl], [yb, yb], color=INK2, lw=1.8, z=2)
    arrow(ax, xl, yb, xl, ya - 0.02, lw=1.8, ms=12, color=INK2)
    ax.text((xl + xr) / 2, yb + 0.2, "backward: dgrad + wgrad GEMMs in NVFP4 (SR on dY), attention backward in bf16", fontsize=10.4, ha="center", va="center", color=INK)
    ax.text((xl + xr) / 2, yb - 0.22, "then dW (fp32) → flat grad buffer → clip + AdamW on fp32 master weights (1-2 kernels)", fontsize=10, ha="center", va="center", color=INK2)
    box(ax, 0.45, yb - 0.3, 1.2, 0.62, fc=GR[0], lw=1.4, r=0.07, z=3)
    ax.text(1.05, yb + 0.1, "update", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(1.05, yb - 0.12, "fp32 master", fontsize=8.8, ha="center", va="center", zorder=6)
    arrow(ax, xl - 0.06, yb - 0.22, 1.67, yb - 0.1, lw=1.4, ms=10, color=INK2)

    # ---- row B left: one ViT block with the GEMM formats --------------------------------------------
    text(ax, 0.45, 6.95, "ViT block: formats of the three GEMMs of every Linear", size=14, weight="bold")
    bwid, bh = 1.0, 0.5
    fw = [("fwd bf16", "BF16"), ("dgrad nv4+SR", "NVFP4"), ("wgrad nv4+SR", "NVFP4")]
    fm = [("fwd mx8", "MXFP8"), ("dgrad nv4+SR", "NVFP4"), ("wgrad nv4+SR", "NVFP4")]
    y1, y2 = 6.25, 4.85
    row1 = [("LN", "bf16", GR[1], None), ("qkv", "d → 3d", PAPER, fw), ("RoPE", "", GR[1], None),
            ("attention", "flash, bf16", FMT["BF16"], None), ("proj", "d → d", PAPER, fm)]
    row2 = [("LN", "bf16", GR[1], None), ("fc1", "d → 4d, GELU", PAPER, fm), ("fc2", "4d → d", PAPER, fm)]
    for rowi, (row, y) in enumerate([(row1, y1), (row2, y2)]):
        for i, (a, b, f, ch) in enumerate(row):
            x = 0.5 + i * (bwid + 0.2)
            _tbox(ax, x, y, bwid, bh, a, b, fill=f, ts=10.2, ss=8.4)
            if ch:
                _chips(ax, x + 0.01, y - 0.05, bwid - 0.02, ch)
            if i < len(row) - 1:
                arrow(ax, x + bwid + 0.01, y + bh / 2, x + bwid + 0.19, y + bh / 2, lw=1.4, ms=9)
    xe, xs0, ym = 0.5 + 4 * (bwid + 0.2) + bwid / 2, 0.5 + bwid / 2, 5.47
    line(ax, [xe, xe], [y1 - 0.7, ym], color=INK2, lw=1.4, z=2)
    line(ax, [xe, xs0], [ym, ym], color=INK2, lw=1.4, z=2)
    arrow(ax, xs0, ym, xs0, y2 + bh + 0.02, lw=1.4, ms=9, color=INK2)
    ax.text(3.9, ym + 0.1, "residual stream", fontsize=8.8, color=INK2, ha="center", va="bottom")
    note(ax, 0.5, 3.62, "dgrad dX = dY·W and wgrad dW = dYᵀ·X use the same Linears;\nViT-L block 0 keeps MXFP8 for dgrad / wgrad", size=9.6, w=5.6, h=0.5)

    # ---- row B right: the head --------------------------------------------------------------------
    text(ax, 6.75, 6.95, "EoMT head (runs 4x per step)", size=14, weight="bold")
    ax.text(6.75, 3.62, "3 taps before the joint blocks + 1 at the end, each with its own loss", fontsize=9.4, color=INK2, va="center")
    hw = 0.86
    xs = [6.75 + i * (hw + 0.15) for i in range(5)]
    chead = [("fwd mx8", "MXFP8"), ("dgrad nv4+SR", "NVFP4"), ("wgrad nv4+SR", "NVFP4")]
    lane = [("patch", "32x32 grid", GR[0], None), ("ConvT1", "k2 s2", PAPER, chead), ("dw conv", "GELU, LN2D", GR[1], None),
            ("ConvT2", "k2 s2", PAPER, chead), ("dw conv", "GELU, LN2D", GR[1], None)]
    for i, (a, b, f, ch) in enumerate(lane):
        _tbox(ax, xs[i], y1, hw, bh, a, b, fill=f, ts=9.6, ss=8.2)
        if ch:
            _chips(ax, xs[i] + 0.01, y1 - 0.05, hw - 0.02, ch, size=8.0)
        if i < 4:
            arrow(ax, xs[i] + hw + 0.01, y1 + bh / 2, xs[i + 1] - 0.01, y1 + bh / 2, lw=1.4, ms=9)
    BFc = FMT["BF16"]
    _tbox(ax, xs[0], y2, hw, bh, "queries", "100 x C", fill=GR[0], ts=9.6, ss=8.2)
    _tbox(ax, xs[1], y2, hw + 0.2, bh, "mask head", "3 Linear, bf16", fill=BFc, ts=9.6, ss=8.2)
    _tbox(ax, xs[3], y2, hw + 0.45, bh, "einsum", "bf16 GEMM", fill=BFc, ts=9.6, ss=8.2)
    arrow(ax, xs[0] + hw + 0.01, y2 + bh / 2, xs[1] - 0.01, y2 + bh / 2, lw=1.4, ms=9)
    arrow(ax, xs[1] + hw + 0.21, y2 + bh / 2, xs[3] - 0.01, y2 + bh / 2, lw=1.4, ms=9)
    arrow(ax, xs[4] + hw / 2, y1 - 0.02, xs[3] + hw + 0.3, y2 + bh + 0.02, lw=1.4, ms=9)
    _tbox(ax, xs[0], 4.05, hw, bh, "queries", "", fill=GR[0], ts=9.6)
    _tbox(ax, xs[1], 4.05, hw + 0.2, bh, "class head", "Linear, bf16", fill=BFc, ts=9.6, ss=8.2)
    arrow(ax, xs[0] + hw + 0.01, 4.05 + bh / 2, xs[1] - 0.01, 4.05 + bh / 2, lw=1.4, ms=9)
    ax.text(xs[3] + 0.1, 4.3, "→ (mask, class) logits → loss", fontsize=9.4, va="center")

    # ---- legend ---------------------------------------------------------------------------------
    ly = 3.32
    lx = 0.5
    for k, lab in [("BF16", "bf16 (fwd of qkv, attention, small heads)"), ("MXFP8", "MXFP8"), ("NVFP4", "NVFP4, SR on dY"), ("MXFP4", "MXFP4: arm only")]:
        box(ax, lx, ly - 0.11, 0.22, 0.22, fc=FMT[k], lw=1.3, r=0.04, z=3)
        ax.text(lx + 0.3, ly, lab, fontsize=10, va="center")
        lx += 0.3 + 0.075 * len(lab) + 0.3
    box(ax, 10.6, ly - 0.11, 0.2, 0.22, fc=PAPER, lw=1.3, r=0.04, z=3)
    _band2(ax, 10.62, ly - 0.09, 0.16, 0.07)
    ax.text(10.9, ly, "fwd | bwd", fontsize=9.6, va="center")

    # ---- row C: three cards ----------------------------------------------------------------------
    cards = [("Weights, every step",
              ["fp32 master weights sit in one flat buffer.",
               "Each step quantises W and Wᵀ (NVFP4: 2-D 16x16 block scale) for dgrad / wgrad.",
               "qkv forward stays bf16; proj, fc1, fc2 forward use MXFP8 weights.",
               "dW comes out fp32, then clip + AdamW run in 1-2 kernels."]),
             ("FP4 bookkeeping",
              ["X and dY are quantised once into two copies: row (K = channels) and transposed (K = tokens).",
               "Tensor scale = headroom x amax of the previous step; the quantiser writes the new amax.",
               "SR noise comes from a Philox counter that advances inside the graph.",
               "The FP4 state rolls once per step, so graph replay equals eager."]),
             ("Tiers and exceptions",
              ["FINAL (MXFP8 tier): all dgrad / wgrad in MXFP8, head in bf16.",
               "P3 (ViT-L): NVFP4 dgrad / wgrad, head forward MXFP8; block 0 keeps MXFP8.",
               "P3_H2 (ViT-B default): head forward bf16, head dgrad / wgrad MXFP8.",
               "MXFP4: only an R20b arm in the slots NVFP4 uses; no whole-step run."])]
    cw = 3.6
    for j, (ttl, items) in enumerate(cards):
        x = 0.45 + j * (cw + 0.15)
        box(ax, x, 0.45, cw, 2.6, fc=PAPER, lw=1.6, r=0.1, z=2)
        ax.text(x + 0.15, 2.8, ttl, fontsize=12.5, fontweight="bold", va="center")
        y = 2.5
        for it in items:
            wrapped = textwrap.wrap(it, 47)
            ax.text(x + 0.15, y, "\n".join(wrapped), fontsize=9.4, va="top", linespacing=1.2)
            y -= 0.175 * len(wrapped) + 0.13
    source(fig, "structure: src/.../dinov3_eomt_semantic_segmentation (task_model.py, train_model.py); formats: CONTRACTS 11.4 / 13.2 / 15.1, round17-23 NOTES.", y=0.008)


# --- next figure goes here ---
