"""infer-opt figures (7). Every number is copied from infer-opt NOTES / results; sources are printed on each figure."""
from __future__ import annotations

import numpy as np  # noqa: F401

from .style import (  # noqa: F401
    ARROW, FMT, GRAY, INK, INK2, LABEL, MUTED, PAPER, RULE, SOFT, arrow, bar_label, barh_fmt, box, callout, canvas,
    fig, fmt_bar, legend_fmt, line, note, on, panel_title, render, source, star, star_marker, style_axes, text, textbox,
    tint, title, ptext,
)

# ----------------------------------------------------------------------------------------------------
# infer_01  one frame's time budget
# ----------------------------------------------------------------------------------------------------


@fig("infer_01_frame_budget", 12, 7.8)
def infer_01(fig):
    title(
        fig,
        "Where does one frame's time go?",
        "ViT-L/16 on an RTX 5090  |  one 1900x1440 video frame, cut into 2 crops of 512x512",
    )

    # ---- (a) wall-clock per frame, original script -> GPU pre/post -> one CUDA graph ---------------
    ax = fig.add_axes([0.235, 0.535, 0.715, 0.265])
    style_axes(ax, grid="x")
    panel_title(ax, "(a) From the original script to ONE graph launch", "wall-clock per frame, benchmark loop, ms")
    seg_c = {"prep": "#e7e2d3", "model": "#cfc8b2", "post": "#9b937e", "other": "#f3efe2"}
    # original script (round0 bf16-autocast): prep 5.77 / model 17.05 / post 35.36 / e2e 60.65 -> other 2.47
    left = 0.0
    for name, v in [("prep", 5.77), ("model", 17.05), ("post", 35.36), ("other", 2.47)]:
        ax.barh(2, v, left=left, height=0.56, color=seg_c[name], edgecolor=INK, linewidth=1.6,
                hatch="//" if name == "post" else None, zorder=3)
        left += v
    # R7: prep 0.70 / model 16.46 / post 0.22
    left = 0.0
    for name, v in [("prep", 0.70), ("model", 16.46), ("post", 0.22)]:
        ax.barh(1, v, left=left, height=0.56, color=seg_c[name], edgecolor=INK, linewidth=1.6, zorder=3)
        left += v
    barh_fmt(ax, 0, 10.24, "BF16", height=0.56)
    ax.set_xlim(0, 68)
    ax.set_ylim(-0.55, 2.8)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Original script\n(PyTorch eager)", "R7: GPU pre/post\n+ fused head",
                        "R9: whole frame in\none CUDA graph"], fontsize=12.5)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    for x, s in [(2.9, "prep 5.8"), (14.3, "model 17.1"), (40.5, "post 35.4: CPU numpy argmax, GPU 7% busy"),
                 (61.2, "other\n2.5")]:
        ax.text(x, 2.34, s, fontsize=12, ha="center", va="bottom", color=INK, linespacing=1.0)
    bar_label(ax, 60.65 + 3.2, 2.0, "60.7 ms", ha="left", va="center", size=14)
    bar_label(ax, 17.4 + 0.8, 1.0, "17.4 ms", ha="left", va="center", size=14)
    bar_label(ax, 10.24 + 0.8, 0.0, "10.2 ms", ha="left", va="center", size=14)
    textbox(ax, 28.5, 1.0, f"prep 5.8 + post 35.4  {ARROW}  0.7 + 0.2 ms\nonly a uint8 mask (2.7 MB) comes back", size=11.5)
    textbox(ax, 19.5, 0.0, "1 cudaGraphLaunch per frame, GPU busy 99%", size=11.5)

    # ---- (b) GPU frame time inside the CUDA graph, by number format --------------------------------
    ax2 = fig.add_axes([0.235, 0.105, 0.715, 0.285])
    style_axes(ax2, grid="x")
    panel_title(ax2, "(b) Same ruler, inside the graph: only the GEMM number format changes",
                "GPU frame time = prep + model + post (no video decode), CUDA-graph replay, ms")
    tiers = [("BF16", 9.93, "BF16 (R9 r5_v1)"), ("FP16", 7.76, "FP16 accumulate"),
             ("MXFP8", 4.98, "MXFP8 †"), ("NVFP4", 4.10, "NVFP4"), ("MXFP4", 4.12, "MXFP4")]
    mult = {"BF16": "1.00x", "FP16": "1.28x", "MXFP8": "1.98x", "NVFP4": "2.42x", "MXFP4": "2.41x"}
    for i, (k, v, _) in enumerate(tiers):
        y = len(tiers) - 1 - i
        barh_fmt(ax2, y, v, k, height=0.56)
        bar_label(ax2, v + 0.7, y, f"{v:.2f} ms   {mult[k]}", ha="left", va="center", size=13.5)
    ax2.set_xlim(0, 68)
    ax2.set_ylim(-0.6, 4.6)
    ax2.set_yticks(range(len(tiers)))
    ax2.set_yticklabels([t[2] for t in tiers][::-1], fontsize=12.5)
    ax2.set_xticks([0, 10, 20, 30, 40, 50, 60])
    textbox(ax2, 30, 2.4, f"model section alone: 9.77 {ARROW} 4.02 ms\n= 2.43x with NVFP4 (ViT-L)", size=13)
    star_marker(ax2, 27.0, 2.4, size=30)
    source(
        fig,
        "(a) round0 / round7 / round9 NOTES, e2e wall-clock, sync every iteration.  (b) round14 results_speed.json "
        "(speed12.py, 3 rounds, median).\n† MXFP8: round13 session (r5_v1 was 9.89 ms there). "
        "Two different harnesses on one ruler: read (b) as ratios.",
        y=0.008,
    )


# ----------------------------------------------------------------------------------------------------
# infer_02  FPS ladder
# ----------------------------------------------------------------------------------------------------


def _bracket(ax, x0, x1, y, label, size=12, drop=14):
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], color=INK, lw=1.6, zorder=6)
    ax.text((x0 + x1) / 2, y + 3, label, fontsize=size, ha="center", va="bottom", color=INK)


@fig("infer_02_fps_ladder", 12, 9.4)
def infer_02(fig):
    title(fig, "From 15 FPS to 200+ FPS", "whole-video throughput (126-frame loop, cv2 decode included) and the GPU-only upper bound")

    # ---- (a) ViT-L video FPS ----------------------------------------------------------------------
    ax = fig.add_axes([0.075, 0.565, 0.905, 0.265])
    style_axes(ax, grid="y")
    panel_title(ax, "(a) ViT-L/16: whole-video FPS, step by step", "systems work first (BF16 numerics unchanged), then the GEMM number format")
    stages = [
        ("Original\nscript", 15.1, "BF16", "1.0×"),
        ("R8 decode+\n3 streams", 57.3, "BF16", "3.7×"),
        ("R9 r5_v1\nfused + graph", 94.7, "BF16", "6.3×"),
        ("R12\nFP16-acc", 120.4, "FP16", "1.27×"),
        ("R13\nMXFP8 †", 183.4, "MXFP8", "1.88×"),
        ("R14\nNVFP4", 207.5, "NVFP4", "2.19×"),
        ("R14\nMXFP4", 207.6, "MXFP4", "2.19×"),
    ]
    for i, (lab, v, k, r) in enumerate(stages):
        fmt_bar(ax, i, v, k, width=0.62)
        ax.text(i, v + 5, f"{v:.1f}", fontsize=14, fontweight="bold", ha="center", va="bottom")
        ax.text(i, v + 32, r, fontsize=12.5, ha="center", va="bottom", color=INK2)
    ax.set_xlim(-0.55, 6.55)
    ax.set_ylim(0, 335)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=12)
    _bracket(ax, -0.4, 2.4, 292, "× vs the original script")
    _bracket(ax, 2.6, 6.4, 292, "× vs r5_v1, same session")
    ax.set_ylabel("FPS", fontsize=13)

    # ---- (b) ViT-B GPU upper bound vs video FPS --------------------------------------------------
    ax2 = fig.add_axes([0.075, 0.125, 0.905, 0.255])
    style_axes(ax2, grid="y")
    panel_title(ax2, "(b) ViT-B/16: the GPU outruns the video decoder",
                "solid = GPU upper bound (CUDA graph, no decode)   light = whole video, cv2 decode included")
    tiers = [("BF16\nr5_v1", "BF16", 264.5, 215.4, "1.00×"), ("FP16-acc\nr12", "FP16", 352.6, 241.7, "1.33×"),
             ("MXFP8 †\nr13", "MXFP8", 497.6, 272.4, "1.91×"), ("NVFP4\nr14", "NVFP4", 553.5, 243.9, "2.09×"),
             ("MXFP4\nr14", "MXFP4", 548.4, 239.8, "2.07×")]
    for i, (lab, k, gpu, vid, r) in enumerate(tiers):
        ax2.bar(i - 0.2, gpu, width=0.36, color=FMT[k], edgecolor=INK, linewidth=1.7, zorder=3)
        ax2.bar(i + 0.2, vid, width=0.36, color=tint(FMT[k], 0.42), edgecolor=INK, linewidth=1.7, zorder=3)
        ptext(ax2, i - 0.2, gpu - 14, f"{gpu:.0f}", FMT[k], size=13, fontweight="bold", ha="center", va="top", z=8)
        ptext(ax2, i + 0.2, vid - 14, f"{vid:.0f}", tint(FMT[k], 0.42), size=12.5, ha="center", va="top", z=8, color=INK)
        ax2.text(i - 0.2, max(gpu + 10, 300), r, fontsize=12, ha="center", va="bottom", color=INK2)
    ax2.axhline(286, xmin=0.0, xmax=0.77, color=INK, lw=1.6, ls=(0, (5, 4)), zorder=2)
    ax2.set_xlim(-0.6, 5.9)
    ax2.set_ylim(0, 700)
    ax2.set_yticks([0, 200, 400, 600])
    ax2.set_xticks(range(len(tiers)))
    ax2.set_xticklabels([t[0] for t in tiers], fontsize=12)
    ax2.set_ylabel("FPS", fontsize=13)
    textbox(ax2, 4.62, 520, "cv2 decode\n≈ 3.5 ms / frame\n→ ~286 FPS cap", size=11.5, ha="left")
    textbox(ax2, 4.62, 215, "ViT-B end to end:\n16.5 → 214.1 FPS\n(13.0× at R9)", size=11.5, ha="left")
    source(
        fig,
        "round8 / round9 NOTES, results/summary.md; speed12.py in round13 / round14 (CUDA-graph replay, median of 3).  "
        "† round13 session (r5_v1 97.3 / 260.2 FPS).\n"
        "R12-R14 ratios in (a) and all ratios in (b) are same-session. Cap line = single-thread decode time from round8.",
        y=0.008,
    )


# ----------------------------------------------------------------------------------------------------
# infer_03  one transformer block under four number formats
# ----------------------------------------------------------------------------------------------------

_ROWS = ["ln1", "q1", "qkv", "rope", "attn", "qa", "proj", "ln2", "q2", "fc1", "qg", "fc2"]


def _cell(ax, x, cy, w, kind, ttl, sub, fill, h=0.47):
    y = cy - h / 2
    if kind == "fused":
        box(ax, x, y + 0.06, w, h - 0.12, fc=PAPER, ec=MUTED, lw=1.3, r=0.06, ls=(0, (4, 3)), z=2)
        ax.text(x + w / 2, cy, ttl, fontsize=10.5, ha="center", va="center", color=MUTED, style="italic", zorder=5)
        return
    box(ax, x, y, w, h, fc=fill, ec=INK, lw=1.6, r=0.07, z=3)
    ptext(ax, x + 0.1, cy + 0.095, ttl, fill, size=11.5, fontweight="bold", ha="left", va="center")
    ptext(ax, x + 0.1, cy - 0.105, sub, fill, size=9.6, ha="left", va="center")


@fig("infer_03_block_dataflow", 12, 10.6)
def infer_03(fig):
    W, H = 12, 10.6
    ax = canvas(fig, W, H)
    title(fig, "One transformer block, four number formats",
          "ViT-L/16 layer, same weights. Every box is one CUDA kernel; each row is the same logical step in all four stacks.")
    col_w, bw = 2.65, 2.33
    x0s = [0.45, 3.30, 6.15, 9.00]
    ctr0, pitch = 7.95, 0.56
    ys = {k: ctr0 - i * pitch for i, k in enumerate(_ROWS)}
    top, bot = ys["ln1"] + 0.3, ys["fc2"] - 0.3

    cols = [
        dict(key="BF16", head="BF16 · r5_v1", n="7 kernels / layer", fm="bf16 operands, fp32 accumulate", attn="BF16", cells={
            "ln1": ("ln", "add_ln", "fp32 x → LN → bf16"),
            "qkv": ("gemm", "qkv GEMM", "bf16 + bias·RoPE·layout"),
            "rope": ("fused", "RoPE fused in qkv epilogue", ""),
            "attn": ("attn", "flash attention", "bf16, fp32 softmax stats"),
            "proj": ("gemm", "proj GEMM", "bf16 + bias·γ·fp32 residual"),
            "ln2": ("ln", "add_ln", "fp32 x → LN → bf16"),
            "fc1": ("gemm", "fc1 GEMM", "bf16 + bias·GELU"),
            "fc2": ("gemm", "fc2 GEMM", "bf16 + bias·γ·fp32 residual"),
        }, gemm="6.81 ms"),
        dict(key="FP16", head="FP16-acc · r12", n="7 kernels / layer", fm="fp16, f16 accumulate, fp32 per k-tile", attn="FP16", cells={
            "ln1": ("ln", "ln16", "fp32 x → LN → fp16"),
            "qkv": ("gemm", "qkv GEMM", "fp16 + bias·RoPE·layout"),
            "rope": ("fused", "RoPE fused in qkv epilogue", ""),
            "attn": ("attn", "attention", "fp16, S and PV f16-acc"),
            "proj": ("gemm", "proj GEMM", "fp16 + bias·γ·fp32 residual"),
            "ln2": ("ln", "ln16", "fp32 x → LN → fp16"),
            "fc1": ("gemm", "fc1 GEMM", "fp16 + bias·GELU"),
            "fc2": ("gemm", "fc2 GEMM", "fp16 + bias·γ·fp32 residual"),
        }, gemm="not profiled"),
        dict(key="MXFP8", head="MXFP8 · r13", n="9 kernels / layer", fm="e4m3 + UE8M0 per 1×32", attn="FP16", cells={
            "ln1": ("ln", "ln_mx8", "LN + quantize to e4m3"),
            "q1": ("fused", "quantize fused into ln_mx8", ""),
            "qkv": ("gemm", "qkv GEMM (cuBLASLt)", "block-scaled → bf16"),
            "rope": ("pass", "epi_rope", "bias·RoPE·layout → fp16"),
            "attn": ("attn", "attention", "fp16, S and PV f16-acc"),
            "qa": ("quant", "quant_rows", "fp16 → e4m3 + UE8M0"),
            "proj": ("gemm", "proj GEMM (own mgx)", "MXFP8 + γ·fp32 residual"),
            "ln2": ("ln", "ln_mx8", "LN + quantize to e4m3"),
            "q2": ("fused", "quantize fused into ln_mx8", ""),
            "fc1": ("gemm", "fc1 GEMM (mgx)", "MXFP8 + GELU + quantize-out"),
            "qg": ("fused", "quantize fused into fc1 epilogue", ""),
            "fc2": ("gemm", "fc2 GEMM (mgx)", "MXFP8 + γ·fp32 residual"),
        }, gemm="2.81 ms"),
        dict(key="NVFP4", head="NVFP4 · r14", n="12 kernels / layer (MXFP4 alike)", fm="e2m1 + E4M3 per 16 + fp32 tensor scale", attn="FP16", cells={
            "ln1": ("ln", "ln_bf16", "fp32 x → LN → bf16"),
            "q1": ("quant", "quant (fp4q2)", "bf16 → e2m1 + E4M3"),
            "qkv": ("gemm", "qkv GEMM (QuACK)", "NVFP4 + bias → bf16"),
            "rope": ("pass", "rope14", "RoPE + layout → fp16"),
            "attn": ("attn", "attention", "fp16, S and PV f16-acc"),
            "qa": ("quant", "quant", "bf16 → e2m1 + E4M3"),
            "proj": ("gemm", "proj GEMM (QuACK)", "NVFP4 + γ·fp32 residual"),
            "ln2": ("ln", "ln_bf16", "fp32 x → LN → bf16"),
            "q2": ("quant", "quant", "bf16 → e2m1 + E4M3"),
            "fc1": ("gemm", "fc1 GEMM (QuACK)", "NVFP4 + bias·GELU → bf16"),
            "qg": ("quant", "quant", "GELU out → e2m1 + E4M3"),
            "fc2": ("gemm", "fc2 GEMM (QuACK)", "NVFP4 + γ·fp32 residual"),
        }, gemm="1.76 ms"),
    ]
    for x0, col in zip(x0s, cols):
        k = col["key"]
        hx = x0 + 0.02
        box(ax, hx, 8.38, col_w - 0.04, 0.98, fc=FMT[k], lw=1.8, r=0.1, z=3)
        ptext(ax, hx + 0.14, 9.08, col["head"], FMT[k], size=14.5, fontweight="bold", ha="left", va="center")
        ptext(ax, hx + 0.14, 8.80, col["n"], FMT[k], size=11, ha="left", va="center")
        ptext(ax, hx + 0.14, 8.55, col["fm"], FMT[k], size=9.4, ha="left", va="center")
        bx = x0 + 0.32
        cx = bx + bw / 2
        # data-flow spine and fp32 residual rail
        line(ax, [cx, cx], [top, bot], color=INK2, lw=1.4, z=1)
        rail = x0 + 0.12
        line(ax, [rail, rail], [top + 0.05, bot - 0.05], color=GRAY[3], lw=3.2, z=1)
        for row in _ROWS:
            cell = col["cells"].get(row)
            if cell is None:
                continue
            kind, ttl, sub = cell
            fill = {"gemm": FMT[k], "attn": FMT[col["attn"]], "ln": GRAY[1], "pass": GRAY[0],
                    "quant": tint(FMT[k], 0.42), "fused": PAPER}[kind]
            _cell(ax, bx, ys[row], bw, kind, ttl, sub, fill)
            if row in ("ln1", "ln2"):
                arrow(ax, rail, ys[row], bx, ys[row], color=GRAY[4], lw=1.8, ms=11)
            if row in ("proj", "fc2"):
                line(ax, [bx, rail], [ys[row], ys[row]], color=GRAY[4], lw=1.8, z=2)
                ax.plot([rail], [ys[row]], "o", ms=10, mfc=PAPER, mec=INK, mew=1.4, zorder=6)
                ax.text(rail, ys[row] - 0.005, "+", fontsize=10, ha="center", va="center", zorder=7)
        # footer: backbone GEMM time per frame (nsys)
        box(ax, x0 + 0.02, 1.02, col_w - 0.04, 0.5, fc=tint(FMT[k], 0.3), lw=1.5, r=0.08, z=3)
        ptext(ax, x0 + col_w / 2, 1.27, f"backbone GEMMs  {col['gemm']}", tint(FMT[k], 0.3), size=11.5, fontweight="bold",
              ha="center", va="center", color=INK)
    ax.text(0.47, 0.86, "nsys, ViT-L, per frame (4 backbone GEMM kinds of 24 layers); r12 was not profiled", fontsize=10.5, color=INK2, va="center")

    # legend
    lx, ly = 0.47, 0.58
    for lab, fill in [("GEMM", FMT["BF16"]), ("attention", FMT["FP16"]), ("LN / RoPE pass", GRAY[1]),
                      ("quantize pass", tint(FMT["NVFP4"], 0.42))]:
        box(ax, lx, ly - 0.1, 0.22, 0.2, fc=fill, lw=1.3, r=0.04, z=3)
        ax.text(lx + 0.3, ly, lab, fontsize=10.8, va="center")
        lx += 0.3 + 0.085 * len(lab) + 0.3
    box(ax, lx, ly - 0.1, 0.22, 0.2, fc=PAPER, ec=MUTED, lw=1.2, r=0.04, ls=(0, (3, 2)), z=3)
    ax.text(lx + 0.3, ly, "fused away", fontsize=10.8, va="center")
    lx += 0.3 + 0.085 * 10 + 0.3
    line(ax, [lx, lx + 0.35], [ly, ly], color=GRAY[3], lw=3.2)
    ax.text(lx + 0.45, ly, "fp32 residual x", fontsize=10.8, va="center")
    ax.text(0.47, 0.34, "colour = number format of that kernel; attention shows its own accumulate format (BF16 flash, then FP16-acc from r12 on)",
            fontsize=10.2, color=INK2, va="center")
    source(fig, "round3 / round5, round12 §1, round13 §0-1, round14 §1 NOTES; nsys sums: round10 §2, round13 §3, round14 §7.2.", y=0.006)


# ----------------------------------------------------------------------------------------------------
# infer_04  pipeline + whole-frame CUDA graph
# ----------------------------------------------------------------------------------------------------


@fig("infer_04_pipeline_graph", 12, 8.7)
def infer_04(fig):
    from matplotlib.patches import Rectangle

    W, H = 12, 8.7
    ax = canvas(fig, W, H)
    title(fig, "One frame, one graph launch",
          "decode, copy, compute and read-back overlap in three CUDA streams; the whole frame runs as one CUDA Graph")
    text(ax, 0.45, H - 1.22, "Pipeline of R8 + R9 (not to scale)", size=15.5, weight="bold")
    A, B = GRAY[1], GRAY[0]
    Y = [H - 1.7 - i * 0.52 for i in range(6)]
    lanes = [("decode thread", "cv2.read() into a pinned ring"), ("H2D stream", "copy stream"),
             ("compute stream", "ONE CUDA Graph / frame"), ("D2H stream", "copy stream"),
             ("result thread", "host"), ("overlay thread", "optional, CPU only")]
    for (a, b), y in zip(lanes, Y):
        line(ax, [2.65, 11.75], [y - 0.26, y - 0.26], color=RULE, lw=1.0, z=1)
        ax.text(0.45, y + 0.075, a, fontsize=12, fontweight="bold", va="center")
        ax.text(0.45, y - 0.12, b, fontsize=9.6, va="center", color=INK2)

    def blk(x0, x1, lane, fill, label="", ls="-", size=10.5, lw=1.5):
        y = Y[lane]
        box(ax, x0, y - 0.19, x1 - x0, 0.38, fc=fill, lw=lw, r=0.05, ls=ls, z=3)
        if label:
            ax.text((x0 + x1) / 2, y, label, fontsize=size, ha="center", va="center", zorder=5)

    # decode (continuous), H2D, compute (back to back), D2H, result
    blk(2.75, 4.45, 0, A, "decode i · 3.5 ms")
    blk(4.5, 6.2, 0, B, "decode i+1")
    blk(6.25, 7.95, 0, PAPER, "decode i+2", ls=(0, (3, 2)), lw=1.2)
    blk(4.5, 4.75, 1, A)
    blk(6.25, 6.5, 1, B)
    for x0, x1, fill, first in [(4.8, 7.4, A, True), (7.45, 10.05, B, False)]:
        blk(x0, x1, 2, fill)
        line(ax, [x0 + 0.25, x0 + 0.25], [Y[2] - 0.19, Y[2] + 0.19], color=INK, lw=1.2, z=4)
        line(ax, [x1 - 0.45, x1 - 0.45], [Y[2] - 0.19, Y[2] + 0.19], color=INK, lw=1.2, z=4)
        if first:
            ax.text(x0 + 0.125, Y[2], "prep", fontsize=9, ha="center", va="center", rotation=90, zorder=5)
            ax.text(x0 + 1.2, Y[2] + 0.04, "model_fn", fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=5)
            ax.text(x1 - 0.22, Y[2], "post", fontsize=9, ha="center", va="center", rotation=90, zorder=5)
        for k, key in enumerate(["BF16", "FP16", "MXFP8", "NVFP4", "MXFP4"]):
            ax.add_patch(Rectangle((x0 + 0.25 + k * 0.38, Y[2] - 0.19), 0.38, 0.07, fc=FMT[key], ec=INK, lw=0.8, zorder=4))
    blk(7.45, 7.7, 3, A)
    blk(10.1, 10.35, 3, B)
    blk(7.75, 8.85, 4, A, "sync + conf", size=9.8)
    blk(10.4, 11.5, 4, B, "sync + conf", size=9.8)
    blk(8.9, 10.9, 5, PAPER, "overlay + mp4 write", ls=(0, (3, 2)), lw=1.2)
    for (x0, y0, x1, y1) in [(4.42, Y[0] - 0.19, 4.55, Y[1] + 0.19), (4.72, Y[1] - 0.19, 4.85, Y[2] + 0.19),
                             (7.38, Y[2] - 0.19, 7.5, Y[3] + 0.19), (7.68, Y[3] - 0.19, 7.8, Y[4] + 0.19)]:
        arrow(ax, x0, y0, x1, y1, lw=1.6, ms=10)
    # in-lane notes
    note_c = INK2
    ax.text(2.78, Y[1], "async copy, 0.15 ms", fontsize=10, va="center", color=note_c)
    ax.text(2.78, Y[2], "pluggable model_fn:\nr5_v1 · r12 · r13 · r14", fontsize=10, va="center", color=note_c, linespacing=1.15)
    ax.text(2.78, Y[3], "uint8 mask 2.7 MB + 4 int64, 0.05 ms", fontsize=10, va="center", color=note_c)
    ax.text(2.78, Y[4], "event.synchronize(), confidence = sum / count", fontsize=10, va="center", color=note_c)
    ax.text(2.78, Y[5], "with overlay the script tops out at ~33 FPS (CPU)", fontsize=10, va="center", color=note_c)
    textbox(ax, 8.3, Y[1] + 0.2, "inflight = 2 frames\n4-8 pinned slots, 1 event / frame", size=10.8)
    # legend for frames
    ly = H - 1.22
    box(ax, 8.6, ly - 0.09, 0.2, 0.18, fc=A, lw=1.2, r=0.04)
    ax.text(8.88, ly, "frame i", fontsize=11, va="center")
    box(ax, 9.95, ly - 0.09, 0.2, 0.18, fc=B, lw=1.2, r=0.04)
    ax.text(10.23, ly, "frame i+1", fontsize=11, va="center")

    # ---- bottom: three measured facts -------------------------------------------------------------
    # (a) launch cost
    a1 = fig.add_axes([0.075, 0.085, 0.215, 0.255])
    style_axes(a1, grid="y")
    panel_title(a1, "(a) launch cost", "per kernel, measured on this card", size=15)
    a1.bar([0, 1], [4.97, 0.75], width=0.55, color=[GRAY[3], GRAY[1]], edgecolor=INK, linewidth=1.7, zorder=3)
    a1.text(0, 4.97 + 0.15, "4.97 µs", fontsize=13, fontweight="bold", ha="center", va="bottom")
    a1.text(1, 0.75 + 0.15, "0.75 µs", fontsize=13, fontweight="bold", ha="center", va="bottom")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["eager\nlaunch", "inside\na graph"], fontsize=11.5)
    a1.set_ylim(0, 7.6)
    a1.set_xlim(-0.6, 1.7)
    a1.set_yticks([0, 2, 4, 6])
    a1.text(0.97, 0.97, "R0: 1302 kernels\n× 4.97 µs ≈ 6.5 ms\nof launch time", transform=a1.transAxes,
            fontsize=10.2, ha="right", va="top", color=INK2)
    # (b) GPU busy
    a2 = fig.add_axes([0.385, 0.085, 0.30, 0.255])
    style_axes(a2, grid="y")
    panel_title(a2, "(b) GPU busy", "nsys, share of the time span", size=15)
    busy = [("R0 model\nViT-L", 71, GRAY[3]), ("R0 model\nViT-B", 50, GRAY[3]), ("R0\npost", 7, GRAY[3]),
            ("R9\nViT-L", 99.0, FMT["BF16"]), ("R9\nViT-B", 94.4, FMT["BF16"])]
    for i, (lab, v, c) in enumerate(busy):
        a2.bar(i, v, width=0.6, color=c, edgecolor=INK, linewidth=1.7, zorder=3)
        a2.text(i, v + 2.5, f"{v:g}%", fontsize=12.5, fontweight="bold", ha="center", va="bottom")
    a2.set_xticks(range(len(busy)))
    a2.set_xticklabels([b[0] for b in busy], fontsize=10.2)
    a2.set_ylim(0, 125)
    a2.set_yticks([0, 25, 50, 75, 100])
    # (c) overlap
    a3 = fig.add_axes([0.765, 0.085, 0.215, 0.255])
    style_axes(a3, grid="y")
    panel_title(a3, "(c) hidden copies", "share overlapped with kernels", size=15)
    for i, (lab, l, b) in enumerate([("H2D", 91, 80), ("D2H", 70, 67)]):
        a3.bar(i - 0.19, l, width=0.34, color=GRAY[3], edgecolor=INK, linewidth=1.7, zorder=3)
        a3.bar(i + 0.19, b, width=0.34, color=GRAY[0], edgecolor=INK, linewidth=1.7, zorder=3)
        a3.text(i - 0.19, l + 2.5, f"{l}%", fontsize=12, fontweight="bold", ha="center", va="bottom")
        a3.text(i + 0.19, b + 2.5, f"{b}%", fontsize=12, fontweight="bold", ha="center", va="bottom")
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["H2D", "D2H"], fontsize=12)
    a3.set_ylim(0, 125)
    a3.set_yticks([0, 25, 50, 75, 100])
    a3.set_xlim(-0.6, 1.6)
    a3.text(0.02, 0.97, "dark ViT-L, light ViT-B", transform=a3.transAxes, fontsize=10, ha="left", va="top", color=INK2)
    source(fig, "round8 / round9 NOTES (design, nsys overlap 91 / 80 / 70 / 67 %, GPU busy 99.0 / 94.4 %), round0 NOTES (busy 71 / 50 / 7 %), results/peaks.json (launch cost).", y=0.008)


# ----------------------------------------------------------------------------------------------------
# infer_05  number formats vs the baseline
# ----------------------------------------------------------------------------------------------------


@fig("infer_05_precision_speed", 12, 9.8)
def infer_05(fig):
    title(fig, "FP32 / BF16 / MXFP8 / NVFP4 / MXFP4 vs the baseline",
          "what the tensor cores can do, what the whole model gets, and where we started")
    keys5 = ["BF16", "FP16", "MXFP8", "NVFP4", "MXFP4"]

    # (a) instruction peaks
    a = fig.add_axes([0.075, 0.545, 0.40, 0.265])
    style_axes(a, grid="y")
    panel_title(a, "(a) Tensor-core instruction peak", "TFLOPS, microbenchmark on this RTX 5090")
    peaks = [("FP32", 118), ("BF16", 247), ("FP16", 493), ("MXFP8", 981), ("NVFP4", 1964), ("MXFP4", 1965)]
    for i, (k, v) in enumerate(peaks):
        fmt_bar(a, i, v, k, width=0.62)
        a.text(i, v + 40, f"{v:,}", fontsize=12.5, fontweight="bold", ha="center", va="bottom")
    a.set_xticks(range(6))
    a.set_xticklabels(["FP32\nFMA", "BF16\nHMMA", "FP16\nacc", "MXFP8", "NVFP4", "MXFP4"], fontsize=11.5)
    a.set_ylim(0, 2350)
    a.set_yticks([0, 500, 1000, 1500, 2000])
    callout(a, (1.66, 440), (-0.35, 1450), "bf16 / fp16 with fp32 accumulate\nis throttled on GeForce;\nfp16 accumulate runs at 2×", size=10.8, rad=0.25)

    # (b) whole model ms
    b = fig.add_axes([0.575, 0.545, 0.40, 0.265])
    style_axes(b, grid="x")
    panel_title(b, "(b) Whole model, ms per frame", "CUDA-graph replay; solid ViT-L, light ViT-B; × vs BF16")
    ml = [9.77, 7.08, 4.90, 4.02, 4.04]
    mb = [3.62, 2.70, 1.94, 1.74, 1.75]
    sl = ["1.00×", "1.38×", "1.98×", "2.43×", "2.42×"]
    sb = ["1.00×", "1.34×", "1.90×", "2.08×", "2.07×"]
    for i, k in enumerate(keys5):
        y = 4 - i
        b.barh(y + 0.19, ml[i], height=0.34, color=FMT[k], edgecolor=INK, linewidth=1.6, zorder=3)
        b.barh(y - 0.19, mb[i], height=0.34, color=tint(FMT[k], 0.42), edgecolor=INK, linewidth=1.6, zorder=3)
        b.text(ml[i] + 0.15, y + 0.19, f"{ml[i]:.2f}  {sl[i]}", fontsize=11.5, fontweight="bold", va="center")
        b.text(mb[i] + 0.15, y - 0.19, f"{mb[i]:.2f}  {sb[i]}", fontsize=11, va="center", color=INK2)
    b.set_yticks(range(5))
    b.set_yticklabels([LABEL[k] + (" †" if k == "MXFP8" else "") for k in keys5][::-1], fontsize=12)
    b.set_xlim(0, 14.5)
    b.set_ylim(-0.6, 4.6)
    b.set_xticks([0, 2, 4, 6, 8, 10])

    # (c) ceiling vs achieved
    c = fig.add_axes([0.075, 0.095, 0.40, 0.265])
    style_axes(c, grid="y")
    panel_title(c, "(c) Hardware ceiling vs what the model gets", "× over BF16; hatched = instruction peak, solid = ViT-L, light = ViT-B")
    grp = [("FP16", 2.00, 1.38, 1.34), ("MXFP8", 3.97, 1.98, 1.90), ("NVFP4", 7.95, 2.43, 2.08)]
    for i, (k, p, l, bb) in enumerate(grp):
        c.bar(i - 0.27, p, width=0.25, color=FMT[k], edgecolor=INK, linewidth=1.6, hatch="//", zorder=3)
        c.bar(i, l, width=0.25, color=FMT[k], edgecolor=INK, linewidth=1.6, zorder=3)
        c.bar(i + 0.27, bb, width=0.25, color=tint(FMT[k], 0.42), edgecolor=INK, linewidth=1.6, zorder=3)
        c.text(i - 0.27, p + 0.15, f"{p:.2f}", fontsize=11, fontweight="bold", ha="center", va="bottom")
        c.text(i, l + 0.15, f"{l:.2f}", fontsize=11, fontweight="bold", ha="center", va="bottom")
        c.text(i + 0.27, bb + 0.15, f"{bb:.2f}", fontsize=11, ha="center", va="bottom", color=INK2)
    c.set_xticks(range(3))
    c.set_xticklabels(["FP16-acc", "MXFP8", "NVFP4 / MXFP4"], fontsize=11.5)
    c.set_ylim(0, 9.4)
    c.set_xlim(-0.6, 2.75)
    c.set_yticks([0, 2, 4, 6, 8])
    textbox(c, -0.5, 7.5, "outside the GEMMs: 2.0 of 3.8 ms\n(attention 0.85, LN + quantize 0.60 ...)\nstay in fp16 / bf16", size=10.6, ha="left")

    # (d) where we started
    d = fig.add_axes([0.575, 0.095, 0.40, 0.265])
    style_axes(d, grid="y")
    panel_title(d, "(d) Where we started: original script", "model section, ms (bf16 autocast was already the baseline)")
    data = [("ViT-L", 40.84, 17.05, "12.3", "16.5"), ("ViT-B", 15.26, 9.62, "17.3", "19.3")]
    for i, (m, f32, b16, f1, f2) in enumerate(data):
        fmt_bar(d, i - 0.2, f32, "FP32", width=0.34)
        fmt_bar(d, i + 0.2, b16, "BF16", width=0.34)
        d.text(i - 0.2, f32 + 1.0, f"{f32:.1f} ms", fontsize=12, fontweight="bold", ha="center", va="bottom")
        d.text(i - 0.2, f32 + 6.5, f"e2e {f1} FPS", fontsize=10.5, ha="center", va="bottom", color=INK2)
        d.text(i + 0.2, b16 + 1.0, f"{b16:.1f} ms", fontsize=12, fontweight="bold", ha="center", va="bottom")
        d.text(i + 0.2, b16 + 6.5, f"e2e {f2} FPS", fontsize=10.5, ha="center", va="bottom", color=INK2)
    d.set_xticks([0, 1])
    d.set_xticklabels(["ViT-L/16", "ViT-B/16"], fontsize=12)
    d.set_ylim(0, 58)
    d.set_xlim(-0.6, 1.6)
    d.set_yticks([0, 10, 20, 30, 40])
    d.text(0.98, 0.96, "FP32 = grey, BF16 = blue", transform=d.transAxes, fontsize=10.5, ha="right", va="top", color=INK2)
    source(fig, "round10 §1 (peaks), round13 / round14 (speed12, CUDA-graph replay, median of 3), round0 / results/summary.md (original script).  "
           "† MXFP8 from the round13 session.", y=0.008)


# ----------------------------------------------------------------------------------------------------
# infer_06  kernel breakdown
# ----------------------------------------------------------------------------------------------------


@fig("infer_06_kernel_breakdown", 12, 7.8)
def infer_06(fig):
    from matplotlib.patches import Rectangle

    title(fig, "Where the GPU time goes, kernel by kernel",
          "ViT-L/16 model section, ms per frame (nsys kernel sums; prep / post excluded)")
    ax = fig.add_axes([0.06, 0.115, 0.45, 0.65])
    style_axes(ax, grid="y")
    panel_title(ax, "(a) Three stacks, one model", "colour = the GEMMs (their number format); grey = everything else")
    cats = ["GEMM (backbone + ConvT)", "attention", "LN + quantize", "RoPE / layout pass", "head (dw_ln, einsum)", "other (patch-embed ..)"]
    cat_fill = [None, GRAY[3], GRAY[1], GRAY[0], GRAY[2], GRAY[4]]
    stacks = [
        ("BF16", "BF16\nr5_v1", [7.263, 1.450, 0.233, 0.0, 0.426, 0.203]),
        ("MXFP8", "MXFP8\nr13", [2.972, 0.839, 0.271, 0.161, 0.332, 0.037]),
        ("NVFP4", "NVFP4\nr14", [1.825, 0.848, 0.595, 0.193, 0.289, 0.070]),
    ]
    for i, (k, lab, vals) in enumerate(stacks):
        bot = 0.0
        for j, v in enumerate(vals):
            if v <= 0:
                continue
            fc = FMT[k] if j == 0 else cat_fill[j]
            ax.bar(i, v, bottom=bot, width=0.62, color=fc, edgecolor=INK, linewidth=1.5, zorder=3,
                   hatch="//" if j == 2 else None)
            if v >= 0.7:
                ptext(ax, i, bot + v / 2, f"{v + 1e-6:.2f}", fc, size=12.5, fontweight="bold", ha="center", va="center", z=8)
            bot += v
        ax.text(i, bot + 0.18, f"{bot + 1e-6:.2f} ms", fontsize=14, fontweight="bold", ha="center", va="bottom")
    ax.set_xlim(-0.55, 3.85)
    ax.set_ylim(0, 11.2)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[1] for s in stacks], fontsize=12.5)
    textbox(ax, 0.5, 7.0, "−52%", size=13, ha="center")
    textbox(ax, 1.5, 5.8, "−17%", size=13, ha="center")
    ly = 9.8
    for j in range(5, -1, -1):
        fc = FMT["NVFP4"] if j == 0 else cat_fill[j]
        ax.add_patch(Rectangle((2.62, ly - 0.17), 0.16, 0.34, fc=fc, ec=INK, lw=1.3, zorder=4, hatch="//" if j == 2 else None))
        ax.text(2.86, ly, cats[j] if j else "GEMM (colour = format)", fontsize=10.8, va="center", ha="left")
        ly -= 0.72
    # ---- (b) what is left in the NVFP4 frame --------------------------------------------------------
    bx = fig.add_axes([0.745, 0.115, 0.23, 0.65])
    bx.set_xlim(0, 2.1)
    bx.set_ylim(1.4, 14.0)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.spines["left"].set_visible(False)
    bx.set_yticks([])
    bx.tick_params(labelsize=11.5)
    bx.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    bx.grid(axis="x", color=RULE, linewidth=1.1, zorder=0)
    bx.set_axisbelow(True)
    fig.text(0.545, 0.815, "(b) What is left, and what could go", fontsize=16.5, fontweight="bold", ha="left")
    fig.text(0.547, 0.787, "NVFP4 frame (3.82 ms); hatched = estimates", fontsize=11.5, color=INK2, ha="left")
    now = [("GEMMs", 1.825, FMT["NVFP4"], None), ("attention", 0.848, GRAY[3], None), ("LN + quantize", 0.595, GRAY[1], "//"),
           ("head", 0.289, GRAY[2], None), ("RoPE / layout", 0.193, GRAY[0], None), ("other", 0.070, GRAY[4], None)]
    y = 12.4
    for lab, v, fc, h in now:
        bx.barh(y, v, height=0.55, color=fc, edgecolor=INK, linewidth=1.5, hatch=h, zorder=3)
        bx.text(v + 0.04, y, f"{v + 1e-6:.2f}", fontsize=11.5, fontweight="bold", va="center")
        fig.text(0.72, 0.115 + 0.65 * (y - 1.4) / 12.6, lab, fontsize=11.5, ha="right", va="center")
        y -= 0.8
    could = [("fuse LN + quantize", 0.18), ("GELU quant into fc1", 0.10), ("attn-out quant fused", 0.07),
             ("RoPE into attention", 0.19), ("FP4 attention kernel", 0.17)]
    fig.text(0.545, 0.115 + 0.65 * (7.25 - 1.4) / 12.6, "could still go (ms saved)", fontsize=12.5, fontweight="bold", ha="left", va="center")
    y = 6.3
    for lab, v in could:
        bx.barh(y, v, height=0.55, color=GRAY[1], edgecolor=INK, linewidth=1.4, hatch="//", linestyle="--", zorder=3)
        bx.text(v + 0.04, y, f"−{v:.2f}", fontsize=11.5, va="center")
        fig.text(0.72, 0.115 + 0.65 * (y - 1.4) / 12.6, lab, fontsize=11.5, ha="right", va="center")
        y -= 0.8
    textbox(bx, 2.05, 2.15, "fusions alone: −0.4 .. −0.5 ms\n(10-12% of the frame)", size=10.8, ha="right")
    source(fig, "round10 §2 (r5_v1 nsys), round13 §3 (MXFP8 nsys, PDL off), round14 §7.2 / §7.4 (NVFP4 nsys, to-do list).\n"
           "r5_v1 prep + post (0.17 ms) removed so all three bars are model-only.", y=0.008)


# ----------------------------------------------------------------------------------------------------
# infer_07  speed vs accuracy
# ----------------------------------------------------------------------------------------------------


@fig("infer_07_accuracy_tradeoff", 12, 7.6)
def infer_07(fig):
    title(fig, "What the speed costs in accuracy",
          "whole-model numbers are speed-only (gate off); the old head-IoU gate is shown from the round-10 emulation")
    keys4 = ["FP16", "MXFP8", "NVFP4", "MXFP4"]
    # (a) cosine deviation
    a = fig.add_axes([0.075, 0.235, 0.40, 0.50])
    style_axes(a, grid="y")
    panel_title(a, "(a) Mask logits vs the BF16 baseline", "1 − cosine, log scale (lower is better); real kernels, recorded only")
    cosL = [0.999996, 0.999892, 0.998997, 0.995952]
    cosB = [0.999992, 0.999505, 0.997203, 0.988281]
    dL = [1 - c for c in cosL]
    dB = [1 - c for c in cosB]
    a.plot(range(4), dL, color=INK2, lw=1.6, zorder=3)
    a.plot(range(4), dB, color=INK2, lw=1.6, zorder=3)
    for i, k in enumerate(keys4):
        a.plot([i], [dL[i]], marker="o", ms=15, mfc=FMT[k], mec=INK, mew=1.6, zorder=6, linestyle="none")
        a.plot([i], [dB[i]], marker="s", ms=14, mfc=FMT[k], mec=INK, mew=1.6, zorder=6, linestyle="none")
        a.text(i + 0.17, dL[i] * 0.78, f"{cosL[i]:.6g}" if i < 2 else f"{cosL[i]:.4f}", fontsize=11, ha="left", va="top")
        a.text(i + 0.17, dB[i] * 1.28, f"{cosB[i]:.6g}" if i < 2 else f"{cosB[i]:.4f}", fontsize=11, ha="left", va="bottom", color=INK2)
    a.set_yscale("log")
    a.set_ylim(1.5e-6, 6e-2)
    a.set_xlim(-0.5, 3.5)
    a.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    a.set_yticklabels([r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"], fontsize=12)
    a.set_xticks(range(4))
    a.set_xticklabels(["FP16-acc", "MXFP8", "NVFP4", "MXFP4"], fontsize=12.5)
    a.plot([0.045], [0.945], marker="o", ms=11, mfc=GRAY[1], mec=INK, mew=1.4, transform=a.transAxes, clip_on=False, zorder=8)
    a.text(0.085, 0.945, "ViT-L", transform=a.transAxes, fontsize=11.5, ha="left", va="center")
    a.plot([0.255], [0.945], marker="s", ms=10, mfc=GRAY[1], mec=INK, mew=1.4, transform=a.transAxes, clip_on=False, zorder=8)
    a.text(0.295, 0.945, "ViT-B   (labels = cosine)", transform=a.transAxes, fontsize=11.5, ha="left", va="center")
    # (b) emulated head IoU
    b = fig.add_axes([0.575, 0.235, 0.40, 0.50])
    style_axes(b, grid="y")
    panel_title(b, "(b) Round-10 emulation: head IoU", "whole backbone fake-quantised (RN); solid ViT-L, light ViT-B")
    tiers = [("BF16", 0.9946, 0.9943), ("FP16", 0.9942, 0.9934), ("MXFP8", 0.8671, 0.9469), ("NVFP4", 0.8658, 0.8189), ("MXFP4", 0.6961, 0.4767)]
    for i, (k, l, bb) in enumerate(tiers):
        b.bar(i - 0.2, l, width=0.36, color=FMT[k], edgecolor=INK, linewidth=1.6, zorder=3)
        b.bar(i + 0.2, bb, width=0.36, color=tint(FMT[k], 0.42), edgecolor=INK, linewidth=1.6, zorder=3)
        lift = 0.075 if abs(l - bb) < 0.06 else 0.0   # same-height neighbours: B label sits above the L label
        b.text(i - 0.2, l + 0.015, f"{l:.3f}" if l > 0.9 else f"{l:.2f}", fontsize=10.8, fontweight="bold", ha="center", va="bottom")
        b.text(i + 0.2, bb + 0.015 + lift, f"{bb:.3f}" if bb > 0.9 else f"{bb:.2f}", fontsize=10.8, ha="center", va="bottom", color=INK2)
    b.axhline(0.95, color=INK, lw=1.6, ls=(0, (5, 4)), zorder=2)
    b.text(4.62, 1.095, "dashed = old gate: head IoU ≥ 0.95", fontsize=11, ha="right", va="bottom")
    b.set_ylim(0, 1.16)
    b.set_xlim(-0.6, 4.6)
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xticks(range(5))
    b.set_xticklabels(["BF16", "FP16", "MXFP8", "NVFP4", "MXFP4"], fontsize=12)
    # banner
    ov = canvas(fig, 12, 7.6)
    box(ov, 0.45, 0.38, 11.1, 0.78, fc=PAPER, lw=1.7, r=0.12, z=2)
    ov.text(0.65, 0.77, "The accuracy gate is OFF for MXFP8 / NVFP4 / MXFP4 (user decision, 2026-10-05: the checkpoint is bf16-trained).\n"
            "Only the mask-logit cosine was recorded. Head IoU and confidence were never measured with the real FP8 / FP4 kernels.",
            fontsize=11.8, va="center", ha="left", zorder=5, linespacing=1.3)
    source(fig, "round12 §2.3 / round13 §0 / round14 §7.1 (cosines vs r5_v1); round10 §4.4 stage A (emulation, 8-frame gate M, 'backbone4' rows).", y=0.008)


# ----------------------------------------------------------------------------------------------------
# infer_08  whole-network structure with the quantised GEMMs
# ----------------------------------------------------------------------------------------------------

_Q3 = ["MXFP8", "NVFP4", "MXFP4"]


def _qband(ax, x, y, w, h=0.1, z=4):
    """Three-colour band = 'this GEMM runs in MXFP8 / NVFP4 / MXFP4 (one per stack)'."""
    from matplotlib.patches import Rectangle
    for i, k in enumerate(_Q3):
        ax.add_patch(Rectangle((x + i * w / 3, y), w / 3, h, fc=FMT[k], ec=INK, lw=1.0, zorder=z))


def _nbox(ax, x, y, w, h, ttl, sub="", fill=PAPER, q=False, ts=10.8, ss=9.0, lw=1.6):
    """A network box centred text; q=True marks a quantised GEMM (three-colour band along the bottom)."""
    box(ax, x, y, w, h, fc=fill, lw=lw, r=0.07, z=3)
    cy = y + h / 2 + (0.05 if q else 0)
    ptext(ax, x + w / 2, cy + (0.09 if sub else 0), ttl, fill, size=ts, fontweight="bold", ha="center", va="center", z=6)
    if sub:
        ptext(ax, x + w / 2, cy - 0.12, sub, fill, size=ss, ha="center", va="center", z=6)
    if q:
        _qband(ax, x + 0.04, y + 0.03, w - 0.08, 0.09)


@fig("infer_08_model_structure", 12, 9.9)
def infer_08(fig):
    W, H = 12, 9.9
    ax = canvas(fig, W, H)
    title(fig, "Inference: where MXFP8 / NVFP4 / MXFP4 sit in the network",
          "DINOv3 ViT + EoMT head, 2 crops of 512x512 per frame; three-colour band = quantised GEMM, the rest keeps its precision")

    # ---- row A: the whole frame pipeline -------------------------------------------------------
    A_BF = tint(FMT["BF16"], 0.3)
    stages = [("Frame", "1900x1440", "uint8 BGR", GRAY[0], False, "uint8"),
              ("GPU prep", "resize 675x512", "2 crops 512²", GRAY[0], False, "bf16 [2,3,512,512]"),
              ("Patch embed", "16x16 conv = GEMM", "bf16, kept", A_BF, False, "fp32 x [2,1029,C]"),
              ("ViT blocks", "x24 (L) / x12 (B)", "last 3 = joint", PAPER, True, "fp32 residual"),
              ("EoMT head", "ConvT x2 quantised", "rest bf16", PAPER, True, "mask logits [2,100,128,128]"),
              ("GPU post", "softmax·sigmoid", "argmax → mask", GRAY[0], False, "uint8 mask 2.7 MB")]
    bw, gap = 1.55, 0.36
    for i, (a, b, c, fill, q, cap) in enumerate(stages):
        x = 0.45 + i * (bw + gap)
        box(ax, x, 7.62, bw, 0.92, fc=fill, lw=1.7, r=0.08, z=3)
        ptext(ax, x + bw / 2, 8.33, a, fill, size=11.5, fontweight="bold", ha="center", va="center", z=6)
        ptext(ax, x + bw / 2, 8.06 if not q else 8.1, b, fill, size=9.4, ha="center", va="center", z=6)
        ptext(ax, x + bw / 2, 7.84 if not q else 7.9, c, fill, size=9.4, ha="center", va="center", z=6)
        if q:
            _qband(ax, x + 0.05, 7.66, bw - 0.1, 0.1)
        ax.text(x + bw / 2, 7.36, cap, fontsize=9, ha="center", va="center", color=INK2)
        if i < len(stages) - 1:
            arrow(ax, x + bw + 0.02, 8.08, x + bw + gap - 0.02, 8.08, lw=1.8, ms=12)

    # ---- row B left: one ViT block ------------------------------------------------------------
    text(ax, 0.45, 6.88, "ViT block (x24 for L, x12 for B)", size=14.5, weight="bold")
    bwid, bh = 1.0, 0.62
    r1 = [("LN", "fp32 stats", GRAY[1], False), ("qkv", "d → 3d", PAPER, True), ("RoPE", "fp32", GRAY[1], False),
          ("attention", "fp16-acc", FMT["FP16"], False), ("proj", "d → d, +x", PAPER, True)]
    r2 = [("LN", "fp32 stats", GRAY[1], False), ("fc1", "d → 4d, GELU", PAPER, True), ("fc2", "4d → d, +x", PAPER, True)]
    y1, y2 = 5.95, 4.95
    for i, (a, b, f, q) in enumerate(r1):
        x = 0.5 + i * (bwid + 0.2)
        _nbox(ax, x, y1, bwid, bh, a, b, fill=f, q=q)
        if i < len(r1) - 1:
            arrow(ax, x + bwid + 0.01, y1 + bh / 2, x + bwid + 0.19, y1 + bh / 2, lw=1.5, ms=10)
    for i, (a, b, f, q) in enumerate(r2):
        x = 0.5 + i * (bwid + 0.2)
        _nbox(ax, x, y2, bwid, bh, a, b, fill=f, q=q)
        if i < len(r2) - 1:
            arrow(ax, x + bwid + 0.01, y2 + bh / 2, x + bwid + 0.19, y2 + bh / 2, lw=1.5, ms=10)
    xe, xs0, ym = 0.5 + 4 * (bwid + 0.2) + bwid / 2, 0.5 + bwid / 2, 5.78   # proj output -> MLP half (elbow, below row 1)
    line(ax, [xe, xe], [y1 - 0.02, ym], color=INK2, lw=1.4, z=2)
    line(ax, [xe, xs0], [ym, ym], color=INK2, lw=1.4, z=2)
    arrow(ax, xs0, ym, xs0, y2 + bh + 0.02, lw=1.4, ms=10, color=INK2)
    # activation quantisers sit between producer and GEMM
    for qx, qy, lab in [(0.5 + 1 * (bwid + 0.2) - 0.1, y1 + bh / 2, "Q"), (0.5 + 4 * (bwid + 0.2) - 0.1, y1 + bh / 2, "Q"),
                        (0.5 + 1 * (bwid + 0.2) - 0.1, y2 + bh / 2, "Q"), (0.5 + 2 * (bwid + 0.2) - 0.1, y2 + bh / 2, "Q")]:
        box(ax, qx - 0.1, qy - 0.1, 0.2, 0.2, fc=tint(FMT["NVFP4"], 0.55), lw=1.3, r=0.04, z=8)
        ax.text(qx, qy, lab, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=9)
    # residual rail
    line(ax, [0.5, 6.1], [y2 - 0.3, y2 - 0.3], color=GRAY[3], lw=3.2, z=1)
    ax.text(0.5, y2 - 0.52, "fp32 residual stream x: never quantised", fontsize=9.6, color=INK2, va="center")
    for xx in (0.5 + 4 * (bwid + 0.2) + bwid / 2, 0.5 + 2 * (bwid + 0.2) + bwid / 2):
        line(ax, [xx, xx], [y2 - 0.3, (y1 if xx > 4 else y2) ], color=GRAY[3], lw=1.6, z=1)
    note(ax, 0.5, 3.72, "last 3 blocks are 'joint': 100 learned query tokens join the\n1029 image tokens (N = 1129 per crop) and share the same Linears", size=9.6, w=5.6, h=0.5)

    # ---- row B right: the head ----------------------------------------------------------------
    text(ax, 6.75, 6.88, "EoMT head (patch tokens + queries)", size=14.5, weight="bold")
    hw, hh = 0.86, 0.62
    xs = [6.75 + i * (hw + 0.15) for i in range(5)]
    lane1 = [("patch", "32x32 grid", GRAY[0], False), ("ConvT1", "k2 s2", PAPER, True), ("dw conv", "GELU, LN2D", GRAY[1], False),
             ("ConvT2", "k2 s2", PAPER, True), ("dw conv", "GELU, LN2D", GRAY[1], False)]
    for i, (a, b, f, q) in enumerate(lane1):
        _nbox(ax, xs[i], y1, hw, hh, a, b, fill=f, q=q, ts=9.6, ss=8.4)
        if i < 4:
            arrow(ax, xs[i] + hw + 0.01, y1 + hh / 2, xs[i + 1] - 0.01, y1 + hh / 2, lw=1.4, ms=9)
    ax.text(xs[4] + hw / 2, y1 + hh + 0.14, "C x 128 x 128", fontsize=9, ha="center", color=INK2)
    A_B = FMT["BF16"]
    _nbox(ax, xs[0], y2, hw, hh, "queries", "100 x C", fill=GRAY[0], ts=9.6, ss=8.4)
    _nbox(ax, xs[1], y2, hw + 0.2, hh, "mask head", "3 Linear, bf16", fill=A_B, ts=9.6, ss=8.4)
    arrow(ax, xs[0] + hw + 0.01, y2 + hh / 2, xs[1] - 0.01, y2 + hh / 2, lw=1.4, ms=9)
    _nbox(ax, xs[3], y2, hw + 0.45, hh, "einsum bqc,bchw", "bf16 GEMM", fill=A_B, ts=9.6, ss=8.4)
    arrow(ax, xs[1] + hw + 0.21, y2 + hh / 2, xs[3] - 0.01, y2 + hh / 2, lw=1.4, ms=9)
    arrow(ax, xs[4] + hw / 2, y1 - 0.02, xs[4] + hw / 2 - 0.05, y2 + hh + 0.02, lw=1.4, ms=9)
    _nbox(ax, xs[1], 3.95, hw + 0.2, hh, "class head", "Linear, 3 logits", fill=A_B, ts=9.6, ss=8.4)
    _nbox(ax, xs[0], 3.95, hw, hh, "queries", "", fill=GRAY[0], ts=9.6, ss=8.4)
    arrow(ax, xs[0] + hw + 0.01, 3.95 + hh / 2, xs[1] - 0.01, 3.95 + hh / 2, lw=1.4, ms=9)
    ax.text(xs[3] + 0.1, 4.28, "→ class logits [2,100,3]", fontsize=9.4, va="center")
    ax.text(xs[3] + 0.1, 4.55, "mask logits [2,100,128,128]", fontsize=9.4, va="center", color=INK2)

    # ---- legend ---------------------------------------------------------------------------------
    ly = 3.38
    lx = 0.5
    box(ax, lx, ly - 0.11, 0.55, 0.22, fc=PAPER, lw=1.3, r=0.04, z=3)
    _qband(ax, lx + 0.02, ly - 0.09, 0.51, 0.07)
    ax.text(lx + 0.65, ly, "quantised GEMM (MXFP8 | NVFP4 | MXFP4)", fontsize=9.6, va="center")
    lx = 4.05
    box(ax, lx, ly - 0.11, 0.2, 0.22, fc=tint(FMT["NVFP4"], 0.55), lw=1.3, r=0.04, z=3)
    ax.text(lx + 0.1, ly, "Q", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(lx + 0.3, ly, "activation quantiser", fontsize=9.6, va="center")
    for lab, fill, x in [("fp16 accumulate", FMT["FP16"], 6.0), ("bf16 GEMM", FMT["BF16"], 7.75), ("LN / RoPE / dw conv", GRAY[1], 9.2)]:
        box(ax, x, ly - 0.11, 0.2, 0.22, fc=fill, lw=1.3, r=0.04, z=3)
        ax.text(x + 0.3, ly, lab, fontsize=9.6, va="center")

    # ---- the three stacks side by side -----------------------------------------------------------
    top = 3.08
    cw0, cw = 1.75, 3.12
    cols = [("MXFP8 · r13", "MXFP8"), ("NVFP4 · r14", "NVFP4"), ("MXFP4 · r14", "MXFP4")]
    for j, (lab, k) in enumerate(cols):
        x = 0.45 + cw0 + j * cw
        box(ax, x + 0.03, top - 0.4, cw - 0.06, 0.4, fc=FMT[k], lw=1.6, r=0.07, z=3)
        ptext(ax, x + cw / 2, top - 0.2, lab, FMT[k], size=12, fontweight="bold", ha="center", va="center", z=6)
    rows = [("element · scale", ["E4M3 · UE8M0 per 1x32", "e2m1 · E4M3/16 + fp32 tensor scale", "e2m1 · UE8M0 per 1x32"]),
            ("bits per value", ["8.25", "4.5", "4.25"]),
            ("GEMM", ["own mgx (qkv: cuBLASLt)", "QuACK mma.sync mxf4nvf4", "QuACK, UE8M0 scales"]),
            ("quantiser", ["fused into LN / fc1 epilogue", "separate fp4q2 kernels", "separate fp4q2 kernels"]),
            ("activation scale", ["dynamic per block", "tensor scale: static, 1st call", "dynamic per block"]),
            ("weights", ["quantised once, K-major", "once, 1-D block + tensor scale", "once, K-major"]),
            ("ViT-L model section", ["4.90 ms  ·  1.98x", "4.02 ms  ·  2.43x", "4.04 ms  ·  2.42x"])]
    for i, (lab, vals) in enumerate(rows):
        y = top - 0.4 - 0.31 * (i + 1) + 0.155
        line(ax, [0.45, 0.45 + cw0 + 3 * cw], [y - 0.155, y - 0.155], color=RULE, lw=1.0, z=1)
        ax.text(0.5, y, lab, fontsize=9.8, fontweight="bold", va="center")
        for j, v in enumerate(vals):
            ax.text(0.45 + cw0 + j * cw + 0.1, y, v, fontsize=9.6, va="center", fontweight="bold" if i == 6 else "normal")
    source(fig, "structure: src/.../dinov3_eomt_semantic_segmentation (task_model.py, scale_block.py); stacks: infer-opt round12-14 NOTES.", y=0.008)


# --- next figure goes here ---
