"""Hand-drawn (matplotlib xkcd-mode) helpers for the lightly-train optimisation figures.

Design rules (from the dataviz skill, adapted to a deliberately hand-drawn look):
  * one colour per number format, the same in every figure (documented 5-slot palette,
    validated with validate_palette.js: adjacent CVD dE 9.1, normal-vision dE 19.6);
  * FP32 / "original" is neutral grey, never a categorical hue;
  * text always wears ink colours, never the series colour; every bar carries its value;
  * hatching (45 degrees only) means "estimated / not measured";
  * all coordinates of schematic canvases are in inches.
"""
from __future__ import annotations

import logging
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 'xkcd' family lookups are noise

OUT = pathlib.Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = "#fbf7ec"
INK = "#2b2a28"
INK2 = "#55524c"
MUTED = "#7d7a71"
RULE = "#e2dac4"
SOFT = "#efe8d2"

# number-format identity colours (documented categorical slots 1-5, FP32 = neutral grey)
FMT = {
    "FP32": "#a8a59d",
    "BF16": "#2a78d6",
    "FP16": "#eb6834",
    "MXFP8": "#1baf7a",
    "NVFP4": "#eda100",
    "MXFP4": "#e87ba4",
}
LABEL = {"FP32": "FP32", "BF16": "BF16", "FP16": "FP16-acc", "MXFP8": "MXFP8", "NVFP4": "NVFP4", "MXFP4": "MXFP4"}
# neutral tints for "what kind of kernel" categories (identity of the *format* is carried by FMT only)
GRAY = ["#e7e2d3", "#d3ccb8", "#b9b19b", "#9b937e", "#7b745f"]

ARROW = r"$\rightarrow$"


def render(name, w, h, draw, dpi=150, scale=0.75, length=110, randomness=1.6):
    """Create a hand-drawn figure of w x h inches, call draw(fig), save figures/<name>.png."""
    with plt.xkcd(scale=scale, length=length, randomness=randomness):
        plt.rcParams.update(
            {
                "font.family": ["Comic Sans MS", "DejaVu Sans"],  # DejaVu only supplies glyphs Comic lacks (arrows)
                "path.effects": [pe.withStroke(linewidth=3, foreground=PAPER)],
                "axes.linewidth": 1.7,
                "axes.edgecolor": INK,
                "axes.labelcolor": INK,
                "text.color": INK,
                "xtick.color": INK,
                "ytick.color": INK,
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "xtick.major.width": 1.4,
                "ytick.major.width": 1.4,
                "hatch.linewidth": 1.3,
                "figure.facecolor": PAPER,
                "axes.facecolor": PAPER,
                "savefig.facecolor": PAPER,
            }
        )
        fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=PAPER)
        draw(fig)
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=dpi, facecolor=PAPER)
        plt.close(fig)
    try:  # 256-colour palette: visually identical for flat hand-drawn art, about a third of the size
        from PIL import Image

        Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                 dither=Image.Dither.NONE).save(path, optimize=True)
    except ImportError:
        pass
    return path


def canvas(fig, w, h, rect=(0, 0, 1, 1)):
    """An axes whose data coordinates are inches (origin bottom-left), no frame."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def title(fig, text, sub=None, y=0.968):
    fig.text(0.035, y, text, fontsize=25, fontweight="bold", ha="left", va="top", color=INK)
    if sub:
        fig.text(0.037, y - 0.058, sub, fontsize=13.5, ha="left", va="top", color=INK2)


def source(fig, text, y=0.012):
    fig.text(0.035, y, text, fontsize=10.5, ha="left", va="bottom", color=MUTED)


def style_axes(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=13)
    if grid:
        ax.grid(axis=grid, color=RULE, linewidth=1.1, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, text, sub=None, size=16.5):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", pad=27 if sub else 10, color=INK)
    if sub:  # offset in points, so it sits the same distance above the axes whatever the axes height
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    fontsize=11.5, color=INK2, ha="left", va="bottom")


def box(ax, x, y, w, h, fc=PAPER, ec=INK, lw=1.6, r=0.07, hatch=None, ls="-", z=2, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=z, alpha=alpha,
    )
    if hatch:
        p.set_hatch(hatch)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=13, color=INK, ha="left", va="center", weight="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.9, rad=0.0, style="-|>", ms=15, ls="-", z=4):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)
    return a


def line(ax, xs, ys, color=INK, lw=1.8, ls="-", z=3):
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=z, solid_capstyle="round")


def note(ax, x, y, s, size=12, w=None, h=None, fc=PAPER, ec=INK, lw=1.4, ha="left", z=6):
    """A boxed hand-written note. (x, y) is its bottom-left corner; w, h in inches (auto if None)."""
    lines = s.split("\n")
    nl = len(lines)
    longest = max(len(t) for t in lines)
    w = w or (0.105 * size / 12 * longest + 0.28)
    h = h or (0.205 * size / 12 * nl + 0.2)
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.09, z=z)
    ax.text(x + 0.14, y + h / 2, s, fontsize=size, ha="left", va="center", color=INK, zorder=z + 1, linespacing=1.25)
    return w, h


def squiggle(ax, x0, x1, y, amp=0.035, waves=14, color=INK, lw=2.0):
    xs = np.linspace(x0, x1, 200)
    ys = y + amp * np.sin(np.linspace(0, waves * 2 * np.pi, 200))
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)


def star(ax, x, y, r=0.16, fc=FMT["NVFP4"], ec=INK, z=7):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 11)[:-1]
    rad = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
    pts = np.c_[x + rad * np.cos(ang), y + rad * np.sin(ang)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=z))


def swatch(ax, x, y, key, size=0.2, hatch=None):
    box(ax, x, y - size / 2, size, size, fc=FMT[key], r=0.04, lw=1.4, hatch=hatch)


def legend_fmt(ax, x, y, keys, size=13, gap=0.28, sw=0.22, hatch_keys=()):
    """Horizontal legend of number formats: swatch + ink label. Returns x after the last item."""
    cx = x
    for k in keys:
        swatch(ax, cx, y, k, size=sw, hatch="//" if k in hatch_keys else None)
        t = LABEL[k]
        text(ax, cx + sw + 0.09, y, t, size=size)
        cx += sw + 0.09 + 0.098 * size / 13 * len(t) + gap
    return cx


def bar_label(ax, x, y, s, size=13, ha="center", va="bottom", weight="bold", color=INK):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight=weight, color=color, zorder=8)


def fmt_bar(ax, x, h, key, width=0.62, hatch=None, dashed=False, **kw):
    """One measured (or hatched = estimated) bar in the colour of its number format."""
    return ax.bar(
        x, h, width=width, color=FMT[key] if not dashed else PAPER, edgecolor=INK, linewidth=1.7,
        hatch=hatch, linestyle="--" if dashed else "-", zorder=3, **kw,
    )


def barh_fmt(ax, y, v, key, height=0.58, hatch=None, dashed=False, **kw):
    return ax.barh(
        y, v, height=height, color=FMT[key] if not dashed else PAPER, edgecolor=INK, linewidth=1.7,
        hatch=hatch, linestyle="--" if dashed else "-", zorder=3, **kw,
    )


def callout(ax, xy, xytext, s, size=12, ha="left", va="center", rad=0.2, fc=PAPER, lw=1.6, z=9, coords="data"):
    """Hand-written note with an arrow pointing at xy (data coordinates unless coords is given)."""
    return ax.annotate(
        s, xy=xy, xycoords=coords, xytext=xytext, textcoords=coords, fontsize=size, ha=ha, va=va, color=INK,
        zorder=z, linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.5", fc=fc, ec=INK, lw=1.4),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw, connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=3, mutation_scale=13),
    )


def textbox(ax, x, y, s, size=12, ha="left", va="center", fc=PAPER, z=9, coords="data", pad=0.38):
    """A boxed note without an arrow."""
    return ax.text(
        x, y, s, fontsize=size, ha=ha, va=va, color=INK, zorder=z, linespacing=1.25, transform=(
            ax.transAxes if coords == "axes" else ax.transData),
        bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.5", fc=fc, ec=INK, lw=1.4),
    )


def star_marker(ax, x, y, size=30, fc=FMT["NVFP4"], z=10):
    """A star drawn as a marker (works on any data scale)."""
    ax.plot([x], [y], marker="*", markersize=size, markerfacecolor=fc, markeredgecolor=INK, markeredgewidth=1.5,
            linestyle="none", zorder=z)


# ---- figure registry -------------------------------------------------------------------------------
REGISTRY = {}


def fig(name, w, h):
    """Decorator: register draw(fig) under figures/<name>.png with a canvas of w x h inches."""
    def deco(fn):
        REGISTRY[name] = (w, h, fn)
        return fn
    return deco


def build(only=None):
    for name, (w, h, fn) in REGISTRY.items():
        if only and not any(o in name for o in only):
            continue
        print("render", name, render(name, w, h, fn))


# ---- small colour helpers --------------------------------------------------------------------------
def _rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def tint(color, a=0.45, base=PAPER):
    """Blend a colour towards the paper (a = share of the colour)."""
    c = _rgb(color) * a + _rgb(base) * (1 - a)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def on(color):
    """Text colour that reads on this fill (white only on the saturated blue)."""
    r, g, b = _rgb(color)
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#ffffff" if lum < 0.2 else INK


def ptext(ax, x, y, s, fill, size=12, color=None, z=6, **kw):
    """Text that sits on a filled box: its halo takes the box colour (no paper-coloured glow)."""
    return ax.text(x, y, s, fontsize=size, color=color or on(fill), zorder=z,
                   path_effects=[pe.withStroke(linewidth=2.6, foreground=fill)], **kw)
