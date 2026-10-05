"""FP4 policy strings (CONTRACTS 11.4 + 6a 12.3): per GEMM kind format selection for the round20 simulated block / head.

Grammar (tokens separated by ``,`` or ``+``)::

    <lin>.<kinds>=<fmt>         lin   : qkv proj fc1 fc2 | fc (=fc1+fc2) | attn (=qkv+proj) | all (=qkv proj fc1 fc2) | head
                                kinds : any non-empty subset of f d w  (f = fwd, d = dgrad, w = wgrad)
                                fmt   : bf16 mx8 nv4 mx4
    entry options (apply to the entry that precedes them)
        sr | rn     rounding of the GRADIENT operand dY of the entry's d / w kinds (FP4 only).  Default = sr (CONTRACTS 11.3).
        rht         16-point RHT along the contraction axis of the entry's kinds (w: tokens / pixels, f and d: K / N).
        w2d | w1d   weight block-scale grid of the entry's f / d kinds (FP4 only).  Default w2d.
    global options
        nofirst nolast nojoint   blocks 0 / last / last three revert to the FINAL formats (no FP4, no change at all)
        prefix                   wgrad FP4 contraction only over the patch tokens (last 1024 per image); prefix tokens exact (bf16 side GEMM)
        amax=exact|delayed       nv4 per-tensor amax: current tensor / 2 x the amax of the previous call at the same call site (first call exact)

Unspecified kinds keep FINAL: qkv f=bf16 d=mx8 w=mx8; proj fc1 fc2 mx8; head bf16.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FMTS = ("bf16", "mx8", "nv4", "mx4")
FP4 = ("nv4", "mx4")
LINS = ("qkv", "proj", "fc1", "fc2")
ALIASES = {"fc": ("fc1", "fc2"), "attn": ("qkv", "proj"), "all": LINS, "head": ("head",), **{k: (k,) for k in LINS}}
ENTRY_OPTS = ("sr", "rn", "rht", "w2d", "w1d")
GLOBAL_OPTS = ("nofirst", "nolast", "nojoint", "prefix")
FINAL_DEFAULT = {"qkv": {"f": "bf16", "d": "mx8", "w": "mx8"}, "proj": {"f": "mx8", "d": "mx8", "w": "mx8"},
                 "fc1": {"f": "mx8", "d": "mx8", "w": "mx8"}, "fc2": {"f": "mx8", "d": "mx8", "w": "mx8"},
                 "head": {"f": "bf16", "d": "bf16", "w": "bf16"}}


@dataclass
class Entry:
    lins: tuple
    kinds: str
    fmt: str
    opts: set = field(default_factory=set)


@dataclass(frozen=True)
class LinSpec:
    """Resolved per-layer spec: formats of the three GEMMs plus the FP4 options."""
    name: str
    ff: str = "mx8"
    fd: str = "mx8"
    fw: str = "mx8"
    sr_d: bool = False
    sr_w: bool = False
    rht_f: bool = False
    rht_d: bool = False
    rht_w: bool = False
    w2d_f: bool = True
    w2d_d: bool = True
    prefix: bool = False

    @property
    def fmts(self):
        return (self.ff, self.fd, self.fw)

    @property
    def any_fp4(self) -> bool:
        return any(f in FP4 for f in self.fmts)

    @property
    def is_final(self) -> bool:
        return all(self.fmts[i] == FINAL_DEFAULT[self.name]["fdw"[i]] for i in range(3)) if self.name in FINAL_DEFAULT else False


@dataclass
class Policy:
    text: str = ""
    entries: list = field(default_factory=list)
    nofirst: bool = False
    nolast: bool = False
    nojoint: bool = False
    prefix: bool = False
    amax: str = "delayed"

    def excluded(self, i: int, nblocks: int) -> bool:
        return (self.nofirst and i == 0) or (self.nolast and i == nblocks - 1) or (self.nojoint and i >= nblocks - 3)

    def spec(self, name: str, i: int = 0, nblocks: int = 24) -> LinSpec:
        """Resolved LinSpec of linear ``name`` ('qkv'...'head') in block ``i`` (head: i ignored)."""
        fm = dict(FINAL_DEFAULT[name])
        if name != "head" and self.excluded(i, nblocks):
            return LinSpec(name, fm["f"], fm["d"], fm["w"])
        sr = {"d": True, "w": True}
        rht = {"f": False, "d": False, "w": False}
        w2d = {"f": True, "d": True}
        for e in self.entries:
            if name not in e.lins:
                continue
            for k in e.kinds:
                fm[k] = e.fmt
                if k in sr:
                    sr[k] = "rn" not in e.opts  # default (neither option) = SR for the gradient operand (11.3)
                rht[k] = "rht" in e.opts
                if k in w2d:
                    w2d[k] = "w1d" not in e.opts
        return LinSpec(name, fm["f"], fm["d"], fm["w"], sr_d=sr["d"] and fm["d"] in FP4, sr_w=sr["w"] and fm["w"] in FP4,
                       rht_f=rht["f"] and fm["f"] in FP4, rht_d=rht["d"] and fm["d"] in FP4, rht_w=rht["w"] and fm["w"] in FP4,
                       w2d_f=w2d["f"], w2d_d=w2d["d"], prefix=self.prefix and fm["w"] in FP4 and name != "head")

    def describe(self) -> str:
        return self.text or "FINAL"


def parse(text: str) -> Policy:
    pol = Policy(text=text.strip())
    cur = None
    for tok in [t.strip() for t in re.split(r"[,+]", text) if t.strip()]:
        if tok.startswith("amax="):
            assert tok[5:] in ("exact", "delayed"), tok
            pol.amax = tok[5:]
        elif "=" in tok:
            target, _, fmt = tok.partition("=")
            lin, _, kinds = target.partition(".")
            if lin not in ALIASES or not kinds or not set(kinds) <= set("fdw") or fmt not in FMTS:
                raise ValueError(f"bad policy entry {tok!r}")
            cur = Entry(ALIASES[lin], "".join(k for k in "fdw" if k in kinds), fmt)
            pol.entries.append(cur)
        elif tok in GLOBAL_OPTS:
            setattr(pol, tok, True)
        elif tok in ENTRY_OPTS:
            if cur is None:
                raise ValueError(f"option {tok!r} before any entry in {text!r}")
            cur.opts.add(tok)
        else:
            raise ValueError(f"unknown policy token {tok!r} in {text!r}")
    return pol
