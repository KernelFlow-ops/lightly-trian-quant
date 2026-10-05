"""Per-kernel static analysis (registers / spills / smem / SASS instruction mix) without ncu.

  python tests/static_analysis.py <binary(.so|.cubin)> <out.json>  [name-filter]

Uses ``cuobjdump --dump-resource-usage`` and ``cuobjdump -sass`` (the common/kernel_static.sh script needs ripgrep, which the
new container does not have, and reports whole-binary counts only).  Adds an occupancy estimate for sm_120
(65536 regs / SM, 99 KB smem / block, 1536 threads / SM max, 24 blocks / SM max).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
OPS = ["HMMA", "QMMA", "OMMA", "IMMA", "LDGSTS", "UTMALDG", "UTMASTG", "LDSM", "STS", "LDS", "LDG", "STG", "BAR", "SYNCS", "SHFL", "MUFU", "FFMA", "FADD", "FMUL", "F2F", "F2FP", "ATOM", "RED"]
SM = {"regs": 65536, "smem_kb": 100, "threads": 1536, "blocks": 24}


def run(args):
    return subprocess.run([CUOBJDUMP, *args], capture_output=True, text=True, check=False).stdout


def main():
    binary, out = sys.argv[1], sys.argv[2]
    flt = sys.argv[3] if len(sys.argv) > 3 else ""
    res = run(["--dump-resource-usage", binary])
    usage = {}
    cur = None
    for line in res.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
            continue
        if cur and "REG:" in line:
            d = dict(re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line))
            usage[cur] = {"regs": int(d.get("REG", 0)), "stack": int(d.get("STACK", 0)), "smem_static": int(d.get("SHARED", 0)), "local": int(d.get("LOCAL", 0))}
    sass = run(["-sass", binary])
    funcs = {}
    cur = None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = Counter()
            continue
        if cur is None:
            continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)(?:\.[A-Z0-9_.]+)?", line)
        if m:
            funcs[cur][m.group(1)] += 1
    rows = {}
    for name, cnt in funcs.items():
        if flt and flt not in name:
            continue
        u = usage.get(name, {})
        r = {"regs": u.get("regs"), "stack_bytes": u.get("stack"), "local_bytes": u.get("local"), "smem_static": u.get("smem_static"),
             "instr_total": sum(cnt.values()), "mix": {op: cnt.get(op, 0) for op in OPS if cnt.get(op, 0)}}
        rows[name] = r
        mix = " ".join(f"{k}={v}" for k, v in r["mix"].items())
        print(f"{name[:90]:90s} regs={r['regs']} stack={r['stack_bytes']} smem={r['smem_static']} instr={r['instr_total']} {mix}")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
