"""Static analysis of the generated attention kernels (ncu substitute): dump CUDA source, ptxas -v, SASS mix."""
import subprocess, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
import attn_tl, ops
import tilelang
OUT = HERE / "static"; OUT.mkdir(exist_ok=True)
tl_root = Path(tilelang.__file__).resolve().parent
incs = [tl_root / "src", tl_root / "3rdparty" / "cutlass" / "include"]
for kind, rope in (("plain_qkv", False), ("rope_qkv", True)):
    for (H, N) in ((16, 1029), (16, 1129)):
        cfg = ops._cfg(kind, N, H, None)
        name = f"{kind}_H{H}_N{N}"
        src = attn_tl.source(2, H, N, 64, layout="qkv", rope=rope, prefix=N - 1024, hw=1024, **cfg)
        cu = OUT / f"{name}.cu"; cu.write_text(src)
        cubin = OUT / f"{name}.cubin"
        cmd = ["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-arch=sm_120a", "-cubin", "-Xptxas", "-v", "-lineinfo",
               *sum([["-I", str(i)] for i in incs], []), "-o", str(cubin), str(cu)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (OUT / f"{name}.ptxas.txt").write_text(r.stderr)
        if r.returncode != 0:
            print(name, "nvcc FAILED", r.stderr[-800:]); continue
        sass = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", str(cubin)], capture_output=True, text=True).stdout
        (OUT / f"{name}.sass").write_text(sass)
        regs = [l.strip() for l in r.stderr.splitlines() if "registers" in l or "spill" in l or "smem" in l]
        mix = {op: sum(1 for l in sass.splitlines() if f" {op}" in l or f".{op}" in l or l.strip().split(" ")[-1:] == [op]) for op in ()}
        import re
        cnt = {}
        for l in sass.splitlines():
            m = re.search(r"\b(HMMA|QMMA|LDGSTS|LDSM|LDG|STG|LDS|STS|BAR|MUFU|SHFL|FFMA|FMUL|FADD|F2FP|EXIT)\b", l)
            if m: cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
        print(name, cfg, "\n  ", " | ".join(regs), "\n  SASS:", dict(sorted(cnt.items())), flush=True)
