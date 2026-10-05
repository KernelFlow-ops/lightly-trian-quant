"""Round 14 step 2: build / launch helper for the own block-scaled FP4 GEMM (fp4g_core.cu).  Style of round10/12 mg8b.py / mg16b.py:
each config (BM, BN, BKB, ST, WM, WN, MINB, FMT, EPI, PDLW) is compiled with nvcc to a cubin (cached by hash, own build dir) and launched through the generic
cudaLibrary launcher fp4l.cu on torch's current stream (CUDA-graph safe, optional PDL attribute)."""
import hashlib
import os
import struct
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
BUILD = "/root/autodl-tmp/infer-opt-tmp/round14_fp4_full/build/fp4g"
NVCC = "/usr/local/cuda/bin/nvcc"
PARAMS_FMT = "=14Q10i"      # A W SA SW ts bias gamma R C sin cos Qo SFo tso | M N K Ntok prefix d H sfo_kt pad0 pad1
_launcher = None
_KERN = {}
DEFAULT = dict(bm=128, bn=128, bkb=64, st=4, wm=2, wn=4, minb=1, fmt=0, epi=0, pdlw=0)


def launcher():
    global _launcher
    if _launcher is None:
        from torch.utils.cpp_extension import load
        os.makedirs(BUILD + "/l", exist_ok=True)
        _launcher = load(name="r14_fp4l", sources=[str(_HERE / "fp4l.cu")], build_directory=BUILD + "/l", verbose=False)
    return _launcher


def _key(cfg):
    d = dict(DEFAULT)
    d.update(cfg)
    return d


def stage_bytes(c):
    steps = c["bkb"] // 32
    tiles = steps // (1 if c["fmt"] == 0 else 2)
    sa_rb = c["bm"] // 128 if c["bm"] > 128 else 1
    sb_rb = c["bn"] // 128 if c["bn"] > 128 else 1
    return c["bm"] * c["bkb"] + c["bn"] * c["bkb"] + (sa_rb + sb_rb) * tiles * 512


def compile_cubin(cfg):
    c = _key(cfg)
    src = _HERE / "fp4g_core.cu"
    h = hashlib.md5(src.read_bytes() + repr(sorted(c.items())).encode()).hexdigest()[:12]
    os.makedirs(BUILD + "/cubin", exist_ok=True)
    out = f"{BUILD}/cubin/fp4g_{h}.cubin"
    log = f"{BUILD}/cubin/fp4g_{h}.ptxas.txt"
    if not os.path.exists(out):
        defs = [f"-D{k.upper()}={v}" for k, v in c.items()]
        r = subprocess.run([NVCC, "-arch=sm_120a", "-O3", "-std=c++17", "-cubin", "-lineinfo", "-Xptxas", "-v", "-o", out + ".tmp", str(src)] + defs,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed for %s:\n%s" % (c, r.stderr[-2500:]))
        open(log, "w").write(r.stderr)
        os.replace(out + ".tmp", out)
    return out, log, c


class Kernel:
    def __init__(self, cfg):
        path, log, c = compile_cubin(cfg)
        self.cfg, self.log = c, log
        self.threads = c["wm"] * c["wn"] * 32
        self.smem = c["st"] * stage_bytes(c)
        assert self.smem <= 99 * 1024, ("smem", self.smem, c)
        self.h = launcher().load(path, "gemm_kernel", self.smem)

    def occupancy(self):
        return launcher().occupancy(self.h, self.threads)

    def __call__(self, A, W, SA, SW, *, M, N, K, ts=None, bias=None, gamma=None, R=None, C=None, sin=None, cos=None, Ntok=1, prefix=0, d=0, H=0,
                 Qo=None, SFo=None, tso=None, pdl=False):
        c = self.cfg
        assert N % c["bn"] == 0 and K % (2 * c["bkb"]) == 0, (N, K, c)
        ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        sfo_kt = 0 if Qo is None else (N // 64 if c["fmt"] == 0 else N // 128)
        blob = struct.pack(PARAMS_FMT, ptr(A), ptr(W), ptr(SA), ptr(SW), ptr(ts), ptr(bias), ptr(gamma), ptr(R), ptr(C), ptr(sin), ptr(cos), ptr(Qo), ptr(SFo), ptr(tso),
                           M, N, K, Ntok, prefix, d, H, sfo_kt, 0, 0)
        launcher().launch(self.h, N // c["bn"], (M + c["bm"] - 1) // c["bm"], 1, self.threads, blob, pdl)


def get(cfg):
    k = tuple(sorted(_key(cfg).items()))
    if k not in _KERN:
        _KERN[k] = Kernel(cfg)
    return _KERN[k]
