"""Round 13: build / launch helper + wrappers of the own MXFP8 block-scaled GEMM `mgx` (mgx_core.cu).

Build: each config is compiled with nvcc to a cubin (cached by hash) and loaded through the generic cubin launcher mgxl.cu (cudaLibrary + cudaLaunchKernelExC
on torch's current stream: CUDA-Graph safe) -- the same machinery as round10 mg8b.py / round12 mg16b.py (read-only sources, copied).
Operands: `mx13.ActBuf` (e4m3 bytes [M,K] + blocked scales) and `mx13.MXW` (weights).  Tile configs: tuned_mx.json (written by tune_mx.py), else DEFAULT.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import torch

import mx13

_HERE = Path(__file__).resolve().parent
BUILD = str(mx13.TMP / "build_mgx")
NVCC = "/usr/local/cuda/bin/nvcc"
F16, BF = torch.float16, torch.bfloat16
PARAMS_FMT = "=12Q8i"          # A W SFA SFB bias gamma R C sin cos Cq Csf | M N K Ntok prefix d H pdl
PDL = bool(int(os.environ.get("MGX_PDL", "1")))   # programmatic dependent launch: weights are prefetched before griddepcontrol.wait
_launcher = None
_KERN: dict = {}
EPI = {"fc1": 0, "proj": 1, "fc2": 1, "qkv": 2, "plain": 4, "ct1": 4, "ct2": 4}

DEFAULT = {
    "fc1": dict(epi=0, bm=128, bn=128, bkb=64, st=3, wm=2, wn=2, minb=2),
    "fc2": dict(epi=1, bm=128, bn=64, bkb=64, st=3, wm=2, wn=2, minb=2),
    "proj": dict(epi=1, bm=128, bn=64, bkb=64, st=3, wm=2, wn=2, minb=2),
    "qkv": dict(epi=2, bm=128, bn=128, bkb=64, st=3, wm=2, wn=4, minb=2, stg=1),
    "plain": dict(epi=4, bm=128, bn=128, bkb=64, st=3, wm=2, wn=4, minb=2),
}
DEFAULT["ct1"] = DEFAULT["ct2"] = DEFAULT["plain"]
_T = _HERE / os.environ.get("MGX_TUNED", "tuned_mx.json")
TUNED: dict = json.loads(_T.read_text()) if _T.exists() else {}


def launcher():
    global _launcher
    if _launcher is None:
        from torch.utils.cpp_extension import load
        os.makedirs(BUILD + "/mgl", exist_ok=True)
        _launcher = load(name="mgxl", sources=[str(_HERE / "mgxl.cu")], build_directory=BUILD + "/mgl", verbose=False)
    return _launcher


def _key(cfg):
    d = dict(bm=128, bn=128, bkb=64, st=3, wm=2, wn=4, minb=2, epi=1, order=0, unr=1, nfull=1, stg=0)
    d.update(cfg)
    return d


def nat_of(b):
    return b // 128 if b % 128 == 0 else (1 if 128 % b == 0 else (b + 112 + 127) // 128)


def smem_bytes(c):
    kpw = 128 // c["bkb"]
    sslots = (c["st"] - 1 + kpw - 1) // kpw + 1
    return c["st"] * (c["bm"] + c["bn"]) * c["bkb"] + sslots * (nat_of(c["bm"]) + nat_of(c["bn"])) * 512


def compile_cubin(cfg):
    c = _key(cfg)
    src = _HERE / "mgx_core.cu"
    h = hashlib.md5(src.read_bytes() + repr(sorted(c.items())).encode()).hexdigest()[:12]
    os.makedirs(BUILD + "/cubin", exist_ok=True)
    out = f"{BUILD}/cubin/mgx_{h}.cubin"
    log = f"{BUILD}/cubin/mgx_{h}.ptxas.txt"
    if not os.path.exists(out):
        defs = [f"-D{k.upper()}={v}" for k, v in c.items()]
        r = subprocess.run([NVCC, "-arch=sm_120a", "-O3", "-std=c++17", "-cubin", "-lineinfo", "-Xptxas", "-v", "-o", out + f".{os.getpid()}.tmp", str(src)] + defs,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed for %s:\n%s" % (c, r.stderr[-2500:]))
        open(log, "w").write(r.stderr)
        os.replace(out + f".{os.getpid()}.tmp", out)
    return out, log, c


class Kernel:
    def __init__(self, cfg):
        path, log, c = compile_cubin(cfg)
        self.cfg, self.log = c, log
        self.threads = c["wm"] * c["wn"] * 32
        self.smem = smem_bytes(c)
        self.h = launcher().load(path, "gemm_kernel", self.smem)

    def occupancy(self):
        return launcher().occupancy(self.h, self.threads)

    def __call__(self, A, W, M, N, K, *, bias=None, gamma=None, R=None, C=None, sin=None, cos=None, Cq=None, Csf=None, Ntok=0, prefix=0, d=0, H=0, pdl=None):
        pdl = PDL if pdl is None else pdl
        c = self.cfg
        ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        blob = struct.pack(PARAMS_FMT, ptr(A.q), ptr(W.q), ptr(A.sf), ptr(W.sf), ptr(bias), ptr(gamma), ptr(R), ptr(C), ptr(sin), ptr(cos), ptr(Cq), ptr(Csf),
                           M, N, K, Ntok, prefix, d, H, 1 if pdl else 0)
        launcher().launch(self.h, (N + c["bn"] - 1) // c["bn"], (M + c["bm"] - 1) // c["bm"], 1, self.threads, blob, pdl)


def get(cfg):
    k = tuple(sorted(_key(cfg).items()))
    if k not in _KERN:
        _KERN[k] = Kernel(cfg)
    return _KERN[k]


def pick(kind, N, K, M):
    key = f"{kind}|{N}|{K}|{M}"
    if key in TUNED:
        return TUNED[key]["cfg"]
    best = None
    for k, v in TUNED.items():
        kk, nn, kd, mm = k.split("|")
        if kk == kind and int(nn) == N and int(kd) == K:
            dm = abs(int(mm) - M)
            if best is None or dm < best[0]:
                best = (dm, v["cfg"])
    return best[1] if best else DEFAULT[kind]


def _chk(a, W):
    assert a.K == W.K and W.N % 128 == 0 and a.K % 128 == 0, (a.K, W.K, W.N)


def qkv_rope(a, W, bias, sin, cos, Ntok, prefix, H, cfg=None):
    """a: ActBuf [M,d] -> fp16 [3,B,Ntok,H,64] (bias + RoPE fp32 in the epilogue).  Returns the PHYSICAL tensor (what attn16_fwd takes)."""
    _chk(a, W)
    M, K, N = a.M, a.K, W.N
    cfg = dict(cfg if cfg is not None else pick("qkv", N, K, M))
    out = torch.empty(3, M // Ntok, Ntok, H, 64, device=a.q.device, dtype=F16)
    get(cfg)(a, W, M, N, K, bias=bias, C=out, sin=sin, cos=cos, Ntok=Ntok, prefix=prefix, d=K, H=H)
    return out


def gemm_res(a, W, bias, gamma, res, cfg=None, kind=None):
    """out_f32 = res + gamma * (a @ W^T + bias)   (proj / fc2)"""
    _chk(a, W)
    M, K, N = a.M, a.K, W.N
    kind = kind or ("fc2" if K > N else "proj")
    cfg = dict(cfg if cfg is not None else pick(kind, N, K, M))
    out = torch.empty_like(res)
    get(cfg)(a, W, M, N, K, bias=bias, gamma=gamma, R=res, C=out)
    return out


def gemm_gelu_q(a, W, bias, out, cfg=None):
    """out (ActBuf [M,N]) = MX8(GELU(a @ W^T + bias))   (fc1 -> fc2's operand)"""
    _chk(a, W)
    M, K, N = a.M, a.K, W.N
    cfg = dict(cfg if cfg is not None else pick("fc1", N, K, M))
    get(cfg)(a, W, M, N, K, bias=bias, Cq=out.q, Csf=out.sf)
    return out


def gemm_plain(a, W, cfg=None, kind="plain"):
    """bf16 [M,N] = a @ W^T"""
    _chk(a, W)
    M, K, N = a.M, a.K, W.N
    cfg = dict(cfg if cfg is not None else pick(kind, N, K, M))
    out = torch.empty(M, N, device=a.q.device, dtype=BF)
    get(cfg)(a, W, M, N, K, C=out)
    return out
