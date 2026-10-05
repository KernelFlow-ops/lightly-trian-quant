"""Round 10: copy of round5_gemm_sm120/impl/mgb.py for the fp16-accumulate prototype (mg8_core.cu); own build dir / extension name.

Original doc: Build/launch helper for the hand-written CUDA GEMM (mg_core.cu).

Each config (BM,BN,BK,ST,WM,WN,MINB,EPI,RF32,SPLITK,PDL_TRIG) is compiled with nvcc to a cubin (no torch headers, ~10 s,
cached by hash) and loaded through a tiny torch-extension launcher (mgl.cu) that uses cudaLibrary + cudaLaunchKernelExC
on torch's current stream (CUDA-Graph safe, optional PDL attribute)."""
import hashlib, os, struct, subprocess, sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
BUILD = "/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build/mg8"
NVCC = "/usr/local/cuda/bin/nvcc"
_launcher = None
_KERN = {}
_WS = {}
PARAMS_FMT = "=14Q10i2f2Q2i"  # A W bias gamma R C sin cos ws counters cs stats_in stats_out Cb | M N K Ntok prefix d H pdl np_in pcols_in | eps | pf0 pf1 (ptr) | pf0_bytes pf1_bytes


def launcher():
    global _launcher
    if _launcher is None:
        from torch.utils.cpp_extension import load
        os.makedirs(BUILD + "/mgl", exist_ok=True)
        _launcher = load(name="mg8l", sources=[str(_HERE / "mg16l.cu")], build_directory=BUILD + "/mgl", verbose=False)
    return _launcher


KEYS = ("bm", "bn", "bk", "st", "wm", "wn", "minb", "epi", "rf32", "trig", "splitk", "extra")


def _key(cfg):
    d = dict(bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, epi=0, rf32=1, trig=1, splitk=1, lnf=0, stats=0, occ=0, kdim=0, f16acc=0, outf16=0, pkt=1, extra="")
    d.update(cfg)
    return d


def compile_cubin(cfg):
    """nvcc -> cubin (cached). returns (cubin path, ptxas log path)."""
    c = _key(cfg)
    src = _HERE / "mg8_core.cu"
    h = hashlib.md5(src.read_bytes() + repr(sorted(c.items())).encode()).hexdigest()[:12]
    os.makedirs(BUILD + "/cubin", exist_ok=True)
    out = f"{BUILD}/cubin/mg8_{h}.cubin"
    log = f"{BUILD}/cubin/mg8_{h}.ptxas.txt"
    if not os.path.exists(out):
        defs = [f"-D{k.upper()}={v}" for k, v in c.items() if k not in ("extra", "occ", "kdim")]
        if c["kdim"]:
            defs.append(f"-DKDIM={c['kdim']}")
        defs = [x.replace("-DTRIG=", "-DPDL_TRIG=") for x in defs]
        extra = c["extra"].split() if c["extra"] else []
        r = subprocess.run([NVCC, "-arch=sm_120a", "-O3", "-std=c++17", "-cubin", "-lineinfo", "-Xptxas", "-v", "-o", out + ".tmp",
                            str(src)] + defs + extra, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed for %s:\n%s" % (c, r.stderr[-1500:]))
        open(log, "w").write(r.stderr)
        os.replace(out + ".tmp", out)
    return out, log, c


class Kernel:
    def __init__(self, cfg):
        path, log, c = compile_cubin(cfg)
        self.cfg = c
        self.log = log
        self.threads = c["wm"] * c["wn"] * 32
        self.smem = c["st"] * (c["bm"] + c["bn"]) * c["bk"] * 2   # fp16 operands: same 2 bytes/elt
        if c["occ"] >= 1:  # pad dynamic smem so that at most `occ` CTAs fit per SM (occupancy knob)
            self.smem = min(max(self.smem, (100 * 1024) // c["occ"] - 1024 - 1024), 98 * 1024)
        self.h = launcher().load(path, "gemm_kernel", self.smem)

    def occupancy(self):
        return launcher().occupancy(self.h, self.threads)

    def __call__(self, A, W, bias, gamma=None, R=None, C=None, sin=None, cos=None, Ntok=0, prefix=0, d=0, H=0, pdl=False,
                 cs=None, stats_in=None, stats_out=None, Cb=None, np_in=0, pcols_in=0, eps=1e-5, prefetch=(), oscale=1.0):
        c = self.cfg
        M, K = A.shape
        N = W.shape[0]
        ws = cnt = 0
        S = c["splitk"]
        tiles = ((M + c["bm"] - 1) // c["bm"]) * ((N + c["bn"] - 1) // c["bn"])
        if S > 1 and c["epi"] != 3:
            wsb, cntb = workspace(tiles * S * c["bm"] * c["bn"], A.device)
            ws, cnt = wsb.data_ptr(), cntb.data_ptr()
        ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        pf = [(t.data_ptr(), (t.numel() * t.element_size()) // 128 * 128) for t in prefetch[:2]]  # L2 prefetch regions
        pf += [(0, 0)] * (2 - len(pf))
        blob = struct.pack(PARAMS_FMT, ptr(A), ptr(W), ptr(bias), ptr(gamma), ptr(R), ptr(C), ptr(sin), ptr(cos), ws, cnt,
                           ptr(cs), ptr(stats_in), ptr(stats_out), ptr(Cb),
                           M, N, K, Ntok, prefix, d, H, 1 if pdl else 0, np_in, pcols_in, eps, oscale,
                           pf[0][0], pf[1][0], pf[0][1], pf[1][1])
        launcher().launch(self.h, (N + c["bn"] - 1) // c["bn"], (M + c["bm"] - 1) // c["bm"], S, self.threads, blob, pdl)


def get(cfg):
    k = tuple(sorted(_key(cfg).items()))
    if k not in _KERN:
        _KERN[k] = Kernel(cfg)
    return _KERN[k]


def workspace(nfloat, dev="cuda"):
    """Split-K workspace and counters, retained for the lifetime of captured graphs."""
    pools = _WS.setdefault(str(dev), [])
    for w in pools:
        if w[0].numel() >= nfloat:
            return w
    w = (torch.empty(max(nfloat, 1 << 22), device=dev, dtype=torch.float32), torch.zeros(8192, device=dev, dtype=torch.int32))
    pools.append(w)
    return w


