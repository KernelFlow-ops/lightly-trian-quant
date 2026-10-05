"""Smoke test: custom QuACK epilogues (fn frontend) on sm_120 for the fused transformer block."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
sys.path.insert(0, "/root/lightly-train/train-opt")
sys.path.insert(0, "/root/lightly-train/train-opt/round13_gemm_fusion_sm120/impl")

import torch
import torch.nn.functional as F

from common.bench import gpu_lock
from mb import err_stats, time_us
from quack.activation import dgelu_erf, gelu_erf
from quack.epilogue import gemm_epilogue
from quack.epilogue.ops import RowVecReduce


@gemm_epilogue(outputs=("y",))
def proj_res(acc, c, bias, gamma):
    y = acc + bias
    return {"D": c + gamma * y, "y": y}


@gemm_epilogue(outputs=("h",))
def fc1_gelu(acc, bias):
    z = acc + bias
    return {"D": z, "h": gelu_erf(z)}


@gemm_epilogue(reduces={"db": RowVecReduce("db")})
def fc2_dgrad(acc, c):
    dz, _ = dgelu_erf(c, acc)
    return {"D": dz, "db": dz}


def main():
    d, M = 1024, 7203
    torch.manual_seed(0)
    dev = "cuda"
    with gpu_lock():
        # ---- proj: x1 = x + gamma * (A @ Wp^T + b), y = A @ Wp^T + b
        A = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
        W = torch.randn(d, d, device=dev, dtype=torch.bfloat16) * 0.05
        b = torch.randn(1, d, device=dev, dtype=torch.float32)
        g = torch.rand(1, d, device=dev, dtype=torch.float32)
        x = torch.randn(M, d, device=dev, dtype=torch.float32)
        D = torch.empty(M, d, device=dev, dtype=torch.float32)
        Y = torch.empty(M, d, device=dev, dtype=torch.bfloat16)
        t0 = time.time()
        res = proj_res(A, W.t(), x, out={"D": D, "y": Y}, bias=b, gamma=g)
        torch.cuda.synchronize()
        print("proj_res first call", time.time() - t0, type(res), list(res.keys()) if isinstance(res, dict) else None)
        yref = A.float() @ W.float().t() + b
        print("proj y", err_stats(Y, yref))
        print("proj D", err_stats(D, x + g * yref))
        tm = time_us(lambda: proj_res(A, W.t(), x, out={"D": D, "y": Y}, bias=b, gamma=g))
        print("proj_res us", tm, "TF", 2 * M * d * d / tm["us"] / 1e6)
        from quack.gemm_interface import gemm
        tm = time_us(lambda: gemm(A, W.t(), out=Y))
        print("plain quack proj us", tm["us"])
        tm = time_us(lambda: torch.mm(A, W.t(), out=Y))
        print("torch proj us", tm["us"])

        # ---- fc1: z = A @ W1^T + b ; h = gelu(z)
        N = 4 * d
        W1 = torch.randn(N, d, device=dev, dtype=torch.bfloat16) * 0.05
        b1 = torch.randn(1, N, device=dev, dtype=torch.float32)
        Z = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        H = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        t0 = time.time()
        fc1_gelu(A, W1.t(), out={"D": Z, "h": H}, bias=b1)
        torch.cuda.synchronize()
        print("fc1 first call", time.time() - t0)
        zref = A.float() @ W1.float().t() + b1
        print("fc1 z", err_stats(Z, zref))
        print("fc1 h", err_stats(H, F.gelu(zref)))
        print("fc1 h vs gelu(bf16 z)", err_stats(H, F.gelu(Z.float())))
        tm = time_us(lambda: fc1_gelu(A, W1.t(), out={"D": Z, "h": H}, bias=b1))
        print("fc1_gelu us", tm["us"], "TF", 2 * M * N * d / tm["us"] / 1e6)
        tm = time_us(lambda: gemm(A, W1.t(), out=Z))
        print("plain quack fc1 us", tm["us"])
        tm = time_us(lambda: torch.mm(A, W1.t(), out=Z))
        print("torch fc1 us", tm["us"])

        # ---- fc2 dgrad: dz = (dy @ W2) * gelu'(z) ; db1 = sum_m dz
        dy = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
        W2 = torch.randn(d, N, device=dev, dtype=torch.bfloat16) * 0.05  # fc2 weight [d, 4d]
        Zt = (torch.randn(M, N, device=dev) * 2).to(torch.bfloat16)
        DZ = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        t0 = time.time()
        res = fc2_dgrad(dy, W2, Zt, out={"D": DZ})
        torch.cuda.synchronize()
        print("fc2_dgrad first call", time.time() - t0, list(res.keys()))
        zf = Zt.float().requires_grad_()
        dh = dy.float() @ W2.float()
        gref = torch.autograd.grad(F.gelu(zf), zf, dh)[0]
        print("dz", err_stats(DZ, gref))
        print("db", err_stats(res["db"].reshape(-1), gref.sum(0)))
        tm = time_us(lambda: fc2_dgrad(dy, W2, Zt, out={"D": DZ}))
        print("fc2_dgrad us", tm["us"], "TF", 2 * M * N * d / tm["us"] / 1e6)
        tm = time_us(lambda: gemm(dy, W2, out=DZ))
        print("plain quack fc2-dgrad us", tm["us"])
        tm = time_us(lambda: torch.mm(dy, W2, out=DZ))
        print("torch fc2-dgrad us", tm["us"])


if __name__ == "__main__":
    main()
