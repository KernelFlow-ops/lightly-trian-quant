"""Regression: captured split-K kernels must survive workspace cache growth."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))
from bench import gpu_lock  # noqa: E402
import mgb  # noqa: E402


def main():
    with gpu_lock(), torch.inference_mode():
        M, N, K = 2058, 256, 1024
        cfg = dict(epi=1, rf32=1, bm=64, bn=64, bk=32, st=2,
                   wm=2, wn=2, minb=3, splitk=2)
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        gamma = torch.rand(N, device="cuda")
        residual = torch.randn(M, N, device="cuda")
        out = torch.empty_like(residual)
        kernel = mgb.get(cfg)
        kernel(a, w, bias, gamma, residual, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(a, w, bias, gamma, residual, out)

        old_pool = mgb.workspace(1, a.device)
        old_ptr, old_size = old_pool[0].data_ptr(), old_pool[0].numel()
        del old_pool
        mgb.workspace(old_size + 1, a.device)
        assert any(pool[0].data_ptr() == old_ptr for pool in mgb._WS[str(a.device)])
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        reference = residual + gamma * F.linear(a.float(), w.float(), bias.float())
        max_error = (out - reference).abs().max().item()
        assert max_error < 0.05, max_error
        assert all(bool((counters == 0).all()) for pools in mgb._WS.values() for _, counters in pools)
        print(f"captured split-K survived workspace growth; max abs error {max_error:.6f}")


if __name__ == "__main__":
    main()
