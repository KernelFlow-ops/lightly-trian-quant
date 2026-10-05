"""Quick probe: QuACK FP4 autotune time, steady TFLOPS and custom-epilogue (bias / ts-scalar) feasibility."""
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4_cands as C
import fp4_ref as R
from common.bench import gpu_lock
from round13_gemm_fusion_sm120.impl.mb import time_us

dev = "cuda"
with gpu_lock():
    from quack.activation import gelu_erf
    from quack.epilogue import gemm_epilogue

    @gemm_epilogue()
    def bias_ts(acc, bias, ts):
        return {"D": acc * ts + bias}

    M, N, K = 7203, 4096, 1024
    X = torch.randn(M, K, device=dev).bfloat16()
    W = (torch.randn(N, K, device=dev) * 0.05).bfloat16()
    bias = torch.randn(1, N, device=dev)
    for fmt in ("nv4", "mx4"):
        tx, tw = R.quantize(X, fmt, col=False), R.quantize_weight(W, fmt)
        A_, B_ = (tx.q, tx.sf, tx.ts), (tw.q, tw.sf, tw.ts)
        ao, bo = C.operand(A_, fmt), C.operand(B_, fmt)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        t0 = time.time()
        C.quack_fp4(ao, bo, out)
        torch.cuda.synchronize()
        print(fmt, "plain bf16 out first call (autotune+compile) s:", round(time.time() - t0, 1), flush=True)
        r = time_us(lambda: C.quack_fp4(ao, bo, out))
        print(fmt, "plain bf16 out us", round(r["us"], 1), "TFLOPS", round(2 * M * N * K / r["us"] / 1e6, 1), r["mode"], flush=True)
        out32 = torch.empty(M, N, device=dev, dtype=torch.float32)
        t0 = time.time()
        C.quack_fp4(ao, bo, out32)
        torch.cuda.synchronize()
        print(fmt, "fp32 out first call s:", round(time.time() - t0, 1), flush=True)
        r = time_us(lambda: C.quack_fp4(ao, bo, out32))
        print(fmt, "fp32 out us", round(r["us"], 1), flush=True)
        # custom epilogue
        if fmt == "nv4":
            ts = (tx.ts * tw.ts).reshape(1)
            kw = C.epi_kw(tx.sf, tw.sf, fmt)
            t0 = time.time()
            try:
                bias_ts(tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t(), out={"D": out}, bias=bias, ts=ts, **kw)
                torch.cuda.synchronize()
                print(fmt, "epi bias_ts first call s:", round(time.time() - t0, 1), flush=True)
                r = time_us(lambda: bias_ts(tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t(), out={"D": out}, bias=bias, ts=ts, **kw))
                print(fmt, "epi bias_ts us", round(r["us"], 1), r["mode"])
                ref = (R.dequant_row(tx) @ R.dequant_row(tw).t()) + bias
                print("epi relL2", float((out.float() - ref).norm() / ref.norm()))
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(fmt, "epi error", type(e).__name__, str(e)[:500])
        else:
            E = C.epilogues()
            kw = C.epi_kw(tx.sf, tw.sf, fmt)
            t0 = time.time()
            try:
                E["bias"](tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t(), out={"D": out}, bias=bias, **kw)
                torch.cuda.synchronize()
                print(fmt, "r13 bias epi first call s:", round(time.time() - t0, 1), flush=True)
                r = time_us(lambda: E["bias"](tx.q.view(torch.float4_e2m1fn_x2), tw.q.view(torch.float4_e2m1fn_x2).t(), out={"D": out}, bias=bias, **kw))
                print(fmt, "r13 bias epi us", round(r["us"], 1))
            except Exception as e:
                print(fmt, "epi error", type(e).__name__, str(e)[:500])
