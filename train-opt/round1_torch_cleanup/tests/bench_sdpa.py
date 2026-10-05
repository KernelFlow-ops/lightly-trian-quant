"""Locked fwd/bwd SDPA backend comparison at the fixed training batch."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from common.bench import _stats, assert_gpu_exclusive, env_info, gpu_lock  # noqa: E402
from common.config import B_AUTO, MODEL_SHAPES  # noqa: E402

BACKENDS = {
    "default": None,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def _one(q, k, v, mask, backend):
    context = sdpa_kernel(backends=[backend]) if backend is not None else nullcontext()
    with context:
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        q.grad = k.grad = v.grad = None
        fwd_start.record()
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        fwd_end.record()
        output.sum().backward()
        bwd_end.record()
    return fwd_start, fwd_end, bwd_end


def measure(q, k, v, mask, backend, *, warmup=20, iters=50, repeats=3):
    for _ in range(warmup):
        _one(q, k, v, mask, backend)
    torch.cuda.synchronize()
    fwd, bwd, combined = [], [], []
    for _ in range(repeats):
        events = [_one(q, k, v, mask, backend) for _ in range(iters)]
        torch.cuda.synchronize()
        for start, middle, end in events:
            fw = start.elapsed_time(middle) * 1000
            bw = middle.elapsed_time(end) * 1000
            fwd.append(fw)
            bwd.append(bw)
            combined.append(fw + bw)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        _one(q, k, v, mask, backend)
        torch.cuda.synchronize()
    selected = [item.key for item in profiler.key_averages() if "scaled_dot_product" in item.key or "flash_attention" in item.key]
    return {"fwd_us": _stats(fwd), "bwd_us": _stats(bwd), "fwd_bwd_us": _stats(combined), "selected_ops": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--phase", choices=("plain", "joint_mask", "joint_nomask"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    batch = args.batch_size or B_AUTO[args.model]
    if batch is None:
        raise RuntimeError("B_AUTO is unset")
    shape = MODEL_SHAPES[args.model]
    heads = shape["heads"]
    tokens = 1029 if args.phase == "plain" else 1129
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    with gpu_lock():
        before = assert_gpu_exclusive()
        q, k, v = [torch.randn(batch, heads, tokens, shape["head_dim"], device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        mask = None
        if args.phase == "joint_mask":
            mask = torch.ones(batch, 1, tokens, tokens, device="cuda", dtype=torch.bool)
            mask[:, :, :100, 105:] = torch.rand(batch, 1, 100, tokens - 105, device="cuda") > 0.5
        results = {}
        for name, backend in BACKENDS.items():
            try:
                results[name] = measure(q, k, v, mask, backend)
            except RuntimeError as exc:
                results[name] = {"error": str(exc), "fwd_us": None, "bwd_us": None, "fwd_bwd_us": None}
        after = assert_gpu_exclusive()
        environment = env_info()
    output = Path(args.out) if args.out else ROOT / "round1_torch_cleanup" / "profile" / f"sdpa_{args.model}_{args.phase}_{args.dtype}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({"command": [sys.executable, *sys.argv], "model": args.model, "batch": batch, "phase": args.phase, "dtype": args.dtype, "warmup": 20, "iters": 50, "repeats": 3, "results": results, "env": environment, "gpu_snapshot_before": before, "gpu_snapshot_after": after}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
