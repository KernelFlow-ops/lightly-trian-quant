"""Round 10 GEMM timing harness: CUDA-graph replay, candidates interleaved in one process, median of 3 rounds.

Every candidate is a `Cand(name, make, flops_scale=1, epilogue=...)` where `make(shape)` returns a list of R zero-arg callables
(R independent weight copies, activations shared); the R calls are captured back-to-back in one graph, so each kernel sees a
cold-ish weight (R * weight bytes >= ~192 MB > the 96 MB L2) and activations that are L2-hot, like in the model.  Reported time = graph time / R.

Call under the dual lock (round10_lowp_probe/run_locked.sh or lock2.gpu_lock2).
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

ROOT = Path("/root/lightly-train/infer-opt")
R10 = ROOT / "round10_lowp_probe"
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe")
for p in (ROOT / "common", R10 / "impl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lock2 import other_gpu_procs  # noqa: E402

# (model) -> d, ff, heads, layers
MODELS = {"vitl16": dict(d=1024, ff=4096, H=16, L=24), "vitb16": dict(d=768, ff=3072, H=12, L=12)}
KINDS = ("qkv", "proj", "fc1", "fc2")


EXTRA_KINDS = ("patch", "ct1", "ct2", "einsum")      # patch-embed GEMM, head ConvT1 / ConvT2 GEMMs, mask einsum (per crop, M = 100 queries)
EXTRA_M = {"patch": 2048, "ct1": 2048, "ct2": 8192, "einsum": 100}


def kind_nk(model: str, kind: str) -> tuple[int, int]:
    """(N, K) of the GEMM  out[M,N] = X[M,K] W[N,K]^T."""
    d, ff = MODELS[model]["d"], MODELS[model]["ff"]
    return {"qkv": (3 * d, d), "proj": (d, d), "fc1": (ff, d), "fc2": (d, ff),
            "patch": (d, 768), "ct1": (4 * d, d), "ct2": (4 * d, d), "einsum": (16384, d)}[kind]


def n_calls_per_frame(model: str, kind: str, M: int) -> int:
    """number of such GEMMs per frame at this M (2058: normal layers, 2258: the 3 joint layers; 4116/4516 = the 2-frame batch)."""
    L = MODELS[model]["L"]
    if kind in EXTRA_KINDS:
        return {"patch": 1, "ct1": 1, "ct2": 1, "einsum": 2}[kind]
    return L - 3 if M in (2058, 4116) else 3


@dataclass
class Cand:
    name: str
    make: Callable[[dict], list]          # shape dict(model, kind, M, N, K) -> list of R closures, or None if unsupported
    note: str = ""                        # what the timed region contains (epilogue / quantization)
    repeat: int = 1                       # replicate the closure list this many times inside the graph (single-kernel candidates: amortise the graph-launch overhead)
    results: dict = field(default_factory=dict)


def _weight_copies(nbytes: int, target: int = 192 << 20, lo: int = 4, hi: int = 96) -> int:
    return max(lo, min(hi, -(-target // max(nbytes, 1))))


R_FOR = _weight_copies


def _capture(fns):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:3]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    return g


def _replay_us(g, nfn: int, reps: int) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(max(2, reps // 4)):
        g.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e3 / (reps * nfn)


def time_shape(shape: dict, cands: list[Cand], rounds: int = 3, target_ms: float = 40.0, verbose: bool = True, retries: int = 2) -> dict:
    """time_shape_once, repeated (up to `retries` times) while another process is seen on the GPU before/after the measurement
    (correctness tests of other workers may run without the lock); the result records the foreign processes of the accepted attempt."""
    res = {}
    for attempt in range(retries + 1):
        before = other_gpu_procs()
        res = time_shape_once(shape, cands, rounds, target_ms, verbose)
        after = other_gpu_procs()
        foreign = sorted(set(before) | set(after))
        for v in res.values():
            v["foreign_gpu_procs"] = foreign
        if not foreign:
            break
        if verbose:
            print(f"   foreign GPU processes {foreign} during {shape['kind']} M={shape['M']}; attempt {attempt + 1}", flush=True)
    return res


def time_shape_once(shape: dict, cands: list[Cand], rounds: int = 3, target_ms: float = 40.0, verbose: bool = True) -> dict:
    """Interleaved graph timing of all candidates on one shape. Returns {name: {us, tflops, rounds_us}}; unsupported -> {error}."""
    built = {}
    out = {}
    for c in cands:
        try:
            fns = c.make(shape)
            if fns is None:
                out[c.name] = {"error": "unsupported"}
                continue
            fns = list(fns) * c.repeat
            g = _capture(fns)
            # pick reps so one measurement is ~target_ms
            t1 = _replay_us(g, len(fns), 4) * len(fns) / 1e3  # ms per replay
            reps = max(2, int(target_ms / max(t1, 0.05)))
            built[c.name] = (g, len(fns), reps, fns)
        except Exception as ex:  # noqa: BLE001
            out[c.name] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
            if verbose:
                print(f"   [{c.name}] FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    samples = {n: [] for n in built}
    for _ in range(rounds):
        for n, (g, nfn, reps, _fns) in built.items():
            samples[n].append(_replay_us(g, nfn, reps))
    fl = 2.0 * shape["M"] * shape["N"] * shape["K"]
    for n, v in samples.items():
        us = statistics.median(v)
        out[n] = {"us": us, "tflops": fl / (us * 1e-6) / 1e12, "rounds_us": v}
    return out


def merge_results(path: Path, payload: dict) -> None:
    """Deep-merge `payload` into the JSON file (read-modify-write; only one writer per file)."""
    data = {}
    if path.exists():
        data = json.loads(path.read_text())

    def upd(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                upd(a[k], v)
            else:
                a[k] = v

    upd(data, payload)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def gpu_state() -> dict:
    return {"other_gpu_procs": other_gpu_procs()}


# ------------------------------------------------------------------------------------------------------------------ shapes
def gemm_shapes(Ms=(2058, 2258)) -> list[dict]:
    out = []
    for model in MODELS:
        for kind in KINDS:
            N, K = kind_nk(model, kind)
            for M in Ms:
                out.append(dict(model=model, kind=kind, M=M, N=N, K=K))
    return out


def shape_key(s: dict) -> str:
    return f"{s['model']}|{s['kind']}|{s['M']}"


def extra_shapes() -> list[dict]:
    out = []
    for model in MODELS:
        for kind in EXTRA_KINDS:
            N, K = kind_nk(model, kind)
            out.append(dict(model=model, kind=kind, M=EXTRA_M[kind], N=N, K=K))
    return out
