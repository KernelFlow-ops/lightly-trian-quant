"""NVFP4 versions of the round13 QuACK epilogues.  The two per-tensor scales are folded in as a device scalar ``ts = ts_a * ts_b`` (1-element tensor, no host sync):
``acc`` is the raw block-scaled accumulation, so ``z = acc * ts + bias``.  (MXFP4 has no per-tensor scale and uses the round13 epilogues unchanged.)
Same math as ``round13_gemm_fusion_sm120/impl/gemm_ops.py::_quack`` (copied structure, not imported-and-modified).
"""
from __future__ import annotations

import os

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

_E = None


def nv4_epilogues():
    global _E
    if _E is None:
        from quack.activation import dgelu_erf, gelu_erf
        from quack.epilogue import gemm_epilogue
        from quack.epilogue.ops import RowVecReduce

        @gemm_epilogue()
        def bias_ts(acc, bias, ts):
            return {"D": acc * ts + bias}

        @gemm_epilogue(outputs=("y",))
        def res_ts(acc, c, bias, gamma, ts):
            y = acc * ts + bias
            return {"D": c + gamma * y, "y": y}

        @gemm_epilogue(outputs=("h",))
        def gelu_ts(acc, bias, ts):
            z = acc * ts + bias
            return {"D": z, "h": gelu_erf(z)}

        @gemm_epilogue(reduces={"db": RowVecReduce("db")})
        def dgelu_ts(acc, c, ts):
            dz, _ = dgelu_erf(c, acc * ts)
            return {"D": dz, "db": dz}

        @gemm_epilogue()
        def plain_ts(acc, ts):
            return {"D": acc * ts}

        _E = dict(bias=bias_ts, res=res_ts, gelu=gelu_ts, dgelu=dgelu_ts, plain=plain_ts)
    return _E
