"""Quantify the Spark device-amax ts vs the fp32 division amax/(6*448)."""
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
import torch

import fp4_ref as R
import spark_fp4_ext
from common.bench import gpu_lock

with gpu_lock():
    spark = spark_fp4_ext.extension()
    g = torch.Generator(device="cuda").manual_seed(0)
    res = []
    for i in range(20):
        x = (torch.randn(512, 1024, device="cuda", generator=g) * (10 ** (i % 5 - 2))).bfloat16()
        ts_ref = R._nv4_ts(x.float())
        work = torch.zeros(2, dtype=torch.float32, device="cuda")
        q = torch.empty(512, 512, dtype=torch.uint8, device="cuda")
        sf = torch.empty(spark.scale_bytes(512, 1024, 0), dtype=torch.uint8, device="cuda")
        spark.fp4quant(x, q, sf, None, 0, work, work[1:2])
        a, b = float(ts_ref), float(work[1])
        ulp = abs(a - b) / (torch.finfo(torch.float32).eps * abs(a))
        # candidate formulas: reciprocal multiply
        amax = x.float().abs().amax()
        mulrec = float(amax * (1.0 / (6.0 * 448.0)))
        res.append((a == b, ulp, mulrec == b))
    print("ts equal to fp32 division:", sum(r[0] for r in res), "/", len(res), "; max |diff| in ulp-ish units:", max(r[1] for r in res), "; equal to amax*(1/2688):", sum(r[2] for r in res), "/", len(res))
