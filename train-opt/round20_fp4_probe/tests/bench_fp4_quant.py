"""R20a part 3: existing FP4 / MXFP8 quantizer kernels on real-size tensors (CUDA-graph replay, interleaved, median of 3) + HBM-bandwidth floor of a
one-read dual (row + transposed copy) FP4 quantizer.   -> profile/fp4_quant_cost.json

Tensors per model = the three shape classes of the 8 per-block activation/gradient tensors (M x d, M x 3d, M x 4d) at plain M, plus the head ConvT1/2 tensors
([7168|28672, 1024] / [14336|57344, 768] and the dZ [M, 4C]).
Candidates: Spark fp4quant (nv4 with given ts / with device amax = two launches; mx4 = floor recipe), QuTLASS fusedQuantizeNvAbsMax / MxAbsMax (row, with 16/32 Hadamard) and
backward_t_bf16 (transpose + Hadamard + MXFP4: the wgrad copy), r17 mx8 quant_dual / row (control), QuACK to_nvfp4_compiled.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND / "tests"))
import torch

import fp4_ref as R
from bench_fp4_gemm import interleave, pad
from common.bench import gpu_lock

dev = "cuda"
MODELS = {"vitl16": dict(d=1024, M=7203, Mj=7903, head=(7168, 28672)), "vitb16": dict(d=768, M=14406, Mj=15806, head=(14336, 57344))}


def hadamard(n, scale):
    H = torch.tensor([[1.0]], device=dev)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H * scale).bfloat16()


def main():
    from round17_mxfp8_sm120.impl import mx8

    out = {"models": {}, "notes": []}
    with gpu_lock():
        import spark_fp4_ext

        spark = spark_fp4_ext.extension()
        qut = None
        try:
            torch.ops.load_library("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/build/qutlass/_qutlass_C.so")
            qut = torch.ops._qutlass_C
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"qutlass load failed: {type(e).__name__}: {str(e)[:200]}")
        from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_nvfp4_compiled

        H16, H32 = hadamard(16, 0.25), hadamard(32, 32 ** -0.5)
        g = torch.Generator(device=dev).manual_seed(0)
        # HBM bandwidth reference: device-to-device copy
        big = torch.empty(512 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16)
        big2 = torch.empty_like(big)
        us, _ = interleave({"copy": lambda: big2.copy_(big)}, inner=1, reps=5)
        out["hbm_copy_GBps"] = 2 * big.numel() * 2 / us["copy"] / 1e3
        print("HBM copy GB/s", round(out["hbm_copy_GBps"]), flush=True)
        del big, big2
        bw = out["hbm_copy_GBps"] * 1e9  # B/s achievable
        for model, c in MODELS.items():
            d = c["d"]
            shapes = {f"M{c['M']}x{d}": (c["M"], d), f"M{c['M']}x{3 * d}": (c["M"], 3 * d), f"M{c['M']}x{4 * d}": (c["M"], 4 * d),
                      f"M{c['Mj']}x{d}": (c["Mj"], d), f"M{c['Mj']}x{4 * d}": (c["Mj"], 4 * d),
                      f"head{c['head'][0]}x{d}": (c["head"][0], d), f"head{c['head'][1]}x{d}": (c["head"][1], d),
                      f"head_dZ{c['head'][0]}x{4 * d}": (c["head"][0], 4 * d), f"head_dZ{c['head'][1]}x{4 * d}": (c["head"][1], 4 * d)}
            res = {}
            for sname, (rows, cols) in shapes.items():
                x = torch.randn(rows, cols, device=dev, generator=g).bfloat16()
                Mp = pad(rows, 128)
                xpad = torch.nn.functional.pad(x, (0, 0, 0, Mp - rows)).contiguous()
                ts = torch.full((1,), float(x.float().abs().max()) / 2688.0, device=dev)
                q = torch.empty(rows, cols // 2, dtype=torch.uint8, device=dev)
                sfn = torch.empty(spark.scale_bytes(rows, cols, 0), dtype=torch.uint8, device=dev)
                sfm = torch.empty(spark.scale_bytes(rows, cols, 1), dtype=torch.uint8, device=dev)
                work = torch.zeros(2, dtype=torch.float32, device=dev)
                xt = x.t().contiguous()  # for the "transpose then quantize" cost of a Spark-based col copy
                qt = torch.empty(cols, Mp // 2, dtype=torch.uint8, device=dev)
                sft = torch.empty(spark.scale_bytes(cols, Mp, 0), dtype=torch.uint8, device=dev)
                xtp = torch.nn.functional.pad(xt, (0, Mp - rows)).contiguous()

                def spark_nv_dev():
                    work.zero_()
                    spark.fp4quant(x, q, sfn, None, 0, work, work[1:2])

                cands = {
                    "mx8_quant_dual(row+col)": lambda: mx8.quantize(x, mpad=Mp),
                    "mx8_quant_row": lambda: mx8.quantize(x, col=False, mpad=Mp),
                    "spark_nv4_row(given ts)": lambda: spark.fp4quant(x, q, sfn, ts, 0, None, None),
                    "spark_nv4_row(device amax, 2 launches)": spark_nv_dev,
                    "spark_mx4_row(floor)": lambda: spark.fp4quant(x, q, sfm, None, 1, None, None),
                    "transpose_copy_only(x.t().contiguous())": lambda: x.t().contiguous(),
                }
                if cols % 128 == 0:
                    cands["spark_nv4_col(transpose copy + quant, ts given)"] = lambda: spark.fp4quant(torch.nn.functional.pad(x.t().contiguous(), (0, Mp - rows)), qt, sft, ts, 0, None, None)
                    if sname == f"M{c['M']}x{d}":
                        cands["quack_to_nvfp4_compiled_row(no sf pack)"] = lambda: to_nvfp4_compiled(x, 16, ts.reshape(()))
                if qut is not None:
                    try:
                        o1 = torch.empty(rows, cols // 2, dtype=torch.uint8, device=dev)
                        s1 = torch.empty(pad(rows, 128), pad(cols // 16, 4), dtype=torch.float8_e4m3fn, device=dev)
                        gs = torch.ones(1, device=dev)
                        cands["qutlass_fusedQuantizeNvAbsMax(row,H16)"] = lambda: qut.fusedQuantizeNvAbsMax(x, H16, o1, s1, gs)
                        o2 = torch.empty(rows, cols // 2, dtype=torch.uint8, device=dev)
                        s2 = torch.empty(pad(rows, 128), pad(cols // 32, 4), dtype=torch.float8_e8m0fnu, device=dev)
                        cands["qutlass_fusedQuantizeMxAbsMax(row,H32)"] = lambda: qut.fusedQuantizeMxAbsMax(x, H32, o2, s2)
                        xe = xpad if Mp != rows else x
                        o3 = torch.empty(cols, Mp // 2, dtype=torch.float4_e2m1fn_x2, device=dev)
                        s3 = torch.empty(cols, Mp // 32, dtype=torch.float8_e8m0fnu, device=dev)
                        cands["qutlass_backward_t_bf16(col: transpose+H32+mx4)"] = lambda: qut.backward_t_bf16(xe, H32, o3, s3)
                    except Exception as e:  # noqa: BLE001
                        out["notes"].append(f"qutlass cand setup {sname}: {type(e).__name__}: {str(e)[:200]}")
                us, errs = interleave(cands, inner=3, reps=10)
                n = rows * cols
                bytes_in = 2.0 * n
                # one-read dual quantizer floor: read bf16 once; write row copy + transposed copy (0.5 B/elem each) + scales (nv4: 1/16 B per elem each; mx4: 1/32) [+ a second read for a separate nv4 amax pass]
                floor_nv4 = (bytes_in + 2 * 0.5 * n + 2 * n / 16) / bw * 1e6
                floor_nv4_amax = floor_nv4 + bytes_in / bw * 1e6
                floor_mx4 = (bytes_in + 2 * 0.5 * n + 2 * n / 32) / bw * 1e6
                floor_row_only_nv4 = (bytes_in + 0.5 * n + n / 16) / bw * 1e6
                res[sname] = {"rows": rows, "cols": cols, "us": us, "errors": errs,
                              "floor_us": {"dual_nv4_one_read": floor_nv4, "dual_nv4_plus_amax_pass": floor_nv4_amax, "dual_mx4_one_read": floor_mx4, "row_only_nv4": floor_row_only_nv4,
                                           "read_bf16_only": bytes_in / bw * 1e6}}
                best = sorted(us.items(), key=lambda kv: kv[1])
                print(f"[{model}] {sname}: " + ", ".join(f"{k}={v:.0f}" for k, v in best[:7]) + f" | floor dual_nv4={floor_nv4:.0f} (+amax {floor_nv4_amax:.0f})", flush=True)
                if errs:
                    print("   errs:", {k: v[:100] for k, v in errs.items()}, flush=True)
                del x, xpad, xt, xtp
                torch.cuda.empty_cache()
            out["models"][model] = res
            (ROUND / "profile" / "fp4_quant_cost.json").write_text(json.dumps(out, indent=1))
    print("wrote fp4_quant_cost.json")


if __name__ == "__main__":
    main()
