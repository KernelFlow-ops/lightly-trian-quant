"""Per-frame kernel-class comparison r5_v1 vs r11 from two nsys cuda_gpu_kern_sum.csv.  python nsys_cmp.py <prefixA> <prefixB> [iters]"""
import csv, sys, json
A, B = sys.argv[1], sys.argv[2]
it = int(sys.argv[3]) if len(sys.argv) > 3 else 30
CLASSES = [("patch-embed conv / GEMM", ("implicit_convolve", "patch_gemm")), ("R5 GEMM (gemm_kernel/main_kernel)", ("gemm_kernel", "main_kernel")), ("flash attention", ("flash_fwd",)),
           ("head ConvT GEMM (cutlass 256x128)", ("256x128_32x3_nn",)), ("dw_ln", ("dw_ln",)), ("add_ln", ("add_ln_kernel",)),
           ("post_a", ("post_a",)), ("mask bmm (cutlass 64x256)", ("64x256_32x4_tn",)),
           ("tail_ln", ("tail_ln",)), ("prep+post_b", ("prep_kernel", "post_b")), ("cat", ("CatArrayBatchedCopy",))]


def load(p):
    rows = list(csv.DictReader(open(p + "_cuda_gpu_kern_sum.csv")))
    tot = sum(float(r["Total Time (ns)"]) for r in rows) / it / 1e3
    n = sum(int(r["Instances"]) for r in rows) / it
    cls, used = {}, set()
    for name, keys in CLASSES:
        t = 0.0
        for i, r in enumerate(rows):
            if i not in used and any(k in r["Name"] for k in keys):
                t += float(r["Total Time (ns)"]) / it / 1e3
                used.add(i)
        cls[name] = t
    cls["other (small elementwise/copy/LN/cutlass_75/wmma ...)"] = tot - sum(cls.values())
    return tot, n, cls


ta, na, ca = load(A)
tb, nb, cb = load(B)
out = {"A": A, "B": B, "total_us": [ta, tb], "kernels": [na, nb], "classes": {k: [ca[k], cb[k]] for k in ca}}
print(f"total GPU kernel time/frame: {ta:.0f} -> {tb:.0f} us  ({ta - tb:+.0f}); kernels/frame {na:.0f} -> {nb:.0f}")
for k in ca:
    print(f"  {k:55s} {ca[k]:8.1f} -> {cb[k]:8.1f}  ({cb[k] - ca[k]:+7.1f})")
if len(sys.argv) > 4:
    json.dump(out, open(sys.argv[4], "w"), indent=1)
