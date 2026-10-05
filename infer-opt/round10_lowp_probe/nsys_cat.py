"""nsys sqlite (r5_v1 whole-frame graph over the 126-frame video) -> per-frame GPU time per category.
usage: nsys_cat.py <sqlite> <model> <out.json>"""
import json, re, sqlite3, statistics, sys, collections
db = sqlite3.connect(sys.argv[1]); model = sys.argv[2]
cur = db.cursor()
names = dict(cur.execute("select id, value from StringIds").fetchall())
rows = cur.execute("select start,end,demangledName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ks = [(s, e, names[n], (gx, gy, gz)) for s, e, n, gx, gy, gz in rows]
# frame boundaries at prep_kernel
starts = [i for i, k in enumerate(ks) if k[2].startswith("void prep_kernel")]
frames = [ks[a:b] for a, b in zip(starts, starts[1:] + [len(ks)])]
frames = frames[:-0 or None]
nb = 24 if model == "vitl16" else 12
def classify(frame):
    out = collections.defaultdict(float)
    gi = 0
    for s, e, n, g in frame:
        d = (e - s) / 1e3
        if n.startswith("gemm_kernel") or n.startswith("main_kernel"):
            kind = ["qkv", "proj", "fc1", "fc2"][gi % 4]; layer = gi // 4; gi += 1
            out[f"gemm_{kind}" + ("_joint" if layer >= nb - 3 else "")] += d
        elif "flash_fwd" in n: out["attention"] += d
        elif "add_ln_kernel" in n or "vectorized_layer_norm" in n: out["layernorm"] += d
        elif "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128" in n: out["head_convT_gemm"] += d
        elif "dw_ln_kernel" in n: out["head_dw_ln"] += d
        elif "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256" in n: out["head_mask_einsum"] += d
        elif "cutlass_75" in n or "wmma_tensorop" in n: out["head_small_gemm"] += d
        elif "implicit_convolve_sgemm" in n: out["patch_embed_conv"] += d
        elif n.startswith("void prep_kernel"): out["prep"] += d
        elif "post_a_kernel" in n or "post_b_kernel" in n: out["post"] += d
        else: out["misc_elementwise_cat"] += d
    return out
per = [classify(f) for f in frames[3:-2]]  # skip warm-up / tail frames
keys = sorted({k for p in per for k in p})
res = {k: statistics.fmean(p.get(k, 0.0) for p in per) for k in keys}
tot = sum(res.values())
span = [(f[-1][1] - f[0][0]) / 1e3 for f in frames[3:-2]]
print(model, "frames", len(per), "GPU kernel us/frame", round(tot, 1), "frame span us (first->last kernel) mean", round(statistics.fmean(span), 1))
groups = collections.OrderedDict([
    ("backbone GEMM (qkv+proj+fc1+fc2)", [k for k in res if k.startswith("gemm_")]),
    ("attention", ["attention"]), ("LayerNorm", ["layernorm"]),
    ("head GEMMs (convT + mask einsum + small)", ["head_convT_gemm", "head_mask_einsum", "head_small_gemm"]),
    ("head dw_ln", ["head_dw_ln"]), ("patch-embed conv", ["patch_embed_conv"]),
    ("prep + post", ["prep", "post"]), ("misc (cat/copy/elementwise)", ["misc_elementwise_cat"])])
summ = {}
for g, ks_ in groups.items():
    v = sum(res.get(k, 0) for k in ks_); summ[g] = v
    print(f"{g:45s} {v/1e3:7.3f} ms  {100*v/tot:5.1f}%")
print("detail:")
for k in keys: print(f"  {k:28s} {res[k]:9.1f} us")
json.dump({"model": model, "per_category_us": res, "grouped_us": summ, "total_kernel_us": tot, "frames_used": len(per),
           "frame_span_us_mean": statistics.fmean(span)}, open(sys.argv[3], "w"), indent=1)
