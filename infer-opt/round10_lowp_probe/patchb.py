s = open("src0/mgb.py").read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:50], s.count(a))
    s = s.replace(a, b)
rep('"""Build/launch helper for the hand-written CUDA GEMM (mg_core.cu).', '"""Round 10: copy of round5_gemm_sm120/impl/mgb.py for the fp16-accumulate prototype (mg16_core.cu); own build dir / extension name.\n\nOriginal doc: Build/launch helper for the hand-written CUDA GEMM (mg_core.cu).')
rep('BUILD = "/root/autodl-tmp/infer-opt-tmp/gemm120/build"', 'BUILD = "/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build/mg16"')
rep('PARAMS_FMT = "=14Q10if4x2Q2i"', 'PARAMS_FMT = "=14Q10i2f2Q2i"')
rep('name="mgl", sources=[str(_HERE / "mgl.cu")]', 'name="mg16l", sources=[str(_HERE / "mg16l.cu")]')
rep('src = _HERE / "mg_core.cu"', 'src = _HERE / "mg16_core.cu"')
rep('out = f"{BUILD}/cubin/mg_{h}.cubin"', 'out = f"{BUILD}/cubin/mg16_{h}.cubin"')
rep('log = f"{BUILD}/cubin/mg_{h}.ptxas.txt"', 'log = f"{BUILD}/cubin/mg16_{h}.ptxas.txt"')
rep("KEYS = (", "KEYS = (")
rep("splitk=1, lnf=0, stats=0, occ=0, kdim=0, extra=\"\")", "splitk=1, lnf=0, stats=0, occ=0, kdim=0, f16acc=1, outf16=1, promo=0, extra=\"\")")
rep('defs = [f"-D{k.upper()}={v}" for k, v in c.items() if k not in ("extra", "occ", "kdim")]', 'defs = [f"-D{k.upper()}={v}" for k, v in c.items() if k not in ("extra", "occ", "kdim", "promo")]\n        if c["promo"]:\n            defs.append(f"-DPROMO={c[\'promo\']}")')
rep("self.smem = c[\"st\"] * (c[\"bm\"] + c[\"bn\"]) * c[\"bk\"] * 2", "self.smem = c[\"st\"] * (c[\"bm\"] + c[\"bn\"]) * c[\"bk\"] * 2   # fp16 operands: same 2 bytes/elt")
rep("cs=None, stats_in=None, stats_out=None, Cb=None, np_in=0, pcols_in=0, eps=1e-5, prefetch=()):", "cs=None, stats_in=None, stats_out=None, Cb=None, np_in=0, pcols_in=0, eps=1e-5, prefetch=(), oscale=1.0):")
rep("M, N, K, Ntok, prefix, d, H, 1 if pdl else 0, np_in, pcols_in, eps,\n", "M, N, K, Ntok, prefix, d, H, 1 if pdl else 0, np_in, pcols_in, eps, oscale,\n")
rep('"""x fp32 [B,Nin,D]', '"""x fp32 [B,Nin,D]')  if False else None
# drop the ln_prep helper (not used by the prototype)
i = s.index("# ----------------------------------------------------------------------------------------------------------------- ln_prep")
s = s[:i]
open("impl/mg16b.py", "w").write(s)
print("ok")
