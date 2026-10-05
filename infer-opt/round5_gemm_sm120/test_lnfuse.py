"""Correctness of the LayerNorm-fused GEMMs (LNF consumer + STATS producer) incl. massive-activation rows.
   x (fp32 residual) --producer(STATS: x1 = x + g*(A@Wp^T+b), bf16 copy, row-stat partials)--> consumer(LNF: qkv+rope / fc1+gelu)
Reference chain: exact fp32/fp64 LayerNorm -> linear; and the unfused bf16 pipeline used by round3 (LN kernel -> bf16 y -> GEMM)."""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common"); sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import torch, torch.nn.functional as F
import mgb
from check import tensor_diff

dev, bf = "cuda", torch.bfloat16
torch.manual_seed(0)
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"


def fold_ln(W, ln_w, ln_b, bias):
    """W [N,K] (fp32 or bf16 values), LN affine (fp32) -> (Wf bf16 [N,K], cs fp32 [N], bias32 fp32 [N])"""
    Wf = (W.float() * ln_w[None, :]).to(bf)
    cs = Wf.float().sum(1)
    b32 = bias.float() + W.float() @ ln_b
    return Wf.contiguous(), cs.contiguous(), b32.contiguous()


def make_x(B, Nt, d, massive=True):
    x = torch.randn(B, Nt, d, device=dev) * 2 + 0.3
    if massive:  # ViT-L style massive activations on a few tokens / channels
        x[:, :5, 7] = 1.5e5
        x[:, :5, 100] = -1.2e5
        x[:, 5:9, 33] = 8e3
        x[:, 20, :] += 40.0  # a row with a big common offset (mean >> std is the hardest case for the algebraic form)
    return x.contiguous()


def exact_ln(x, w, b, eps=1e-5):
    x = x.double()
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return ((x - mu) * torch.rsqrt(var + eps) * w.double() + b.double())


def run(d, ff, B, Nt, massive, prod_bn):
    M = B * Nt
    x0 = make_x(B, Nt, d, massive).reshape(M, d)
    # producer: x1 = x0 + g*(a@Wp^T + bp)     (a: attention output, bf16)
    a = (torch.randn(M, d, device=dev)).to(bf)
    Wp = (torch.randn(d, d, device=dev) * d ** -0.5).to(bf)
    bp = torch.randn(d, device=dev).to(bf)
    g = torch.rand(d, device=dev) * 3
    ln_w = 0.3 + torch.rand(d, device=dev)
    ln_w[7] = 1e-3   # tiny LN weight on the massive channels (as real models do)
    ln_w[100] = 2e-3
    ln_b = torch.randn(d, device=dev) * 0.1
    pc = dict(epi=1, rf32=1, stats=1, bm=64, bn=prod_bn, bk=32, st=2, wm=2, wn=2, minb=3, extra=EX)
    kp = mgb.get(pc)
    x1 = torch.empty(M, d, device=dev)
    xb = torch.empty(M, d, device=dev, dtype=bf)
    NP = (d + prod_bn - 1) // prod_bn
    stats = torch.empty(M, NP, 2, device=dev)
    kp(a, Wp, bp, g, x0, x1, Cb=xb, stats_out=stats)
    x1_ref = x0 + g * (a.float() @ Wp.float().T + bp.float())
    out = {"x1_vs_ref": tensor_diff(x1, x1_ref)["max_abs"], "xb_ok": bool(torch.equal(xb, x1.to(bf)))}
    # stats check (Chan combine done by the consumer; here recompute on the host to verify the partials themselves)
    cnt = torch.tensor([min(prod_bn, d - i * prod_bn) for i in range(NP)], device=dev, dtype=torch.float32)
    mean = stats[..., 0].sum(1) / d
    m2 = (stats[..., 1] + cnt[None] * (stats[..., 0] / cnt[None] - mean[:, None]) ** 2).sum(1)
    out["mean_err"] = (mean - x1.double().mean(1)).abs().max().item()
    out["var_rel_err"] = ((m2 / d) / x1.double().var(1, unbiased=False) - 1).abs().max().item()

    res = {}
    # ---- consumer 1: fc1 + GELU
    W1 = (torch.randn(ff, d, device=dev) * d ** -0.5).to(bf)
    b1 = torch.randn(ff, device=dev).to(bf)
    W1f, cs1, b1_32 = fold_ln(W1, ln_w, ln_b, b1)
    ck = mgb.get(dict(epi=0, lnf=1, bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, extra=EX))
    o = torch.empty(M, ff, device=dev, dtype=bf)
    ck(xb, W1f, b1_32, C=o, cs=cs1, stats_in=stats, np_in=NP, pcols_in=prod_bn)
    yex = exact_ln(x1, ln_w, ln_b)                                   # fp64 exact LN
    ref_exact = F.gelu((yex @ W1.double().T + b1.double()))
    # unfused bf16 pipeline (round3 style): LN in fp32 -> bf16 y -> bf16 GEMM(+bias) -> gelu -> bf16
    y_bf = ((x1 - x1.mean(1, keepdim=True)) * torch.rsqrt(x1.var(1, unbiased=False, keepdim=True) + 1e-5) * ln_w + ln_b).to(bf)
    ref_unf = F.gelu(F.linear(y_bf.float(), W1.float(), b1.float())).to(bf)
    res["fc1"] = dict(ours_vs_exact=tensor_diff(o.float(), ref_exact.float()), unfused_vs_exact=tensor_diff(ref_unf.float(), ref_exact.float()),
                      ours_vs_unfused=tensor_diff(o.float(), ref_unf.float()))
    # ---- consumer 2: qkv + rope
    H = d // 64
    prefix = Nt - 1024
    Wq = (torch.randn(3 * d, d, device=dev) * d ** -0.5).to(bf)
    bq = torch.randn(3 * d, device=dev).to(bf)
    bq_masked = bq.clone(); bq_masked[d:2 * d] = 0  # LinearKMaskedBias
    Wqf, csq, bq_32 = fold_ln(Wq, ln_w, ln_b, bq_masked)
    ang = torch.randn(1024, 32, device=dev)
    sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()
    qk = mgb.get(dict(epi=2, lnf=1, bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, extra=EX))
    oq = torch.empty(3, B, Nt, H, 64, device=dev, dtype=bf)
    qk(xb, Wqf, bq_32, C=oq, cs=csq, stats_in=stats, np_in=NP, pcols_in=prod_bn, sin=sin, cos=cos, Ntok=Nt, prefix=prefix, d=d, H=H)
    # references: linear (fp64) -> rope in fp64
    lin = (yex @ Wq.double().T + bq_masked.double()).reshape(B, Nt, 3, H, 64)
    def rope(t):  # t [B,Nt,H,64] fp64 ; rope on tokens >= prefix
        t = t.clone()
        tt = t[:, prefix:]
        c = cos.double()[None, :, None, :]; s_ = sin.double()[None, :, None, :]
        x1_, x2_ = tt[..., :32], tt[..., 32:]
        rot = torch.cat([-x2_, x1_], -1)
        t[:, prefix:] = tt * c + rot * s_
        return t
    q_ref, k_ref, v_ref = rope(lin[:, :, 0]), rope(lin[:, :, 1]), lin[:, :, 2]
    ref = torch.stack([q_ref, k_ref, v_ref], 0).float()
    # unfused bf16 reference through the v1 path (LN->bf16 y -> torch GEMM -> rope in fp32 -> bf16)
    linb = (F.linear(y_bf.float(), Wq.float(), bq_masked.float())).reshape(B, Nt, 3, H, 64)
    def rope32(t):
        t = t.clone(); tt = t[:, prefix:]
        c = cos[None, :, None, :]; s_ = sin[None, :, None, :]
        rot = torch.cat([-tt[..., 32:], tt[..., :32]], -1)
        t[:, prefix:] = tt * c + rot * s_
        return t
    unf = torch.stack([rope32(linb[:, :, 0]), rope32(linb[:, :, 1]), linb[:, :, 2]], 0).to(bf).float()
    got = oq.float()
    res["qkv"] = dict(ours_vs_exact=tensor_diff(got, ref), unfused_vs_exact=tensor_diff(unf, ref), ours_vs_unfused=tensor_diff(got, unf))
    out["res"] = res
    return out


def fmt(t):
    return "cos %.8f max_abs %.4g mean_abs %.3g" % (t["cos"], t["max_abs"], t["mean_abs"])


if __name__ == "__main__":
    for (name, d, ff, B, Nt, prod_bn) in (("ViT-L N=1029", 1024, 4096, 2, 1029, 64), ("ViT-L joint N=1129", 1024, 4096, 2, 1129, 128),
                                          ("ViT-B N=1029", 768, 3072, 2, 1029, 64), ("ViT-L N=1029 bn96", 1024, 4096, 2, 1029, 96)):
        for massive in (True, False):
            try:
                r = run(d, ff, B, Nt, massive, prod_bn)
            except Exception as e:  # noqa
                import traceback; traceback.print_exc(); print("FAILED", name, massive, str(e)[:300]); continue
            print(f"== {name} massive={massive}  producer: x1 max_abs {r['x1_vs_ref']:.3g} xb_ok {r['xb_ok']} mean_err {r['mean_err']:.3g} var_rel_err {r['var_rel_err']:.3g}")
            for k in ("fc1", "qkv"):
                v = r["res"][k]
                print(f"   {k}: ours vs exact   {fmt(v['ours_vs_exact'])}")
                print(f"   {k}: unfused vs exact {fmt(v['unfused_vs_exact'])}")
                print(f"   {k}: ours vs unfused  {fmt(v['ours_vs_unfused'])}")
