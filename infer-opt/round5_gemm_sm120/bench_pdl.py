"""PDL microbenchmark: a chain of the four mg GEMM kernels of one transformer layer (qkv, proj, fc1, fc2; ViT-L, M=2058,
buffers independent so the data flow is irrelevant, only the launch pattern matters) captured into one CUDA graph,
launched with / without the programmatic-stream-serialization attribute, and with / without early launch_dependents."""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch, mgb
from kt import ab

dev, bf = "cuda", torch.bfloat16
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1 -DNFULL=1"
M, d, ff, H = 2058, 1024, 4096, 16
r = lambda *s: torch.randn(*s, device=dev).to(bf)  # noqa: E731
NL = 4  # layers per graph (independent buffers per layer to defeat L2 reuse of weights)
Ws = [dict(Wq=r(3 * d, d) * d ** -0.5, bq=r(3 * d), Wp=r(d, d) * d ** -0.5, bp=r(d), W1=r(ff, d) * d ** -0.5, b1=r(ff), W2=r(d, ff) * ff ** -0.5, b2=r(d)) for _ in range(NL)]
x = torch.randn(M, d, device=dev)
xb = x.to(bf)
a = r(M, d)
h1 = r(M, ff)
g = torch.rand(d, device=dev)
qkv = torch.empty(3, 2, 1029, H, 64, device=dev, dtype=bf)
x1 = torch.empty(M, d, device=dev)
a1 = torch.empty(M, ff, device=dev, dtype=bf)
x2 = torch.empty(M, d, device=dev)
ang = torch.randn(1024, 32, device=dev)
sin, cos = torch.sin(ang).repeat(1, 2).float().contiguous(), torch.cos(ang).repeat(1, 2).float().contiguous()


def kern(trig):
    e = lambda x_: dict(extra=x_)
    kq = mgb.get(dict(epi=2, bm=64, bn=128, bk=32, st=2, wm=2, wn=2, minb=3, trig=trig, extra=EX + " -DSTG=1"))
    kp = mgb.get(dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=1, wn=4, minb=3, trig=trig, extra=EX))
    k1 = mgb.get(dict(epi=0, bm=96, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, trig=trig, extra=EX + " -DSTG=1"))
    k2 = mgb.get(dict(epi=1, bm=64, bn=64, bk=32, st=2, wm=2, wn=2, minb=3, splitk=2, trig=trig, extra=EX))
    return kq, kp, k1, k2


def chain(trig, pdl, only=None):
    kq, kp, k1, k2 = kern(trig)

    def f():
        for w in Ws:
            if only in (None, "qkv"):
                kq(xb, w["Wq"], w["bq"], C=qkv, sin=sin, cos=cos, Ntok=1029, prefix=5, d=d, H=H, pdl=pdl)
            if only in (None, "proj"):
                kp(a, w["Wp"], w["bp"], g, x, x1, pdl=pdl)
            if only in (None, "fc1"):
                k1(xb, w["W1"], w["b1"], C=a1, pdl=pdl)
            if only in (None, "fc2"):
                k2(a1, w["W2"], w["b2"], g, x1, x2, pdl=pdl)
    return f


fns = {"no PDL": chain(1, False), "PDL attr + trigger": chain(1, True), "PDL attr, no early trigger": chain(0, True)}
r_ = ab(fns, n=3, rounds=9, reps=3)
for k, v in r_.items():
    print("%-30s %.1f us per 4-layer chain (kernel-sum) -> %.2f us per layer" % (k, v["min"] * 3 / 3 * 1.0, v["min"] / NL * 1))
# note: ab() divides by n (=3 chain replays); v['min'] is us per chain
