import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import ops as r5
from kt import ab
dev="cuda"; bf=torch.bfloat16
B,N,d,ff,H=2,1029,1024,4096,16
M=B*N
# synthetic weights (same shapes as ViT-L block), NB distinct sets
def mkw():
    r=lambda *s: (torch.randn(*s,device=dev)*0.02).to(bf)
    w=dict(Wqkv=r(3*d,d),bqkv=r(3*d),Wproj=r(d,d),bproj=r(d),W1=r(ff,d),b1=r(ff),W2=r(d,ff),b2=r(d),g1=torch.rand(d,device=dev),g2=torch.rand(d,device=dev),
           ln1_w=torch.ones(d,device=dev),ln1_b=torch.zeros(d,device=dev),ln1_eps=1e-5,ln2_w=torch.ones(d,device=dev),ln2_b=torch.zeros(d,device=dev),ln2_eps=1e-5)
    return w
NBs=8
Ws=[mkw() for _ in range(NBs)]
ang=torch.randn(1024,32,device=dev); sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
x=(torch.randn(B,N,d,device=dev)*2)
def chain(idxs):
    def f():
        xx=x
        for i in idxs: xx=r5.fused_block(xx,Ws[i],sin,cos,5,H)
        return xx
    return f
fns={"hot: same w x4":chain([0,0,0,0]),"cold: 4 distinct":chain([0,1,2,3]),"cold: 8 distinct":chain(list(range(8)))}
r=ab(fns,n=1,rounds=6,reps=3)
for k,v in r.items(): print(k, "%.1f us"%(v["min"]/(8 if "8" in k else 4)),"per layer")
# per-kernel view for the fc1 gemm alone: hot vs cold weights
h=torch.randn(M,d,device=dev).to(bf)
def g(idx): return lambda: r5.gemm_bias_gelu(h,Ws[idx]["W1"],Ws[idx]["b1"])
r=ab({"fc1 hot":lambda: [g(0)() for _ in range(8)],"fc1 cold(8 distinct)":lambda: [g(i)() for i in range(8)]},n=1,rounds=6,reps=3)
for k,v in r.items(): print(k,"%.1f us per call"%(v["min"]/8))
