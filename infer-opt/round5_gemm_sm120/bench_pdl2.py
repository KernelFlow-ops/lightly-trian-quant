import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch, mgb
from kt import ab
dev, bf = "cuda", torch.bfloat16
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1 -DNFULL=1"
# tiny dependent kernels: 2058x64x64 gemm (RES epilogue): each kernel reads the previous output (true dependency chain)
M,N,K=2058,64,64
A=[torch.randn(M,K,device=dev).to(bf) for _ in range(2)]
W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf); g=torch.ones(N,device=dev)
X=[torch.randn(M,N,device=dev) for _ in range(2)]
k=mgb.get(dict(epi=1,bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX))
def chain(pdl,n=50):
    def f():
        for i in range(n):
            k(A[0],W,b,g,X[i%2],X[(i+1)%2],pdl=pdl)
    return f
r=ab({"no PDL":chain(False),"PDL":chain(True)},n=2,rounds=7,reps=3)
for kk,v in r.items(): print("%-8s %.2f us per kernel"%(kk,v["min"]/50))
