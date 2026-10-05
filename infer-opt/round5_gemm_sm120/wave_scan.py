import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch, mgb
from kt import ab
dev, bf = "cuda", torch.bfloat16
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1 -DNFULL=1 -DSTG=1"
N,K=4096,1024
Wt=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
cfg=dict(epi=0,bm=96,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX)
k=mgb.get(cfg); print("occ",k.occupancy())
fns={}
outs={}
for mt in range(1,25):
    M=96*mt
    h=torch.randn(M,K,device=dev).to(bf); C=torch.empty(M,N,device=dev,dtype=bf); outs[mt]=(h,C)
    fns[mt]=(lambda h=h,C=C: k(h,Wt,b,C=C))
r=ab({str(m):f for m,f in fns.items()},n=10,rounds=5,reps=3)
prev=None
for mt in range(1,25):
    t=r[str(mt)]["min"]; tiles=64*mt
    print("M=%5d tiles=%5d  %.1f us   %.1f TF   us/1000tiles %.1f"%(96*mt,tiles,t,2*96*mt*N*K/t/1e6,t/tiles*1000))
