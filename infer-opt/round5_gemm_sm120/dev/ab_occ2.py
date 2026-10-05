import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
C=torch.empty(M,N,device=dev,dtype=bf)
fns={}
k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL native"]=lambda: k1(h,W,b)
ex="-DORDER=1 -DUNR=2 -DFRAGALL=1"
for occ in (0,2,3,4):
    k=mgb.get(dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=3,occ=occ,extra=ex)); fns[f"mg best occ{occ}(real {k.occupancy()})"]=(lambda k=k: k(h,W,b,C=C))
# TL with more CTAs?  (TL 64x96 occupancy by regs)
r=ab(fns)
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print(kk,"%.1f"%v["min"])
