import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from kt import ab
dev="cuda"; bf=torch.bfloat16
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
C=torch.empty(M,N,device=dev,dtype=bf)
fns={}
for minb in (1,2,3):
  for ex in ("-DORDER=1 -DUNR=2 -DFRAGALL=1",""):
    c=dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=minb,extra=ex)
    p,log,_=mgb.compile_cubin(c)
    import re
    regs=re.search(r"Used (\d+) registers",open(log).read()).group(1)
    k=mgb.get(c); fns[f"mg minb{minb} regs{regs} {'best' if ex else 'base'} occ{k.occupancy()}"]=(lambda k=k: k(h,W,b,C=C))
r=ab(fns)
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print(kk,"%.1f"%v["min"])
