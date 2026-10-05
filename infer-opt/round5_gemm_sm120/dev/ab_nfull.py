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
k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=64,bk=64,stages=2,threads=128,ws=False,tma=False); fns["TL 64x64x64"]=lambda: k1(h,W,b)
k2=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=128,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x128x32"]=lambda: k2(h,W,b)
ex0="-DORDER=1 -DUNR=2 -DFRAGALL=1"
for (bm,bn,bk) in ((64,64,64),(64,128,32)):
  for ex in (ex0, ex0+" -DNFULL=1", ex0+" -DNFULL=1 -DWPRED=0", ""):
    k=mgb.get(dict(epi=0,wm=2,wn=2,bm=bm,bn=bn,bk=bk,st=2,minb=3,extra=ex)); fns[f"mg {bm}x{bn}x{bk} {ex[len(ex0):] or ('base' if not ex else 'best')}"]=(lambda k=k: k(h,W,b,C=C))
r=ab(fns)
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print("%-40s %.1f"%(kk,v["min"]))
