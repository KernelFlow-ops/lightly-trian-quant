import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch
import mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
C=torch.empty(M,N,device=dev,dtype=bf)
fns={}
tl=lambda **c: gemm_tl.gemm_bias_gelu_kernel(M,N,K,**c)
k1=tl(bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x96x32 cp"]=lambda: k1(h,W,b)
k2=tl(bm=128,bn=64,bk=32,stages=4,threads=256,ws=True,tma=True); fns["TL r3cfg 128x64 ws"]=lambda: k2(h,W,b)
k3=tl(bm=64,bn=64,bk=64,stages=2,threads=128,ws=False,tma=False); fns["TL 64x64x64 cp"]=lambda: k3(h,W,b)
for name,cfg in [("mg 64x96x32 minb3",dict(bm=64,bn=96,bk=32,st=2,minb=3)),("mg 64x96x32 minb4",dict(bm=64,bn=96,bk=32,st=2,minb=4)),("mg 96x96x32 minb2",dict(bm=96,bn=96,bk=32,st=2,minb=2)),("mg 64x64x64 minb3",dict(bm=64,bn=64,bk=64,st=2,minb=3))]:
    k=mgb.get(dict(epi=0,wm=2,wn=2,**cfg)); fns[name]=(lambda k=k: k(h,W,b,C=C))
fns["torch linear+gelu"]=lambda: torch.nn.functional.gelu(torch.nn.functional.linear(h,W,b))
fns["torch linear"]=lambda: torch.nn.functional.linear(h,W,b)
r=ab(fns)
for k,v in sorted(r.items(),key=lambda x:x[1]["min"]): print("%-24s min %.1f p50 %.1f  %.1fTF"%(k,v["min"],v["p50"],2*M*N*K/v["min"]/1e6))
