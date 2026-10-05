import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
ex0="-DORDER=1 -DUNR=2 -DFRAGALL=1 -DSTG=1"
for K in (256, 1024, 4096, 16384):
    M,N=2058,4096
    h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
    C=torch.empty(M,N,device=dev,dtype=bf)
    fns={}
    k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x96x32"]=lambda: k1(h,W,b)
    k=mgb.get(dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=3,extra=ex0)); fns["mg 64x96x32"]=lambda: k(h,W,b,C=C)
    k=mgb.get(dict(epi=0,wm=2,wn=2,bm=64,bn=64,bk=32,st=2,minb=3,extra=ex0+" -DNFULL=1")); fns["mg 64x64x32"]=lambda: k(h,W,b,C=C)
    k64=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=64,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x64x32"]=lambda: k64(h,W,b)
    r=ab(fns,n=4 if K>4096 else 10,rounds=5,reps=3)
    print("K=",K, {kk:"%.1f us (%.1f TF)"%(v["min"],2*M*N*K/v["min"]/1e6) for kk,v in r.items()})
