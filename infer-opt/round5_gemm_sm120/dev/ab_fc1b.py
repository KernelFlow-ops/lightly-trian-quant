import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch
import mgb
from impl import gemm_tl
from kt import ab
from multiprocessing import Pool
dev="cuda"; bf=torch.bfloat16
V={"base":"", "frag":"-DFRAGALL=1", "order1_unr2":"-DORDER=1 -DUNR=2", "order1_unr2_frag":"-DORDER=1 -DUNR=2 -DFRAGALL=1","order1":"-DORDER=1"}
def cfgs(): return [(n,dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=3,extra=x)) for n,x in V.items()]
def bld(a): mgb.compile_cubin(a[1])
if __name__=="__main__":
    with Pool(8) as p: p.map(bld,cfgs())
    M,N,K=2058,4096,1024
    h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
    C=torch.empty(M,N,device=dev,dtype=bf)
    fns={}
    k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x96x32 cp"]=lambda: k1(h,W,b)
    for n,c in cfgs():
        k=mgb.get(c); fns["mg "+n]=(lambda k=k: k(h,W,b,C=C))
    r=ab(fns)
    for k,v in sorted(r.items(),key=lambda x:x[1]["min"]): print("%-28s min %.1f p50 %.1f  %.1fTF"%(k,v["min"],v["p50"],2*M*N*K/v["min"]/1e6))
