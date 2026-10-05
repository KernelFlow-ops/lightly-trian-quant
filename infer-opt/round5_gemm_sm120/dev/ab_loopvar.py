import sys, itertools
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from impl import gemm_tl
from kt import ab
from multiprocessing import Pool
dev="cuda"; bf=torch.bfloat16
K=4096; M,N=2058,4096
def cfgs():
    out=[]
    for bk,st,order,unr,frag in itertools.product((32,64),(2,3),(0,1),(1,2),(0,1)):
        ex=f"-DORDER={order} -DUNR={unr} -DFRAGALL={frag} -DSTG=1"
        out.append((f"bk{bk} st{st} ord{order} unr{unr} frag{frag}",dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=bk,st=st,minb=3,extra=ex)))
    return out
def bld(a): mgb.compile_cubin(a[1])
if __name__=="__main__":
    C=cfgs()
    with Pool(16) as p: p.map(bld,C)
    h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
    O=torch.empty(M,N,device=dev,dtype=bf)
    fns={}
    k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x96x32"]=lambda: k1(h,W,b)
    for n,c in C:
        k=mgb.get(c); fns["mg "+n]=(lambda k=k: k(h,W,b,C=O))
    r=ab(fns,n=4,rounds=4,reps=2)
    for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print("%-34s %.1f us %.1f TF"%(kk,v["min"],2*M*N*K/v["min"]/1e6))
