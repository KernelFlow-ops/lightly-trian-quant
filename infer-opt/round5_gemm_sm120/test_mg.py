"""Correctness of the hand-written CUDA GEMM (mg_core.cu): all epilogues, split-K, N%BN!=0, PDL flag."""
import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/common"); sys.path.insert(0,"/root/lightly-train/infer-opt/round3_fusion")
import torch, torch.nn.functional as F
import mgb
from check import tensor_diff
from multiprocessing import Pool
dev="cuda"; bf=torch.bfloat16
torch.manual_seed(0)
r=lambda *s: torch.randn(*s,device=dev).to(bf)

CF=[]
def add(**c): CF.append(c); return c
c_gelu=[add(epi=0,bm=64,bn=96,bk=32,st=2,wm=2,wn=2,minb=3), add(epi=0,bm=64,bn=64,bk=64,st=2,wm=2,wn=2,minb=3), add(epi=0,bm=96,bn=96,bk=32,st=2,wm=2,wn=2,minb=3)]
c_res=[add(epi=1,rf32=rf,bm=bm,bn=bn,bk=32,st=2,wm=2,wn=2,minb=3) for rf in (1,0) for (bm,bn) in ((64,64),(64,128),(64,96))]
c_qkv=[add(epi=2,bm=64,bn=bn,bk=32,st=2,wm=wm,wn=wn,minb=3) for (bn,wm,wn) in ((64,2,2),(64,4,1),(128,2,4),(128,4,2))]
c_sk=[add(epi=1,rf32=1,bm=bm,bn=bn,bk=32,st=2,wm=2,wn=2,minb=3,splitk=S) for (bm,bn,S) in ((64,64,2),(64,64,3),(64,64,4),(128,64,2),(64,128,3))]
def bld(c): mgb.compile_cubin(c)
if __name__=="__main__":
    with Pool(16) as p: p.map(bld,CF)
    print("built",len(CF),flush=True)
    for c in c_gelu:
        M,N,K=2058,4096,1024
        h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); C=torch.empty(M,N,device=dev,dtype=bf)
        mgb.get(c)(h,W,b,C=C)
        d=tensor_diff(C.float(),F.gelu(F.linear(h.float(),W.float(),b.float())))
        print("GELU",c["bm"],c["bn"],c["bk"],"cos %.8f max_abs %.4g"%(d['cos'],d['max_abs']),flush=True)
    for c in c_res:
        for (M,N,K) in ((2058,1024,4096),(2258,768,3072)):
            if N%c["bn"] and c["bn"]==128: pass
            h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev)*3
            R=torch.randn(M,N,device=dev)*100 if c["rf32"] else r(M,N); C=torch.empty_like(R)
            mgb.get(c)(h,W,b,g,R,C)
            d=tensor_diff(C.float(),R.float()+g*F.linear(h.float(),W.float(),b.float()))
            print("RES rf32=%d"%c["rf32"],c["bm"],c["bn"],M,N,K,"cos %.8f max_abs %.4g"%(d['cos'],d['max_abs']),flush=True)
    from impl import ops as ops3
    B,d_,H=2,1024,16
    for c in c_qkv:
        for prefix,Nt in ((5,1029),(105,1129)):
            h,W,b=r(B,Nt,d_),r(3*d_,d_)*d_**-0.5,r(3*d_)
            ang=torch.randn(1024,32,device=dev)
            sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
            out=torch.empty(3,B,Nt,H,64,device=dev,dtype=bf)
            mgb.get(c)(h.reshape(-1,d_),W,b,C=out,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d_,H=H)
            ref=ops3.qkv_rope(h,W,b,sin,cos,prefix,H).contiguous()
            dd=tensor_diff(out.transpose(2,3).float(),ref.float())
            print("QKV bn",c["bn"],"wm",c["wm"],"wn",c["wn"],"prefix",prefix,"cos %.8f max_abs %.4g"%(dd['cos'],dd['max_abs']),flush=True)
    for c in c_sk:
        for (M,N,K) in ((2058,1024,4096),(2258,1024,1024)):
            h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev)*3
            R=torch.randn(M,N,device=dev)*100
            C1=torch.empty_like(R); C2=torch.empty_like(R)
            k=mgb.get(c); k(h,W,b,g,R,C1); k(h,W,b,g,R,C2,pdl=True)
            d=tensor_diff(C1.float(),R.float()+g*F.linear(h.float(),W.float(),b.float()))
            counters_zero = all(bool((counters == 0).all()) for pools in mgb._WS.values() for _, counters in pools)
            print("RES splitK",c["splitk"],c["bm"],c["bn"],M,N,K,"cos %.8f max_abs %.4g"%(d['cos'],d['max_abs']),"deterministic",bool(torch.equal(C1,C2)),"counters_zero",counters_zero,flush=True)
    print("occupancy", {(c['epi'],c['bm'],c['bn'],c['bk']):mgb.get(c).occupancy() for c in c_gelu})
