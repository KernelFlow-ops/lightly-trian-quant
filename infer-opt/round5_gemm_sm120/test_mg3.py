import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/common"); sys.path.insert(0,"/root/lightly-train/infer-opt/round3_fusion")
import torch, torch.nn.functional as F
import mgb
from kt import kt
from check import tensor_diff
dev="cuda"; bf=torch.bfloat16
torch.manual_seed(0)
r=lambda *s: torch.randn(*s,device=dev).to(bf)
# ---- RES fp32 / bf16 (proj-like and fc2-like)
for (M,N,K,cfg) in [(2058,1024,4096,dict(bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3)),(2258,1024,1024,dict(bm=64,bn=128,bk=32,st=2,wm=2,wn=2,minb=3)),(2058,768,3072,dict(bm=64,bn=96,bk=32,st=2,wm=2,wn=2,minb=3))]:
    for rf in (1,0):
        m=mgb.build(epi=1,rf32=rf,**cfg)
        h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev)*3
        R=torch.randn(M,N,device=dev)*100 if rf else r(M,N)
        C=torch.empty_like(R)
        m.run(h,W,b,g,R,C,None,None,0,0,0,0,False,None,None)
        ref=R.float()+g*(F.linear(h.float(),W.float(),b.float()))
        d=tensor_diff(C.float(),ref)
        print("RES",M,N,K,"rf32" if rf else "bf16",cfg["bm"],cfg["bn"],"cos %.8f max_abs %.4g"%(d['cos'],d['max_abs']),flush=True)
# ---- QKV + rope vs round3 TileLang
from impl import ops as ops3
B,Ntok,d_,H=2,1029,1024,16
for (bn,wn,wm) in [(64,2,2),(64,1,4),(128,4,2),(128,2,2)]:
    cfg=dict(bm=64,bn=bn,bk=32,st=2,wm=wm,wn=wn,minb=3)
    try: m=mgb.build(epi=2,**cfg)
    except Exception as e: print("build fail",cfg,str(e)[-300:]); continue
    for prefix in (5,105):
        Nt=prefix+1024 if False else 1029 if prefix==5 else 1129
        h,W,b=r(B,Nt,d_),r(3*d_,d_)*d_**-0.5,r(3*d_)
        ang=torch.randn(1024,32,device=dev)
        sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
        out=torch.empty(3,B,Nt,H,64,device=dev,dtype=bf)
        m.run(h.reshape(-1,d_),W,b,None,None,out,sin,cos,Nt,prefix,d_,H,False,None,None)
        ref=ops3.qkv_rope(h,W,b,sin,cos,prefix,H).contiguous()  # logical [3,B,H,N,64]
        got=out.transpose(2,3)
        dd=tensor_diff(got.float(),ref.float())
        print("QKV bn",bn,"wm",wm,"wn",wn,"prefix",prefix,"cos %.8f max_abs %.4g"%(dd['cos'],dd['max_abs']),flush=True)
# ---- split-K partial
cfg=dict(bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3)
for S in (2,4):
    m=mgb.build(epi=3,splitk=S,**cfg)
    M,N,K=2058,1024,4096
    h,W=r(M,K),r(N,K)*K**-0.5
    P=torch.empty(S,M,N,device=dev)
    m.run(h,W,torch.zeros(N,device=dev,dtype=bf),None,None,P,None,None,0,0,0,0,False,None,None)
    dd=tensor_diff(P.sum(0),F.linear(h.float(),W.float()))
    print("SPLITK",S,"cos %.8f max_abs %.4g"%(dd['cos'],dd['max_abs']))

# ---- split-K with in-kernel last-CTA reduce + RES epilogue (fp32), determinism + counter reset
for (bm,bn,S,M,N,K) in [(64,64,2,2058,1024,4096),(64,64,3,2058,1024,4096),(64,64,4,2058,1024,4096),(128,64,2,2258,1024,4096),(64,64,2,2058,1024,1024)]:
    cfg=dict(bm=bm,bn=bn,bk=32,st=2,wm=2,wn=2,minb=3)
    m=mgb.build(epi=1,rf32=1,splitk=S,**cfg)
    h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev)*3
    R=torch.randn(M,N,device=dev)*100
    C1=torch.empty_like(R); C2=torch.empty_like(R)
    mgb.call(m,h,W,b,g,R,C1,splitk=S,bm=bm,bn=bn); mgb.call(m,h,W,b,g,R,C2,splitk=S,bm=bm,bn=bn)
    ref=R.float()+g*(F.linear(h.float(),W.float(),b.float()))
    d=tensor_diff(C1.float(),ref)
    print("RES splitK",S,bm,bn,M,N,K,"cos %.8f max_abs %.4g"%(d['cos'],d['max_abs']),"deterministic",bool(torch.equal(C1,C2)),"counters_zero",bool((mgb._WS['cuda'][1]==0).all()),flush=True)
