"""Stream-K correctness + timing (proj / fc2 shapes, fp32 residual epilogue, with and without stats)."""
import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, torch.nn.functional as F
import mgb
from kt import ab
from check import tensor_diff
dev="cuda"; bf=torch.bfloat16
EX="-DORDER=1 -DUNR=2 -DFRAGALL=1 -DNFULL=1"
torch.manual_seed(0)
r=lambda *s: torch.randn(*s,device=dev).to(bf)
for (M,N,K) in ((2058,1024,4096),(2058,1024,1024),(2258,1024,4096)):
    h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev)*3
    R=torch.randn(M,N,device=dev)*100
    ref=R+g*(F.linear(h.float(),W.float(),b.float()))
    for (bm,bn,wm,wn,bk,skm) in ((64,64,2,2,32,0),(64,64,2,2,32,2),(128,64,4,2,32,0),(112,64,1,4,64,0),(64,128,2,2,32,0)):
        cfg=dict(epi=1,rf32=1,bm=bm,bn=bn,bk=bk,st=2,wm=wm,wn=wn,minb=3,streamk=1,skm=skm,extra=EX)
        try: k=mgb.get(cfg)
        except Exception as e: print("build fail",cfg,str(e)[-200:]); continue
        C1=torch.empty_like(R); C2=torch.empty_like(R)
        k(h,W,b,g,R,C1); k(h,W,b,g,R,C2,pdl=True)
        d=tensor_diff(C1,ref)
        print("streamK",(M,N,K),(bm,bn,wm,wn,bk,skm),"occ",k.occupancy(),"cos %.9f max_abs %.3g"%(d['cos'],d['max_abs']),"deterministic",bool(torch.equal(C1,C2)),"counters_zero",bool((next(iter(mgb._WS.values()))[1]==0).all()),flush=True)
# timing fc2 M=2058
M,N,K=2058,1024,4096
h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.rand(N,device=dev); R=torch.randn(M,N,device=dev); C=torch.empty_like(R)
fns={}
for name,cfg in (("splitK2 64x64x32",dict(epi=1,bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,splitk=2,extra=EX)),
                 ("noSplit 112x64x64",dict(epi=1,bm=112,bn=64,bk=64,st=2,wm=1,wn=4,minb=3,extra=EX))):
    k=mgb.get(cfg); fns[name]=(lambda k=k: k(h,W,b,g,R,C))
for (bm,bn,wm,wn,bk,st) in ((64,64,2,2,32,2),(64,64,2,2,32,3),(128,64,4,2,32,2),(112,64,1,4,64,2),(64,128,2,2,32,2),(64,96,2,3,32,2),(96,64,3,2,32,2)):
  for skm in (0,1,2,3,4):
    cfg=dict(epi=1,rf32=1,bm=bm,bn=bn,bk=bk,st=st,wm=wm,wn=wn,minb=3,streamk=1,skm=skm,extra=(EX if N%bn==0 else EX.replace(" -DNFULL=1","")))
    try: k=mgb.get(cfg)
    except Exception as e: continue
    fns[f"streamK {bm}x{bn}x{bk} st{st} w{wm}x{wn} skm{skm}(occ{k.occupancy()})"]=(lambda k=k: k(h,W,b,g,R,C))
res=ab(fns,n=10,rounds=5,reps=3)
for kk,v in sorted(res.items(),key=lambda x:x[1]['min'])[:14]: print("%-46s %.1f us  %.1f TF"%(kk,v["min"],2*M*N*K/v["min"]/1e6))
