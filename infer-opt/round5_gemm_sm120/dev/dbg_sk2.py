import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, torch.nn.functional as F, mgb
dev="cuda"; bf=torch.bfloat16
EX="-DORDER=1 -DUNR=2 -DFRAGALL=1 -DNFULL=1"
torch.manual_seed(0)
M,N,K=2058,1024,4096
r=lambda *s: torch.randn(*s,device=dev).to(bf)
h,W,b=r(M,K),r(N,K)*K**-0.5,r(N); g=torch.ones(N,device=dev); R=torch.zeros(M,N,device=dev)
ref=F.linear(h.float(),W.float(),b.float())
cfg=dict(epi=1,rf32=1,bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,streamk=1,skm=0,extra=EX)
k=mgb.get(cfg)
for it in range(6):
    C=torch.empty_like(R); C.fill_(float('nan'))
    k(h,W,b,g,R,C); torch.cuda.synchronize()
    err=(C-ref).abs()
    tiles=err.view(33*64//64,64,16,64) if False else None
    bad=[]
    for i in range(33):
        for j in range(16):
            e=err[i*64:(i+1)*64, j*64:(j+1)*64].max().item()
            if not (e<0.05): bad.append(i*16+j)
    print("run",it,"bad tiles",len(bad), bad[:12])
