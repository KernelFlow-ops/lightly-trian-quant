import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, torch.nn.functional as F
import mgb
from kt import kt
from check import tensor_diff
from multiprocessing import Pool
dev="cuda"; bf=torch.bfloat16
cfgs=[dict(bm=64,bn=96,bk=32,st=2,wm=2,wn=2,minb=3),dict(bm=64,bn=96,bk=64,st=2,wm=2,wn=2,minb=3),dict(bm=96,bn=96,bk=32,st=2,wm=2,wn=2,minb=3),dict(bm=64,bn=64,bk=64,st=2,wm=2,wn=2,minb=3),dict(bm=64,bn=128,bk=32,st=2,wm=2,wn=2,minb=3),dict(bm=64,bn=64,bk=32,st=3,wm=2,wn=2,minb=4)]
def bld(c):
    mgb.build(epi=0,**c)
if __name__=="__main__":
    with Pool(6) as p: p.map(bld,cfgs)
    M,N,K=2058,4096,1024
    h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
    ref=F.gelu(F.linear(h.float(),W.float(),b.float()))
    for c in cfgs:
        m=mgb.build(epi=0,**c)
        C=torch.empty(M,N,device=dev,dtype=bf)
        m.run(h,W,b,None,None,C,None,None,0,0,0,0,False,None,None)
        d=tensor_diff(C.float(),ref)
        us=kt(lambda: m.run(h,W,b,None,None,C,None,None,0,0,0,0,False,None,None))["min"]
        print(c,"occ",m.occupancy(),"cos %.7f"%d['cos'],"%.1fus %.1fTF"%(us,2*M*N*K/us/1e6),flush=True)
