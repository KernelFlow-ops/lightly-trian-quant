import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, torch.nn.functional as F
import mgb
from kt import kt
from check import tensor_diff
dev="cuda"; bf=torch.bfloat16
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
m=mgb.build(epi=0)
print("occupancy",m.occupancy())
C=torch.empty(M,N,device=dev,dtype=bf)
m.run(h,W,b,None,None,C,None,None,0,0,0,0,False,None,None)
ref=F.gelu(F.linear(h.float(),W.float(),b.float()))
print(tensor_diff(C.float(),ref))
us=kt(lambda: m.run(h,W,b,None,None,C,None,None,0,0,0,0,False,None,None))["min"]
print("mg gelu 64x64x64 st2: %.1fus %.1fTF"%(us,2*M*N*K/us/1e6))
