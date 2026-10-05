import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common"); sys.path.insert(0,"/root/lightly-train/infer-opt/round3_fusion")
import torch, mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
ex0="-DORDER=1 -DUNR=2 -DFRAGALL=1"
# ---- fc1
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
C0=torch.empty(M,N,device=dev,dtype=bf); C1=torch.empty(M,N,device=dev,dtype=bf)
fns={}
k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL 64x96x32"]=lambda: k1(h,W,b)
for stg in (0,1):
    k=mgb.get(dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=3,extra=ex0+f" -DSTG={stg}")); Cc=[C0,C1][stg]
    fns[f"mg fc1 64x96x32 STG={stg}"]=(lambda k=k,Cc=Cc: k(h,W,b,C=Cc))
r=ab(fns)
print("fc1 STG bit-identical:", bool(torch.equal(C0,C1)))
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print("%-34s %.1f"%(kk,v["min"]))
# ---- qkv
B,Nt,d,H=2,1029,1024,16; prefix=5
hq=torch.randn(B*Nt,d,device=dev).to(bf); Wq=(torch.randn(3*d,d,device=dev)*d**-0.5).to(bf); bq=torch.randn(3*d,device=dev).to(bf)
ang=torch.randn(1024,32,device=dev); sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
O0=torch.empty(3,B,Nt,H,64,device=dev,dtype=bf); O1=torch.empty_like(O0)
fns={}
kq=gemm_tl.qkv_rope_kernel(B,Nt,d,H,prefix,bm=64,bn=128,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL qkv 64x128x32"]=lambda: kq(hq,Wq,bq,sin,cos)
for stg in (0,1):
  for (bn,wm,wn) in ((128,2,2),(128,2,4),(64,2,2)):
    k=mgb.get(dict(epi=2,wm=wm,wn=wn,bm=64,bn=bn,bk=32,st=2,minb=3,extra=ex0+f" -DSTG={stg} -DNFULL=1")); Oc=[O0,O1][stg]
    fns[f"mg qkv 64x{bn}x32 w{wm}x{wn} STG={stg}"]=(lambda k=k,Oc=Oc: k(hq,Wq,bq,C=Oc,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d,H=H))
r=ab(fns)
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print("%-34s %.1f"%(kk,v["min"]))
# bit identity for last cfg pair
kA=mgb.get(dict(epi=2,wm=2,wn=2,bm=64,bn=128,bk=32,st=2,minb=3,extra=ex0+" -DSTG=0 -DNFULL=1")); kB=mgb.get(dict(epi=2,wm=2,wn=2,bm=64,bn=128,bk=32,st=2,minb=3,extra=ex0+" -DSTG=1 -DNFULL=1"))
kA(hq,Wq,bq,C=O0,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d,H=H); kB(hq,Wq,bq,C=O1,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d,H=H)
print("qkv STG bit-identical:", bool(torch.equal(O0,O1)))
