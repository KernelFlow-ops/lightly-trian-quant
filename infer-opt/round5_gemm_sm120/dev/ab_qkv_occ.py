import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/impl"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
ex0="-DORDER=1 -DUNR=2 -DFRAGALL=1 -DSTG=1 -DNFULL=1"
B,Nt,d,H=2,1029,1024,16; prefix=5
hq=torch.randn(B*Nt,d,device=dev).to(bf); Wq=(torch.randn(3*d,d,device=dev)*d**-0.5).to(bf); bq=torch.randn(3*d,device=dev).to(bf)
ang=torch.randn(1024,32,device=dev); sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
O=torch.empty(3,B,Nt,H,64,device=dev,dtype=bf)
fns={}
kq=gemm_tl.qkv_rope_kernel(B,Nt,d,H,prefix,bm=64,bn=128,bk=32,stages=2,threads=128,ws=False,tma=False); fns["TL qkv 64x128x32"]=lambda: kq(hq,Wq,bq,sin,cos)
for occ in (1,2,3,0):
  for (bm,bn,wm,wn) in ((64,128,2,2),(64,128,1,4)):

    k=mgb.get(dict(epi=2,wm=wm,wn=wn,bm=bm,bn=bn,bk=32,st=2,minb=3,occ=occ,extra=ex0))
    fns[f"mg {bm}x{bn} w{wm}x{wn} occ{occ}(real {k.occupancy()})"]=(lambda k=k: k(hq,Wq,bq,C=O,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d,H=H))
r=ab(fns,n=10,rounds=5,reps=3)
for kk,v in sorted(r.items(),key=lambda x:x[1]['min']): print("%-36s %.1f"%(kk,v["min"]))
