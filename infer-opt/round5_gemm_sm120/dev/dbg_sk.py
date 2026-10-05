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
k=mgb.get(cfg); C=torch.empty_like(R)
k(h,W,b,g,R,C)
err=(C-ref).abs()
tm,tn=33,16
bad=[]
for i in range(tm):
    for j in range(tn):
        e=err[i*64:(i+1)*64, j*64:(j+1)*64].max().item()
        if e>0.05: bad.append((i*16+j,e))
print("P=",min(528*128,170*k.occupancy()),"bad tiles",len(bad),"of",tm*tn)
print(bad[:40])
IT=528*128; P=510
for t,_ in bad[:10]:
    cs=[c for c in range(P) if (IT*c)//P < (t+1)*128 and (IT*(c+1))//P > t*128]
    print("tile",t,"CTAs",cs,[( (IT*c)//P - t*128, (IT*(c+1))//P - t*128) for c in cs])

# ---- inspect the partial slot of tile 2 (slot 0): expected = h[m-rows, 0:288] @ W[cols, 0:288]^T
ws=next(iter(mgb._WS.values()))[0]
sk_maxc=2; BM=BN=64; NT=128; MI=2; NJ=4; WM=WN=2
t=2; tile_m=t//16; tile_n=t%16
slot=ws[(t*sk_maxc+0)*4096:(t*sk_maxc+0)*4096+4096].view(MI*NJ,NT,4).cpu()
hh=h[tile_m*64:(tile_m+1)*64,0:9*32].float(); ww=W[tile_n*64:(tile_n+1)*64,0:9*32].float()
part=hh@ww.T
maxe=0
import itertools
for mi,nj,tid,e in itertools.product(range(MI),range(NJ),range(NT),range(4)):
    warp=tid>>5; lane=tid&31; wm=warp//WN; wn=warp%WN
    row=wm*32+mi*16+(lane>>2)+(e>>1)*8; col=wn*32+nj*8+(lane&3)*2+(e&1)
    v=slot[mi*NJ+nj,tid,e].item()
    maxe=max(maxe,abs(v-part[row,col].item()))
print("partial slot0 of tile 2 vs expected: max err",maxe)
d=(C-ref)[tile_m*64:(tile_m+1)*64, tile_n*64:(tile_n+1)*64]
print("tile2: |C-ref| max", d.abs().max().item(), " | C-ref + part| max", (d+part).abs().max().item(), " | C-ref - part| max", (d-part).abs().max().item(), "|part| max", part.abs().max().item())
# own-range only
hh2=h[tile_m*64:(tile_m+1)*64,9*32:].float(); ww2=W[tile_n*64:(tile_n+1)*64,9*32:].float()
own=hh2@ww2.T
print(" C - own : max diff to expected-part?", ((C[tile_m*64:(tile_m+1)*64, tile_n*64:(tile_n+1)*64]-b.float()[tile_n*64:(tile_n+1)*64][None]) - own).abs().max().item())
e=d.abs()
print("error map tile 2 (max per 8x8 block):")
for i in range(8):
    print(" ".join("%5.2f"%e[i*8:(i+1)*8, j*8:(j+1)*8].max().item() for j in range(8)))
# tile 1 (good) and tile 3 (good) for reference
for tt in (1,3):
    tm_,tn_=tt//16,tt%16
    print("tile",tt,"max err",(C-ref)[tm_*64:(tm_+1)*64,tn_*64:(tn_+1)*64].abs().max().item())
