"""Kernel-level A/B of the LN-fused v2 kernels vs the v1 pair (K1 LayerNorm kernel + GEMM), ViT-L M=2058, interleaved graph replays."""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch, mgb
import ops as r5
from kt import ab
dev="cuda"; bf=torch.bfloat16
B,Nt,d,ff,H=2,1029,1024,4096,16; M=B*Nt; prefix=5
r=lambda *s: (torch.randn(*s,device=dev)*0.03).to(bf)
w=dict(Wqkv=r(3*d,d),bqkv=r(3*d),Wproj=r(d,d),bproj=r(d),W1=r(ff,d),b1=r(ff),W2=r(d,ff),b2=r(d),g1=torch.rand(d,device=dev),g2=torch.rand(d,device=dev))
lw=torch.rand(d,device=dev)+0.5; lb=torch.randn(d,device=dev)*0.1
Wqf,csq,bq32=r5._fold_ln(w["Wqkv"],lw,lb,w["bqkv"].float()); W1f,cs1,b132=r5._fold_ln(w["W1"],lw,lb,w["b1"].float())
x=(torch.randn(B,Nt,d,device=dev)*2)
ang=torch.randn(1024,32,device=dev); sin,cos=torch.sin(ang).repeat(1,2).float().contiguous(),torch.cos(ang).repeat(1,2).float().contiguous()
st=r5.v2_enter(x)
a_attn=r(M,d); a_fc1=r(M,ff)
fns={}
# --- qkv
fns["qkv v1: LN + qkv(TL/mg tuned)"]=lambda: r5.qkv_rope(r5.add_layernorm(x,None,lw,lb)[1],w["Wqkv"],w["bqkv"],sin,cos,prefix,H)
kq,cq=r5._mg_kernel("qkv",3*d,d,M,lnf=1); O=torch.empty(3,B,Nt,H,64,device=dev,dtype=bf)
fns["qkv v2: LN-fused mg"]=lambda: kq(st.xb,Wqf,bq32,C=O,cs=csq,stats_in=st.stats,np_in=st.np_,pcols_in=st.pcols,sin=sin,cos=cos,Ntok=Nt,prefix=prefix,d=d,H=H)
# --- fc1
fns["fc1 v1: LN + fc1"]=lambda: r5.gemm_bias_gelu(r5.add_layernorm(x,None,lw,lb)[1],w["W1"],w["b1"])
k1,c1=r5._mg_kernel("fc1",ff,d,M,lnf=1); A1=torch.empty(M,ff,device=dev,dtype=bf)
fns["fc1 v2: LN-fused mg"]=lambda: k1(st.xb,W1f,b132,C=A1,cs=cs1,stats_in=st.stats,np_in=st.np_,pcols_in=st.pcols)
# --- proj
fns["proj v1"]=lambda: r5.gemm_bias_residual(a_attn,w["Wproj"],w["bproj"],x,w["g1"])
kp,cp=r5._mg_kernel("proj",d,d,M,stats=1,rf32=1); X1=torch.empty_like(x); XB=torch.empty(M,d,device=dev,dtype=bf); ST=torch.empty(M,-(-d//cp["bn"]),2,device=dev)
fns["proj v2: +stats mg"]=lambda: kp(a_attn,w["Wproj"],w["bproj"],w["g1"],x,X1,Cb=XB,stats_out=ST)
# --- fc2
fns["fc2 v1"]=lambda: r5.gemm_bias_residual(a_fc1,w["W2"],w["b2"],x,w["g2"])
k2,c2=r5._mg_kernel("fc2",d,ff,M,stats=1,rf32=1); X2=torch.empty_like(x); ST2=torch.empty(M,-(-d//c2["bn"]),2,device=dev)
fns["fc2 v2: +stats mg"]=lambda: k2(a_fc1,w["W2"],w["b2"],w["g2"],x,X2,Cb=XB,stats_out=ST2)
print("configs:",{"qkv":cq,"fc1":c1,"proj":cp,"fc2":c2})
res=ab(fns,n=10,rounds=7,reps=3)
for k,v in res.items(): print("%-34s %.1f us"%(k,v["min"]))
