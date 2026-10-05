import sys, struct, subprocess, re
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from impl import gemm_tl
from kt import ab
dev="cuda"; bf=torch.bfloat16
M,N,K=2058,4096,1024
h=torch.randn(M,K,device=dev).to(bf); W=(torch.randn(N,K,device=dev)*K**-0.5).to(bf); b=torch.randn(N,device=dev).to(bf)
k1=gemm_tl.gemm_bias_gelu_kernel(M,N,K,bm=64,bn=96,bk=32,stages=2,threads=128,ws=False,tma=False)
src=k1.get_kernel_source()
src=re.sub(r'extern "C" __global__ void main_kernel\([^;]*\);', '', src)
src=src.replace('extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(','__device__ __forceinline__ void main_body(')
src+="""
struct WArgs { const bfloat16_t* A; const bfloat16_t* Bias; bfloat16_t* C; const bfloat16_t* W; };
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const WArgs p) { main_body(p.A, p.Bias, p.C, p.W); }
"""
open("/root/autodl-tmp/infer-opt-tmp/gemm120/tlsrc.cu","w").write(src)
T="/root/miniconda3/lib/python3.12/site-packages/tilelang"
r=subprocess.run(["/usr/local/cuda/bin/nvcc","-arch=sm_120a","-O3","-std=c++17","-cubin","-Xptxas","-v","-DENABLE_BF16",f"-I{T}/src",f"-I{T}/3rdparty/cutlass/include","-o","/root/autodl-tmp/infer-opt-tmp/gemm120/tlsrc.cubin","/root/autodl-tmp/infer-opt-tmp/gemm120/tlsrc.cu"],capture_output=True,text=True)
print(r.returncode, r.stderr[-400:])
smem=2*(64+96)*32*2
hh=mgb.launcher().load("/root/autodl-tmp/infer-opt-tmp/gemm120/tlsrc.cubin","main_kernel",smem)
C=torch.empty(M,N,device=dev,dtype=bf)
def viasrc():
    blob=struct.pack("=4Q",h.data_ptr(),b.data_ptr(),C.data_ptr(),W.data_ptr())
    mgb.launcher().launch(hh,(N+95)//96,(M+63)//64,1,128,blob,False)
    return C
viasrc(); ref=torch.nn.functional.gelu(torch.nn.functional.linear(h.float(),W.float(),b.float()))
print("viasrc max err", (C.float()-ref).abs().max().item())
fns={"TL native":lambda: k1(h,W,b),"TL src via mgl":viasrc}
k=mgb.get(dict(epi=0,wm=2,wn=2,bm=64,bn=96,bk=32,st=2,minb=3,extra="-DORDER=1 -DUNR=2 -DFRAGALL=1"))
fns["mg best"]=lambda: k(h,W,b,C=C)
r=ab(fns)
for kk,v in r.items(): print(kk,"%.1f"%v["min"])
