import sys
sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, tilelang, tilelang.language as T
from bench import time_fn
BF="bfloat16"; F="float32"
def mk(M,N,K,S,bm,bn,bk,st,th):
    Ks=K//S
    @T.prim_func
    def main(A:T.Tensor((M,K),BF),W:T.Tensor((N,K),BF),P:T.Tensor((S,M,N),F)):
        with T.Kernel(T.ceildiv(N,bn),T.ceildiv(M,bm),S,threads=th) as (bx,by,bz):
            a=T.alloc_shared((bm,bk),BF); w=T.alloc_shared((bn,bk),BF); c=T.alloc_fragment((bm,bn),F)
            T.clear(c)
            for k in T.Pipelined(T.ceildiv(Ks,bk),num_stages=st):
                T.copy(A[by*bm,bz*Ks+k*bk],a); T.copy(W[bx*bn,bz*Ks+k*bk],w)
                T.gemm(a,w,c,transpose_B=True)
            T.copy(c,P[bz,by*bm,bx*bn])
    return tilelang.compile(main,out_idx=[2],target="cuda")
M,N,K=2058,1024,4096
a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16); w=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.02
for S,bm,bn,th in [(1,128,128,256),(2,128,128,256),(2,128,64,128),(3,128,64,128),(4,128,64,128),(4,128,128,256)]:
    k=mk(M,N,K,S,bm,bn,32,3,th)
    p=k(a,w); err=(p.sum(0)-(a.float()@w.float().T)).abs().max().item()
    us=time_fn(lambda:k(a,w),warmup=20,iters=50,repeats=3)["gpu_ms_p50"]*1e3
    print(S,bm,bn,th,"gemm-only %.1f us"%us,"err %.3f"%err,flush=True)
