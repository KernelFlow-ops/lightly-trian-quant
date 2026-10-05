"""fp32 residual + trainable LayerScale + LN, with complete backward.

Normalized output is rounded bf16 at the same boundary as the following
autocast Linear. Residual stream always remains fp32. Parameter reductions
are deterministic two-stage fp32 reductions.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd(X,B,G,W,BIAS,R,Y,MEAN,RSTD,M:tl.constexpr,D:tl.constexpr,
         EPS:tl.constexpr,BD:tl.constexpr,BRANCH:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,BD)
    x=tl.load(X+row*D+d,d<D,0).to(tl.float32)
    if BRANCH:
        b=tl.load(B+row*D+d,d<D,0).to(tl.float32)
        gamma=tl.load(G+d,d<D,0)
        x=x+b*gamma
    mean=tl.sum(x,0)/D
    centered=tl.where(d<D,x-mean,0.)
    var=tl.sum(centered*centered,0)/D
    rs=tl.rsqrt(var+EPS)
    w=tl.load(W+d,d<D,0);bias=tl.load(BIAS+d,d<D,0)
    y=centered*rs*w+bias
    if BRANCH:tl.store(R+row*D+d,x,d<D)
    tl.store(Y+row*D+d,y,d<D)
    tl.store(MEAN+row,mean);tl.store(RSTD+row,rs)

@triton.jit
def _bwd(R,B,G,W,MEAN,RSTD,DR,DY,DX,DB,PART,
         M:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BRANCH:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    d=tl.arange(0,BD)
    valid=(row[:,None]<M)&(d[None,:]<D)
    r=tl.load(R+row[:,None]*D+d[None,:],valid,0)
    mean=tl.load(MEAN+row,row<M,0);rs=tl.load(RSTD+row,row<M,0)
    normalized=(r-mean[:,None])*rs[:,None]
    dy=tl.load(DY+row[:,None]*D+d[None,:],valid,0).to(tl.float32)
    w=tl.load(W+d,d<D,0)
    a=dy*w[None,:]
    sum1=tl.sum(a,1)/D
    sum2=tl.sum(a*normalized,1)/D
    dl=(a-sum1[:,None]-normalized*sum2[:,None])*rs[:,None]
    dr=tl.load(DR+row[:,None]*D+d[None,:],valid,0).to(tl.float32)
    dx=dl+dr
    tl.store(DX+row[:,None]*D+d[None,:],dx,valid)
    partial_w=tl.sum(tl.where(valid,dy*normalized,0),0)
    partial_bias=tl.sum(tl.where(valid,dy,0),0)
    group=tl.program_id(0)
    tl.store(PART+(group*3+0)*D+d,partial_w,d<D)
    tl.store(PART+(group*3+1)*D+d,partial_bias,d<D)
    if BRANCH:
        b=tl.load(B+row[:,None]*D+d[None,:],valid,0).to(tl.float32)
        gamma=tl.load(G+d,d<D,0)
        tl.store(DB+row[:,None]*D+d[None,:],dx*gamma[None,:],valid)
        tl.store(PART+(group*3+2)*D+d,tl.sum(tl.where(valid,dx*b,0),0),d<D)
    else:
        tl.store(PART+(group*3+2)*D+d,0.,d<D)

class _ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,branch,gamma,weight,bias,eps):
        assert x.dtype==torch.float32 and x.is_contiguous()
        d=x.shape[-1];m=x.numel()//d
        # Actual Fabric call paths may run outside autocast. Preserve fp32
        # there; round bf16 only when the following Linear will autocast.
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        has=branch is not None
        r=torch.empty_like(x) if has else x
        y=torch.empty_like(x,dtype=dtype)
        mean=torch.empty(m,device=x.device);rs=torch.empty_like(mean)
        b=x if branch is None else branch.contiguous()
        g=weight if gamma is None else gamma
        _fwd[(m,)](x,b,g,weight,bias,r,y,mean,rs,m,d,eps,triton.next_power_of_2(d),has,
                   num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(r,b,g,weight,mean,rs)
        ctx.has=has
        ctx.dtype=dtype
        ctx.set_materialize_grads(False)
        return r,y
    @staticmethod
    def backward(ctx,dr,dy):
        r,b,g,w,mean,rs=ctx.saved_tensors
        d=r.shape[-1];m=r.numel()//d
        dr=torch.zeros_like(r) if dr is None else dr.contiguous()
        dy=torch.zeros_like(r,dtype=ctx.dtype) if dy is None else dy.contiguous()
        dx=torch.empty_like(r);db=torch.empty_like(b) if ctx.has else r
        groups=triton.cdiv(m,8)
        partial=torch.empty((groups,3,d),device=r.device,dtype=torch.float32)
        _bwd[(groups,)](r,b,g,w,mean,rs,dr,dy,dx,db,partial,m,d,triton.next_power_of_2(d),8,ctx.has,
                       num_warps=8,enable_fp_fusion=False)
        reduced=partial.sum(0)
        return dx,db if ctx.has else None,reduced[2] if ctx.has else None,reduced[0],reduced[1],None

def residual_norm(x,branch,gamma,norm):
    return _ResidualNorm.apply(x,branch,gamma,norm.weight,norm.bias,norm.eps)

def norm(x,module):
    return residual_norm(x,None,None,module)[1]
