"""Packed RoPE fwd+bwd in one launch each; original fp32 rotation order."""
import torch
import triton
import triton.language as tl

@triton.jit
def _rotate(X,SIN,COS,Y,TOTAL:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
            PREFIX:tl.constexpr,INVERSE:tl.constexpr,BS:tl.constexpr):
    i=tl.program_id(0)*BS+tl.arange(0,BS)
    col=i%64;which=(i//(H*64))%3;tok=(i//(3*H*64))%N
    active=(tok>=PREFIX)&(which<2)&(i<TOTAL)
    x=tl.load(X+i,i<TOTAL,0).to(tl.float32)
    oth=tl.load(X+i-col+(col+32)%64,active,0).to(tl.float32)
    pos=(tok-PREFIX)*64+col
    sin=tl.load(SIN+pos,active,0);cos=tl.load(COS+pos,active,1)
    if INVERSE: sign=tl.where(col<32,1.,-1.)
    else: sign=tl.where(col<32,-1.,1.)
    y=x*cos+oth*sign*sin
    tl.store(Y+i,y,i<TOTAL)

class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,sin,cos):
        b,n,_,h,d=x.shape
        assert d==64 and x.is_contiguous() and sin.dtype==torch.float32
        prefix=n-sin.shape[0];y=torch.empty_like(x)
        _rotate[(triton.cdiv(x.numel(),512),)](x,sin,cos,y,x.numel(),n,h,prefix,False,512,enable_fp_fusion=False)
        ctx.save_for_backward(sin,cos);ctx.shape=x.shape;ctx.prefix=prefix
        return y
    @staticmethod
    def backward(ctx,dy):
        sin,cos=ctx.saved_tensors;dy=dy.contiguous();dx=torch.empty_like(dy)
        b,n,_,h,d=ctx.shape
        _rotate[(triton.cdiv(dx.numel(),512),)](dy,sin,cos,dx,dx.numel(),n,h,ctx.prefix,True,512,enable_fp_fusion=False)
        return dx,None,None

def rope_qkv(qkv,rope):
    return qkv if rope is None else _Rope.apply(qkv,*rope)

def sdpa_rope(qkv,rope=None,mask=None):
    x=rope_qkv(qkv,rope)
    b,n,_,h,d=x.shape
    q,k,v=(z.transpose(1,2) for z in x.unbind(2))
    if mask is not None and mask.ndim==3:mask=mask[:,None]
    return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(b,n,h*d)
