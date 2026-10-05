"""Exact-erf GELU forward/backward; no tanh approximation."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _gelu(X,Y,N:tl.constexpr,BS:tl.constexpr):
    i=tl.program_id(0)*BS+tl.arange(0,BS)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    y=0.5*x*(1.+libdevice.erf(x*0.7071067811865476))
    tl.store(Y+i,y,i<N)

@triton.jit
def _dgelu(X,DY,DX,N:tl.constexpr,BS:tl.constexpr):
    i=tl.program_id(0)*BS+tl.arange(0,BS)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    dy=tl.load(DY+i,i<N,0).to(tl.float32)
    deriv=0.5*(1.+libdevice.erf(x*0.7071067811865476))+x*0.3989422804014327*tl.exp(-0.5*x*x)
    tl.store(DX+i,dy*deriv,i<N)

class _Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        x=x.contiguous();y=torch.empty_like(x)
        _gelu[(triton.cdiv(x.numel(),512),)](x,y,x.numel(),512,enable_fp_fusion=False)
        ctx.save_for_backward(x);return y
    @staticmethod
    def backward(ctx,dy):
        x,=ctx.saved_tensors;dx=torch.empty_like(x);dy=dy.contiguous()
        _dgelu[(triton.cdiv(x.numel(),512),)](x,dy,dx,x.numel(),512,enable_fp_fusion=False)
        return dx

def gelu(x): return _Gelu.apply(x)
