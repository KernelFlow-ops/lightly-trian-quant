"""Exact-erf GELU -> depthwise 3x3 -> LayerNorm with complete gradients.

NHWC storage is used internally. FP32 residual/parameter storage is preserved;
GELU, convolution, and gradient cast boundaries match the separate operators.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _gelu(x):
    return .5 * x * (1. + libdevice.erf(x * .7071067811865476))

@triton.jit
def _forward(X,W,G,B,Y,Z,MEAN,RS,H:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,
             EPS:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0); c=tl.arange(0,D); valid=c<C
    batch=row//(H*WIDTH); y=(row//WIDTH)%H; x=row%WIDTH
    z=tl.full((D,),0.,tl.float32)
    for ky in tl.static_range(3):
        for kx in tl.static_range(3):
            yy=y+ky-1; xx=x+kx-1
            present=(yy>=0)&(yy<H)&(xx>=0)&(xx<WIDTH)
            value=tl.load(X+((batch*H+yy)*WIDTH+xx)*C+c, valid&present,0).to(tl.float32)
            act=_gelu(value).to(X.dtype.element_ty).to(tl.float32)
            w=tl.load(W+c*9+ky*3+kx,valid,0).to(tl.float32)
            z+=act*w
    z=z.to(X.dtype.element_ty).to(tl.float32)
    mean=tl.sum(tl.where(valid,z,0),0)/C
    centered=tl.where(valid,z-mean,0)
    rs=tl.rsqrt(tl.sum(centered*centered,0)/C+EPS)
    gamma=tl.load(G+c,valid,0).to(tl.float32); bias=tl.load(B+c,valid,0).to(tl.float32)
    out=centered*rs*gamma+bias
    tl.store(Z+row*C+c,z,valid);tl.store(Y+row*C+c,out,valid)
    tl.store(MEAN+row,mean);tl.store(RS+row,rs)

@triton.jit
def _norm_backward(Z,G,MEAN,RS,DY,DZ,PART,M:tl.constexpr,C:tl.constexpr,D:tl.constexpr,R:tl.constexpr):
    r=tl.program_id(0)*R+tl.arange(0,R); c=tl.arange(0,D)
    valid=(r[:,None]<M)&(c[None,:]<C)
    z=tl.load(Z+r[:,None]*C+c[None,:],valid,0).to(tl.float32)
    mean=tl.load(MEAN+r,r<M,0);rs=tl.load(RS+r,r<M,0)
    h=tl.where(valid,(z-mean[:,None])*rs[:,None],0)
    dy=tl.load(DY+r[:,None]*C+c[None,:],valid,0).to(tl.float32)
    g=tl.load(G+c,c<C,0).to(tl.float32);a=dy*g[None,:]
    dz=(a-tl.sum(a,1)[:,None]/C-h*tl.sum(a*h,1)[:,None]/C)*rs[:,None]
    tl.store(DZ+r[:,None]*C+c[None,:],dz,valid)
    tl.store(PART+(tl.program_id(0)*2)*C+c,tl.sum(dy*h,0),c<C)
    tl.store(PART+(tl.program_id(0)*2+1)*C+c,tl.sum(dy,0),c<C)

@triton.jit
def _dx(X,W,DZ,DX,M:tl.constexpr,H:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//C;c=i%C
    batch=row//(H*WIDTH); y=(row//WIDTH)%H;x=row%WIDTH; dx=tl.full((BLOCK,),0.,tl.float32)
    for ky in tl.static_range(3):
        for kx in tl.static_range(3):
            yy=y-ky+1;xx=x-kx+1
            valid=(row<M)&(yy>=0)&(yy<H)&(xx>=0)&(xx<WIDTH)
            dz=tl.load(DZ+((batch*H+yy)*WIDTH+xx)*C+c,valid,0).to(tl.float32)
            w=tl.load(W+c*9+ky*3+kx,row<M,0).to(tl.float32)
            dx+=dz*w
    dx=dx.to(X.dtype.element_ty).to(tl.float32)
    value=tl.load(X+i,row<M,0).to(tl.float32)
    derivative=.5*(1.+libdevice.erf(value*.7071067811865476))+value*.3989422804014327*tl.exp(-.5*value*value)
    tl.store(DX+i,dx*derivative,row<M)

@triton.jit
def _dw(X,DZ,PART,M:tl.constexpr,H:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,R:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0)*R+tl.arange(0,R);c=tl.program_id(1)*BC+tl.arange(0,BC)
    ky=tl.program_id(2)//3;kx=tl.program_id(2)%3
    batch=row//(H*WIDTH);y=(row//WIDTH)%H;x=row%WIDTH; yy=y+ky-1;xx=x+kx-1
    valid=(row[:,None]<M)&(c[None,:]<C)&(yy[:,None]>=0)&(yy[:,None]<H)&(xx[:,None]>=0)&(xx[:,None]<WIDTH)
    value=tl.load(X+((batch[:,None]*H+yy[:,None])*WIDTH+xx[:,None])*C+c[None,:],valid,0).to(tl.float32)
    dz=tl.load(DZ+row[:,None]*C+c[None,:],(row[:,None]<M)&(c[None,:]<C),0).to(tl.float32)
    a=_gelu(value).to(X.dtype.element_ty).to(tl.float32)
    val=tl.sum(a*dz,0)
    tl.store(PART+(tl.program_id(0)*9+ky*3+kx)*C+c,val,c<C)

class _FusedDWNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b,eps):
        batch,h,width,c=x.shape; m=batch*h*width
        y=torch.empty_like(x);z=torch.empty_like(x);mean=torch.empty(m,device=x.device);rs=torch.empty_like(mean)
        _forward[(m,)](x,w,g,b,y,z,mean,rs,h,width,c,eps,triton.next_power_of_2(c),enable_fp_fusion=False)
        ctx.save_for_backward(x,w,g,z,mean,rs);ctx.eps=eps
        return y
    @staticmethod
    def backward(ctx,dy):
        x,w,g,z,mean,rs=ctx.saved_tensors;batch,h,width,c=x.shape;m=batch*h*width
        dy=dy.contiguous();dz=torch.empty_like(x);dx=torch.empty_like(x)
        part=torch.empty((triton.cdiv(m,8),2,c),device=x.device)
        _norm_backward[(triton.cdiv(m,8),)](z,g,mean,rs,dy,dz,part,m,c,triton.next_power_of_2(c),8,enable_fp_fusion=False)
        gb=part.sum(0)
        _dx[(triton.cdiv(x.numel(),256),)](x,w,dz,dx,m,h,width,c,256,enable_fp_fusion=False)
        wp=torch.empty((triton.cdiv(m,128),9,c),device=x.device)
        _dw[(triton.cdiv(m,128),triton.cdiv(c,32),9)](x,dz,wp,m,h,width,c,128,32,enable_fp_fusion=False)
        dw=wp.sum(0).t().contiguous().reshape_as(w)
        return dx,dw.to(w.dtype),gb[0].to(g.dtype),gb[1].to(g.dtype),None

def gelu_depthwise_norm(x,conv,norm):
    if not x.is_cuda:
        return norm(conv(torch.nn.functional.gelu(x)))
    nhwc=x.permute(0,2,3,1).contiguous()
    return _FusedDWNorm.apply(nhwc,conv.weight,norm.weight,norm.bias,norm.eps).permute(0,3,1,2)
