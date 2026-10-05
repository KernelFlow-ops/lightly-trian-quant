"""cuBLASLt bias / GELU_AUX_BIAS candidates with complete Linear backward.

Exact erf uses the Lt BIAS epilogue followed by the exact Triton GELU.
The fused tanh epilogue changes numerical behavior and needs separate C/D gates.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch
from .gelu import gelu

_EXT = None
_PLANS = {}
_WORKSPACES = {}

def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault('MAX_JOBS', '6')
        root=Path(__file__).resolve().parent
        cache=Path('/root/autodl-tmp/train-opt-tmp/B/lt_epilogue_build'); cache.mkdir(parents=True,exist_ok=True)
        _EXT=load(name='trainopt_lt_epilogue',sources=[str(root/'lt_epilogue.cu')],build_directory=str(cache),
                  extra_cuda_cflags=['-O3','-lineinfo'],extra_ldflags=['-lcublasLt'])
    return _EXT

def prepare(x,w,tanh):
    ext=extension();key=(x.device.index,x.shape[0],w.shape[0],x.shape[1],x.dtype,tanh)
    if key not in _PLANS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm up cuBLASLt plans before CUDA graph capture')
        _PLANS[key]=ext.LinearPlan(x.shape[0],w.shape[0],x.shape[1],x.dtype==torch.bfloat16,tanh)
    if x.device not in _WORKSPACES:
        _WORKSPACES[x.device]=torch.empty(32*1024*1024,device=x.device,dtype=torch.uint8)
    return _PLANS[key],_WORKSPACES[x.device]

class _LinearGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,tanh):
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        x2=x.reshape(-1,x.shape[-1]).to(dtype).contiguous();weight=w.to(dtype).contiguous()
        bias=torch.zeros(w.shape[0],device=x.device,dtype=dtype) if b is None else b.to(dtype).contiguous()
        plan,workspace=prepare(x2,weight,tanh)
        y=torch.empty((x2.shape[0],weight.shape[0]),device=x.device,dtype=dtype)
        aux=torch.empty_like(y) if tanh else y
        with torch.autocast('cuda',enabled=False):
            plan.run(x2,weight,bias,y,aux,workspace)
            if not tanh:
                y=gelu(aux)
        ctx.save_for_backward(x2,weight,aux)
        ctx.shape=x.shape;ctx.tanh=tanh;ctx.bias=b is not None;ctx.dtypes=(x.dtype,w.dtype,None if b is None else b.dtype)
        return y.reshape(*x.shape[:-1],w.shape[0])
    @staticmethod
    def backward(ctx,dy):
        x,w,aux=ctx.saved_tensors;dy=dy.reshape_as(aux).contiguous()
        with torch.autocast('cuda',enabled=False):
            pre=aux.detach().requires_grad_()
            with torch.enable_grad():
                y=torch.nn.functional.gelu(pre,approximate='tanh' if ctx.tanh else 'none')
                dz=torch.autograd.grad(y,pre,dy)[0]
            dx=(dz@w).reshape(ctx.shape).to(ctx.dtypes[0])
            dw=(dz.t()@x).to(ctx.dtypes[1])
            db=dz.float().sum(0).to(ctx.dtypes[2]) if ctx.bias else None
        return dx,dw,db,None

def linear_gelu(x,module,*,tanh=False):
    if not x.is_cuda:
        return torch.nn.functional.gelu(torch.nn.functional.linear(x,module.weight,module.bias),approximate='tanh' if tanh else 'none')
    return _LinearGELU.apply(x,module.weight,module.bias,tanh)
