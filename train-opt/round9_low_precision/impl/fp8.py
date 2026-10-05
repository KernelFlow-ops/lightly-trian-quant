"""Real E4M3/E5M2 training Linear, current/delayed scaling, fp32 masters.

The quantizer uses straight-through derivatives explicitly in the Linear
backward; this is a changed numerical training algorithm and requires C/D
validation before promotion. Every fwd/dgrad/wgrad GEMM uses _scaled_mm.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _quant(X,Q,S,ROWS:tl.constexpr,K:tl.constexpr,STRIDE0:tl.constexpr,
           STRIDE1:tl.constexpr,BD:tl.constexpr,LIMIT:tl.constexpr,DELAYED:tl.constexpr):
    r=tl.program_id(0);c=tl.arange(0,BD)
    x=tl.load(X+r*STRIDE0+c*STRIDE1,c<K,0).to(tl.float32)
    if DELAYED:s=tl.load(S+r)
    else:
        s=tl.maximum(tl.max(tl.abs(x),0)/LIMIT,1e-30)
        tl.store(S+r,s)
    q=tl.minimum(tl.maximum(x/s,-LIMIT),LIMIT)
    tl.store(Q+r*K+c,q,c<K)

def quantize_rows(x,dtype=torch.float8_e4m3fn,scale=None):
    assert x.ndim==2
    rows,k=x.shape
    q=torch.empty((rows,k),device=x.device,dtype=dtype)
    delayed=scale is not None
    if scale is None:scale=torch.empty((rows,1),device=x.device,dtype=torch.float32)
    limit=448. if dtype==torch.float8_e4m3fn else 57344.
    _quant[(rows,)](x,q,scale,rows,k,x.stride(0),x.stride(1),triton.next_power_of_2(k),limit,delayed,
                    num_warps=8 if k>2048 else 4)
    return q,scale

def _mm(a,b,sa,sb,bias=None,dtype=torch.bfloat16):
    return torch._scaled_mm(a,b.t(),scale_a=sa,scale_b=sb.t().contiguous(),
                           bias=bias,out_dtype=dtype,use_fast_accum=False)

class ScalingState:
    def __init__(self,delayed=False,history=16,tensorwise=False):
        if history < 1: raise ValueError('scaling history must be positive')
        self.delayed=delayed;self.history=history;self.values={};self.tensorwise=tensorwise
    @torch.no_grad()
    def quantize(self,key,x,dtype=torch.float8_e4m3fn):
        if not self.delayed and not self.tensorwise:return quantize_rows(x,dtype)
        limit=448. if dtype==torch.float8_e4m3fn else 57344.
        current=(x.float().abs().amax(1,keepdim=True)/limit).clamp_min(1e-30)
        if self.tensorwise:current=current.amax().reshape(1,1).expand(x.shape[0],1).contiguous()
        if not self.delayed:return quantize_rows(x,dtype,current)
        entry=self.values.get(key)
        if entry is None or entry[0].shape!=current.shape:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('warm up delayed scaling before capture')
            history=current.unsqueeze(0).expand(self.history,*current.shape).clone()
            entry=(current.clone(),history,torch.zeros(1,device=x.device,dtype=torch.long))
            self.values[key]=entry
        previous,hist,index=entry
        q,s=quantize_rows(x,dtype,previous)
        saved_scale=s.clone()
        hist.index_copy_(0,index,current.unsqueeze(0))
        previous.copy_(hist.amax(0))
        index.add_(1).remainder_(self.history)
        return q,saved_scale

    def state_dict(self):
        return {k:tuple(v.detach().cpu().clone() for v in tensors) for k,tensors in self.values.items()}

    def load_state_dict(self,values):
        for key,tensors in values.items():
            if key not in self.values:raise ValueError('initialize scaling buffers before restoring them')
            for actual,saved in zip(self.values[key],tensors):actual.copy_(saved)

class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,state):
        x2=x.reshape(-1,x.shape[-1]).to(torch.bfloat16)
        qx,sx=state.quantize('x',x2)
        qw,sw=state.quantize('w',weight)
        y=_mm(qx,qw,sx,sw,None if bias is None else bias.bfloat16())
        ctx.save_for_backward(x2,weight)
        ctx.state=state;ctx.xshape=x.shape;ctx.xdtype=x.dtype;ctx.bias=bias is not None
        return y.reshape(*x.shape[:-1],weight.shape[0])
    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors;state=ctx.state
        dy=dy.reshape(-1,dy.shape[-1]).to(torch.bfloat16)
        qdy,sdy=state.quantize('dy',dy,torch.float8_e4m3fn)
        qwt,swt=state.quantize('wt',w.t())
        dx=_mm(qdy,qwt,sdy,swt).reshape(ctx.xshape).to(ctx.xdtype)
        pad=(-x.shape[0])%getattr(state,'reduction_multiple',16)
        if pad:
            x=torch.nn.functional.pad(x,(0,0,0,pad))
            dy=torch.nn.functional.pad(dy,(0,0,0,pad))
        qdyt,sdyt=state.quantize('dyt',dy.t(),torch.float8_e4m3fn)
        qxt,sxt=state.quantize('xt',x.t())
        # sm_120 _scaled_mm accepts bf16/fp16 accumulator outputs only;
        # cast the result to fp32 before handing it to the fp32 master.
        dw=_mm(qdyt,qxt,sdyt,sxt,dtype=torch.bfloat16).float()
        db=dy.float().sum(0) if ctx.bias else None
        return dx,dw,db,None

def linear(x,module,state):
    bias=module.bias
    if bias is not None and hasattr(module,'bias_mask'):
        bias=bias*module.bias_mask.to(bias.dtype)
    with torch.autocast('cuda',enabled=False):
        return _FP8Linear.apply(x,module.weight,bias,state)
