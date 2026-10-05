"""bf16 shadow-weight Linear preserving original fp32 Parameter gradients."""
from __future__ import annotations
import re
import torch

SHADOW_PARAMS=(r'model\.backbone\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)\.weight',)

class _ShadowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,shadow):
        xb=x.to(torch.bfloat16)
        y=torch.nn.functional.linear(xb,shadow,None if bias is None else bias.to(torch.bfloat16))
        ctx.save_for_backward(xb,shadow)
        ctx.x_dtype=x.dtype;ctx.w_dtype=weight.dtype;ctx.bias=bias is not None
        return y
    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors
        # cuBLAS bf16 GEMM with fp32 accumulator/output avoids the bf16 wgrad
        # rounding introduced by an ordinary autocast Linear backward.
        with torch.autocast('cuda',enabled=False):
            dy2=dy.reshape(-1,dy.shape[-1]).to(torch.bfloat16)
            x2=x.reshape(-1,x.shape[-1])
            dx=torch.mm(dy2,w).reshape_as(x).to(ctx.x_dtype)
            dw=torch.mm(dy2.t(),x2,out_dtype=torch.float32).to(ctx.w_dtype)
            db=dy2.float().sum(0) if ctx.bias else None
        return dx,dw,db,None

def linear(x,module,ctx,names):
    name=names[id(module.weight)]
    bias=module.bias
    if bias is not None and hasattr(module,'bias_mask'):
        bias=bias*module.bias_mask.to(bias.dtype)
    if not torch.is_autocast_enabled('cuda'):
        return torch.nn.functional.linear(x,module.weight,bias)
    return _ShadowLinear.apply(x,module.weight,bias,ctx.shadow[name])

def setup(ctx):
    names={id(p):name for name,p in ctx.train_model.named_parameters()}
    matched={name:p for name,p in ctx.train_model.named_parameters()
             if any(re.fullmatch(pattern,name) for pattern in SHADOW_PARAMS)}
    if not matched: raise RuntimeError('No shadow parameters matched the published full-name contract')
    for name,p in matched.items():
        ctx.shadow.setdefault(name,p.detach().to(torch.bfloat16).contiguous())
    if not ctx.meta.get('round5_shadow_hook'):
        @torch.no_grad()
        def refresh(c):
            if c.meta.get('shadow_writer')=='round7_optimizer': return
            torch._foreach_copy_([c.shadow[name] for name in matched],list(matched.values()))
        ctx.post_opt_hooks.append(refresh)
        ctx.meta['round5_shadow_hook']=True
    ctx.meta['shadow_params']=SHADOW_PARAMS
    if not ctx.meta.get('round5_shadow_modules'):
        for module in ctx.model.backbone.blocks:
            for lin in (module.attn.qkv,module.attn.proj,module.mlp.fc1,module.mlp.fc2):
                def forward(x,lin=lin): return linear(x,lin,ctx,names)
                lin.forward=forward
        ctx.meta['round5_shadow_modules']=True
    return names
