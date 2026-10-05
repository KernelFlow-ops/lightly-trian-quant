from __future__ import annotations
from round5_block_fusion.impl.fused_norm import norm, residual_norm
from round5_block_fusion.impl.gelu import gelu
from round5_block_fusion.impl.shadow import setup,linear

NAME='round5_block_fusion'
REQUIRES=()
CONFLICTS=()

def apply(ctx,variant='ln_fuse',**kw):
    if variant not in ('ln_fuse','gelu','shadow','all','lt_erf','lt_tanh'):
        raise ValueError('Available: ln_fuse, gelu, shadow, all, lt_erf, lt_tanh')
    if ctx.meta.get(NAME)==variant: return {'ops':['block'],'desc':variant}
    names=setup(ctx) if variant in ('shadow','all') else None
    if variant in ('lt_erf','lt_tanh'):
        from .lt_epilogue import linear_gelu
        for blk in ctx.model.backbone.blocks:
            fc1=blk.mlp.fc1
            fc1.forward=lambda x, module=fc1: linear_gelu(x,module,tanh=variant=='lt_tanh')
            blk.mlp.act.forward=lambda x: x
    def block(c,i,blk,x,rope,mask):
        if x.dtype != __import__('torch').float32:
            raise TypeError('Block fusion requires the original fp32 residual stream')
        use_ln=variant in ('ln_fuse','all','lt_erf','lt_tanh')
        x_normed=norm(x,blk.norm1) if use_ln else blk.norm1(x)
        a=c.ops.attn(c,blk,x_normed,rope,mask)
        if use_ln:
            x,yn=residual_norm(x,a,blk.ls1.gamma,blk.norm2)
        else:
            x=x+blk.ls1(a);yn=blk.norm2(x)
        h=blk.mlp.fc1(yn)
        h=gelu(h) if variant in ('gelu','all') else blk.mlp.act(h)
        h=blk.mlp.drop(h)
        h=blk.mlp.fc2(h)
        h=blk.mlp.drop(h)
        return x+blk.ls2(h)
    ctx.ops.block=block;ctx.meta[NAME]=variant
    return {'ops':['block'],'desc':f'{variant}: fp32 residual, trainable LayerScale, complete backward'}
