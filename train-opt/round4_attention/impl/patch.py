"""Training attention patch; no modules/Parameters/state keys are replaced."""
from __future__ import annotations
import torch
from round4_attention.impl.attention import attention, reference
from round4_attention.impl.mask import build_compact
from round4_attention.impl.flex import flex
from round4_attention.impl.rope import sdpa_rope
from round4_attention.impl.cuda_attention import attention as cuda_attention

NAME = 'round4_attention'
REQUIRES = ()
CONFLICTS = ()

def apply(ctx, variant='sdpa_rope', **kw):
    supported=('triton','triton_rope','triton_rope_mask','triton_fp32','sdpa_compact','sdpa_rope','flex','cuda_ref','tma')
    if variant not in supported:
        raise ValueError(f'{NAME}: supported variants are {supported}; copied TL is forward-only')
    if ctx.meta.get(NAME) == variant:
        return {'ops':['attn','build_mask'],'desc':variant}
    base_build=ctx.ops.build_mask
    def compact_build(c,j,mask_logits,grid_size,n_tokens):
        if grid_size!=(32,32) or mask_logits.shape[-2:]!=(128,128):
            full=base_build(c,j,mask_logits,grid_size,n_tokens)
            if full is None:return None
            if full.ndim==4:full=full[:,0]
            prefix=c.model.num_queries+1+c.model.backbone.n_storage_tokens
            return full[:,:c.model.num_queries,prefix:].contiguous()
        prob=c.model.attn_mask_probs[j]
        disabled=None
        # Use the exact original conditional until r1 publishes a synchronized
        # host probability. rand invocation kind/shape/name are unchanged.
        draw_disable = c.host_probs[j] < 1 if c.meta.get('round1_annealing_host') else bool(prob < 1)
        if draw_disable:
            disabled=c.rng.rand(f'mask/j{j}',tuple(mask_logits.shape[:2]),device=mask_logits.device)>prob
        return build_compact(mask_logits,disabled)

    def attn(c,block,x_normed,rope,mask):
        module=block.attn if hasattr(block,'attn') else block
        b,n,_=x_normed.shape
        qkv=module.qkv(x_normed).reshape(b,n,3,module.num_heads,64)
        prefix=n-(0 if rope is None else rope[0].shape[0])
        if variant=='sdpa_rope':
            return module.proj_drop(module.proj(sdpa_rope(qkv,rope,mask)))
        if variant=='tma':
            from .tma_attention import attention as tma_attention
            y=tma_attention(qkv,rope,mask,prefix)
            return module.proj_drop(module.proj(y))
        if variant == 'cuda_ref':
            if mask is not None and mask.ndim==4: mask=mask[:,0]
            if mask is not None and mask.shape[1] != c.model.num_queries:
                mask=mask[:,:c.model.num_queries,prefix:].contiguous()
            y=cuda_attention(qkv,rope,mask,prefix)
            return module.proj_drop(module.proj(y))
        if mask is not None and mask.ndim==4: mask=mask[:,0]
        if mask is not None and mask.shape[1] != c.model.num_queries:
            mask=mask[:,:c.model.num_queries,prefix:].contiguous()
        if variant in ('sdpa_compact','flex') or (qkv.dtype != torch.bfloat16 and variant!='triton_fp32'):
            if variant=='flex':
                y=flex(qkv,rope,mask,prefix)
            else: y=reference(qkv,rope,mask,prefix)
        elif variant=='triton':
            q,k,v=(z.transpose(1,2) for z in qkv.unbind(2))
            if rope is not None: q,k=module.apply_rope(q,k,rope)
            packed=torch.stack((q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)),2)
            y=attention(packed,None,mask,prefix)
        else:
            y=attention(qkv,rope,mask,prefix)
        return module.proj_drop(module.proj(y))
    ctx.ops.attn=attn
    if variant!='sdpa_rope':ctx.ops.build_mask=compact_build
    ctx.meta[NAME]=variant
    return {'ops':['attn','build_mask'],'desc':f'{variant}: deterministic packed QKV fwd+bwd'}
