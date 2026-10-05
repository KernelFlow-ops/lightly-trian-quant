"""FlexAttention fallback uses shared-head compact mask through score_mod."""
import torch
from torch.nn.attention.flex_attention import flex_attention

@torch.compile(fullgraph=True, dynamic=False)
def _compiled(q,k,v,mask,prefix):
    def score_mod(score,b,h,m,n):
        qidx=m.clamp(max=mask.shape[1]-1)
        pidx=(n-prefix).clamp(min=0,max=mask.shape[2]-1)
        allow=(m>=mask.shape[1]) | (n<prefix) | mask[b,qidx,pidx]
        return torch.where(allow,score,-float('inf'))
    return flex_attention(q,k,v,score_mod=score_mod)

@torch.compile(fullgraph=True, dynamic=False)
def _plain(q,k,v): return flex_attention(q,k,v)

def flex(qkv,rope=None,compact_mask=None,prefix=None):
    b,n,_,h,d=qkv.shape
    q,k,v=(z.transpose(1,2) for z in qkv.unbind(2))
    if rope is not None:
        sin,cos=rope;prefix=n-sin.shape[0]
        def rotate(x):
            xf=x[:,:,prefix:].float()
            rot=torch.cat((-xf[...,32:],xf[...,:32]),-1)
            return torch.cat((x[:,:,:prefix].float(),xf*cos+rot*sin),-2).to(x.dtype)
        q,k=rotate(q),rotate(k)
    y=_plain(q,k,v) if compact_mask is None else _compiled(q,k,v,compact_mask,prefix)
    return y.transpose(1,2).reshape(b,n,h*d)
