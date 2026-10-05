from __future__ import annotations
import argparse
import math
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROUND))
from impl.attention import attention, reference
from impl.mask import build_compact

def metrics(x,y):
    a,b=x.float().flatten(),y.float().flatten()
    return {"max_abs":(a-b).abs().max().item(),
            "max_rel":((a-b).abs()/b.abs().clamp_min(1e-4)).max().item(),
            "cos":F.cosine_similarity(a,b,dim=0).item()}

@torch.no_grad()
def reference_lse(x,rope,compact,prefix):
    q,k,_=(v.transpose(1,2) for v in x.unbind(2))
    if rope is not None:
        sin,cos=rope
        def rotate(z):
            f=z[:,:,prefix:].float();r=torch.cat((-f[...,32:],f[...,:32]),-1)
            return torch.cat((z[:,:,:prefix].float(),f*cos+r*sin),-2).to(z.dtype)
        q,k=rotate(q),rotate(k)
    scores=(q.float()@k.float().transpose(-2,-1))*0.125
    if compact is not None:
        scores[:,:,:compact.shape[1],prefix:].masked_fill_(~compact[:,None],-float('inf'))
    return (torch.logsumexp(scores,-1)/math.log(2)).flatten(0,1)

def case(batch,heads,n,scale,rope_on,masked,nan_tail=True):
    torch.manual_seed(123)
    # A NaN guard in the physical allocation catches accidental unmasked loads.
    storage=torch.full((batch*(n+64)*3*heads*64,),float('nan'),device='cuda',dtype=torch.bfloat16)
    x=storage[:batch*n*3*heads*64].view(batch,n,3,heads,64)
    x.copy_(torch.randn(x.shape,device='cuda')*scale)
    x=x.detach().requires_grad_()
    prefix=105 if masked else 5
    rope=None
    if rope_on:
        a=torch.randn(n-prefix,32,device='cuda')
        a=torch.cat((a,a),-1)
        rope=(a.sin().contiguous(),a.cos().contiguous())
    mask=None
    if masked:
        mask=torch.rand(batch,100,n-prefix,device='cuda')>0.55
    dy=torch.randn(batch,n,heads*64,device='cuda',dtype=torch.bfloat16)
    y,lse=attention(x,rope,mask,prefix,return_lse=True)
    gx=torch.autograd.grad(y,x,dy)[0]
    xr=x.detach().clone().requires_grad_()
    yr=reference(xr,rope,mask,prefix)
    gr=torch.autograd.grad(yr,xr,dy)[0]
    xf=x.detach().float().requires_grad_()
    yf=reference(xf,rope,mask,prefix)
    gf=torch.autograd.grad(yf,xf,dy.float())[0]
    lse_ref=reference_lse(x.detach(),rope,mask,prefix)
    result={"batch":batch,"heads":heads,"n":n,"scale":scale,"rope":rope_on,"masked":masked,
        "output_vs_bf16":metrics(y,yr),"output_vs_fp32":metrics(y,yf),"bf16_output_vs_fp32":metrics(yr,yf),
        "grads":{},"finite":bool(torch.isfinite(y).all() and torch.isfinite(gx).all() and torch.isfinite(lse).all()),
        'lse_finite':bool(torch.isfinite(lse).all())}
    result['lse_vs_fp32_scores']=metrics(lse,lse_ref)
    for j,name in enumerate(('dq','dk','dv')):
        result['grads'][name]={"vs_bf16":metrics(gx[:,:,j],gr[:,:,j]),
            "vs_fp32":metrics(gx[:,:,j],gf[:,:,j]),"bf16_vs_fp32":metrics(gr[:,:,j],gf[:,:,j])}
    entries=[(result['output_vs_fp32'],result['bf16_output_vs_fp32'])]
    entries += [(v['vs_fp32'],v['bf16_vs_fp32']) for v in result['grads'].values()]
    result['pass']=result['finite'] and result['lse_vs_fp32_scores']['max_abs']<0.002 and all(a['cos']>=0.999 and
        a['max_abs']<=2*b['max_abs']+1e-5 and a['max_rel']<=2*b['max_rel']+1e-5 for a,b in entries)
    print(json.dumps(result),flush=True)
    return result

def mask_case():
    torch.manual_seed(123)
    x=torch.randn(2,100,128,128,device='cuda',dtype=torch.bfloat16)
    disable=torch.rand(2,100,device='cuda')>0.5
    actual=build_compact(x,disable)
    expected=F.interpolate(x,(32,32),mode='bilinear').flatten(2)>0
    expected|=disable[:,:,None]
    result={"mask_mismatches":(actual!=expected).sum().item(),"pass":torch.equal(actual,expected)}
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path);args=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):
        torch.cuda.set_per_process_memory_fraction(0.125)
    cases=[mask_case()]
    if args.quick:
        cases += [case(args.batch,12,137,1.,r,m) for r,m in [(False,False),(True,False),(True,True)]]
    else:
        cases += [case(args.batch,h,n,s,r,m) for h in (16,12) for n,m in [(1029,False),(1129,True)]
                  for s in (1.,8.) for r in (False,True)]
    result={"cases":cases,"pass":all(c['pass'] for c in cases)}
    if args.output: args.output.write_text(json.dumps(result,indent=2))
    if not result['pass']: sys.exit(1)
