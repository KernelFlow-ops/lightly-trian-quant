from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from impl.fp8 import ScalingState,linear,quantize_rows

def metric(a,b):
    a,b=a.float().flatten(),b.float().flatten()
    return {'max_abs':(a-b).abs().max().item(),'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            'cos':F.cosine_similarity(a,b,dim=0).item()}

def dequant(x):
    q,s=quantize_rows(x)
    return q.float()*s

def case(m,k,n,delayed):
    torch.manual_seed(123)
    x=torch.randn(m,k,device='cuda',requires_grad=True)
    w=(torch.randn(n,k,device='cuda')/k**0.5).requires_grad_()
    b=torch.randn(n,device='cuda',requires_grad=True)
    module=SimpleNamespace(weight=w,bias=b)
    state=ScalingState(delayed=delayed)
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    y=linear(x,module,state);g=torch.autograd.grad(y,(x,w,b),dy)
    xf,wf,bf=[z.detach().clone().requires_grad_() for z in (x,w,b)]
    yf=F.linear(xf,wf,bf);gf=torch.autograd.grad(yf,(xf,wf,bf),dy.float())
    result={'m':m,'k':k,'n':n,'delayed':delayed,'output':metric(y,yf),
           'grads':{name:metric(ga,gr) for name,ga,gr in zip(('dx','dw','db'),g,gf)},
           'finite':bool(torch.isfinite(y).all() and all(torch.isfinite(z).all() for z in g))}
    # FP8 is a changed algorithm. Report exact A gates separately from finite
    # operational support; it cannot be silently promoted on this smoke test.
    result['level_a_cos_pass']=result['output']['cos']>=.999 and all(z['cos']>=.999 for z in result['grads'].values())
    xb=x.detach().bfloat16();dyb=dy.bfloat16()
    # Validate the actual quantized operator separately from its training
    # approximation to an unquantized fp32 Linear.
    yq=(dequant(xb)@dequant(w.detach()).t()+b.detach().bfloat16().float()).bfloat16()
    dxq=(dequant(dyb)@dequant(w.detach().t()).t()).bfloat16().float()
    pad=(-xb.shape[0])%16
    if pad:
        xp=F.pad(xb,(0,0,0,pad));dp=F.pad(dyb,(0,0,0,pad))
    else:xp,dp=xb,dyb
    dwq=(dequant(dp.t())@dequant(xp.t()).t()).bfloat16().float()
    result['dequant_operator']={'output':metric(y,yq),'dx':metric(g[0],dxq),'dw':metric(g[1],dwq),'db':metric(g[2],dyb.float().sum(0))}
    result['dequant_gemm_verified']=all(v['cos']>=0.999 for v in result['dequant_operator'].values())
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    cases=[]
    for m,k,n in ([(137,128,256)] if a.quick else [(2058,1024,4096),(2058,4096,1024),(2058,768,3072),(2058,3072,768)]):
        for delay in (False,True):
            try:cases.append(case(m,k,n,delay))
            except Exception as e:
                result={'m':m,'k':k,'n':n,'delayed':delay,'error':str(e)};cases.append(result);print(json.dumps(result),flush=True)
    a.output.write_text(json.dumps({'cases':cases,'all_finite':all(z.get('finite',False) for z in cases)},indent=2))
    if not all(z.get('finite',False) for z in cases):sys.exit(1)
