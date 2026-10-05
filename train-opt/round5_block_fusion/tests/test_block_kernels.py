from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from impl.fused_norm import residual_norm
from impl.gelu import gelu
from impl.shadow import _ShadowLinear

def metric(x,y):
    a,b=x.float().flatten(),y.float().flatten()
    return {'max_abs':(a-b).abs().max().item(),'max_rel':((a-b).abs()/b.abs().clamp_min(1e-4)).max().item(),
            'cos':F.cosine_similarity(a,b,dim=0).item()}

def norm_case(m,d,spike,precision='bf16'):
    torch.manual_seed(123)
    x=torch.randn(m,d,device='cuda')
    if spike: x[0,0]=1.5e5
    b=torch.randn(m,d,device='cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float32)
    gamma=torch.randn(d,device='cuda')*0.1
    w=torch.randn(d,device='cuda');bias=torch.randn(d,device='cuda')
    tensors=[v.requires_grad_() for v in (x,b,gamma,w,bias)]
    dr=torch.randn_like(x);dy=torch.randn_like(b)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        r,y=residual_norm(x,b,gamma,SimpleNamespace(weight=w,bias=bias,eps=1e-6))
    actual=torch.autograd.grad((r,y),tensors,(dr,dy))
    ref=[v.detach().clone().requires_grad_() for v in tensors]
    rr=ref[0]+ref[1]*ref[2];yr=F.layer_norm(rr,(d,),ref[3],ref[4],1e-6).to(b.dtype)
    gr=torch.autograd.grad((rr,yr),ref,(dr,dy))
    fp=[v.detach().float().requires_grad_() for v in tensors]
    rf=fp[0]+fp[1]*fp[2];yf=F.layer_norm(rf,(d,),fp[3],fp[4],1e-6)
    gf=torch.autograd.grad((rf,yf),fp,(dr,dy.float()))
    result={'kernel':'residual_norm','m':m,'d':d,'spike':spike,'precision':precision,'residual':metric(r,rr),'output':metric(y,yf),
            'bf16_output':metric(yr,yf),'grads':{name:{'vs_bf16':metric(a,br),'vs_fp32':metric(a,f),
                       'bf16_vs_fp32':metric(br,f)} for name,a,br,f in zip(('dx','db','dgamma','dweight','dbias'),actual,gr,gf)}}
    result['pass']=torch.equal(r,rr) and result['output']['cos']>=0.999 and all(
        z['vs_fp32']['cos']>=0.999 and z['vs_fp32']['max_abs']<=2*z['bf16_vs_fp32']['max_abs']+2e-4
        for z in result['grads'].values())
    print(json.dumps(result),flush=True);return result

def gelu_case(m,d):
    torch.manual_seed(123);x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x);y=gelu(x);g=torch.autograd.grad(y,x,dy)[0]
    xr=x.detach().clone().requires_grad_();yr=F.gelu(xr);gr=torch.autograd.grad(yr,xr,dy)[0]
    xf=x.detach().float().requires_grad_();yf=F.gelu(xf);gf=torch.autograd.grad(yf,xf,dy.float())[0]
    result={'kernel':'gelu','m':m,'d':d,'output_vs_fp32':metric(y,yf),'bf16_output_vs_fp32':metric(yr,yf),
            'grad_vs_fp32':metric(g,gf),'bf16_grad_vs_fp32':metric(gr,gf),'vs_bf16':metric(y,yr),'grad_vs_bf16':metric(g,gr)}
    result['pass']=result['grad_vs_fp32']['cos']>=0.999 and result['vs_bf16']['max_abs']<=0.01
    print(json.dumps(result),flush=True);return result

def shadow_case(m,d,n):
    torch.manual_seed(123);x=torch.randn(m,d,device='cuda',requires_grad=True)
    w=torch.randn(n,d,device='cuda',requires_grad=True);b=torch.randn(n,device='cuda',requires_grad=True)
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    y=_ShadowLinear.apply(x,w,b,w.detach().bfloat16());g=torch.autograd.grad(y,(x,w,b),dy)
    xr,wr,br=[v.detach().clone().requires_grad_() for v in (x,w,b)]
    with torch.autocast('cuda',dtype=torch.bfloat16): yr=F.linear(xr,wr,br)
    gr=torch.autograd.grad(yr,(xr,wr,br),dy)
    xf,wf,bf=[v.detach().clone().requires_grad_() for v in (x,w,b)]
    yf=F.linear(xf,wf,bf);gf=torch.autograd.grad(yf,(xf,wf,bf),dy.float())
    result={'kernel':'shadow','m':m,'d':d,'n':n,'output_vs_bf16':metric(y,yr),
            'grads':{name:{'vs_bf16':metric(a,br),'vs_fp32':metric(a,f),'bf16_vs_fp32':metric(br,f)}
                     for name,a,br,f in zip(('dx','dw','db'),g,gr,gf)}}
    result['pass']=torch.equal(y,yr) and all(z['vs_fp32']['cos']>=0.999 for z in result['grads'].values())
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--dim',type=int,choices=(768,1024));p.add_argument('--precision',choices=('bf16','fp32'),default='bf16');p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    rows=(137,) if a.quick else (a.batch*1029,a.batch*1129)
    dims=(a.dim,) if a.dim else (768,1024)
    cases=[norm_case(m,d,s,a.precision) for m in rows for d in dims for s in (False,True)]
    cases += [gelu_case(m,4*d) for m in rows for d in dims]
    cases += [shadow_case(rows[0],d,4*d) for d in dims]
    a.output.write_text(json.dumps({'cases':cases,'pass':all(c['pass'] for c in cases)},indent=2))
    if not all(c['pass'] for c in cases):sys.exit(1)
