import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R.parents[0]))
from impl.fused_norm import residual_norm
from impl.gelu import gelu
from impl.shadow import _ShadowLinear
from round4_attention.tests.bench_attention import event_ms

def bench(fn,inputs,grads):
    f=event_ms(lambda:fn(*inputs));y=fn(*inputs)
    if isinstance(y,torch.Tensor):y=(y,)
    b=event_ms(lambda:torch.autograd.grad(y,inputs,grads,retain_graph=True))
    fb=event_ms(lambda:torch.autograd.grad(fn(*inputs),inputs,grads))
    return {'fwd_us':f*1000,'bwd_us':b*1000,'fwd_bwd_us':fb*1000}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--dim',type=int,choices=(768,1024));p.add_argument('--precision',choices=('bf16','fp32'),default='bf16');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD')=='1';torch.manual_seed(0);results=[]
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float32
    for d in ((a.dim,) if a.dim else (1024,768)):
      for n in (1029,1129):
        m=a.batch*n
        x=torch.randn(m,d,device='cuda',requires_grad=True);b=torch.randn(m,d,device='cuda',dtype=dtype,requires_grad=True)
        g=torch.randn(d,device='cuda',requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);bias=torch.randn(d,device='cuda',requires_grad=True)
        norm=SimpleNamespace(weight=w,bias=bias,eps=1e-6);grads=(torch.randn_like(x),torch.randn_like(b))
        def ref(x,b,g,w,bias):
            r=x+b*g;return r,F.layer_norm(r,(d,),w,bias,1e-6).to(dtype)
        def opt(x,b,g,w,bias):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
                return residual_norm(x,b,g,SimpleNamespace(weight=w,bias=bias,eps=1e-6))
        for name,fn in [('torch_residual_ln',ref),('fused_residual_ln',opt)]:
            result={'d':d,'n':n,'batch':a.batch,'precision':a.precision,'variant':name,**bench(fn,(x,b,g,w,bias),grads)}
            results.append(result);print(json.dumps(result),flush=True)
        z=torch.randn(m,4*d,device='cuda',dtype=dtype,requires_grad=True);dy=torch.randn_like(z)
        for name,fn in [('torch_gelu',F.gelu),('triton_gelu',gelu)]:
            result={'d':d,'n':n,'batch':a.batch,'precision':a.precision,'variant':name,**bench(fn,(z,),(dy,))}
            results.append(result);print(json.dumps(result),flush=True)
        del x,b,g,w,bias,z,dy;torch.cuda.empty_cache()
    a.output.write_text(json.dumps({'microbench':results},indent=2))
