"""MXFP8 layout and full backward operator A validation."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tests'))
from impl.mxfp8 import quantize_blocks,MXScalingState
from impl.fp8 import linear
from probe_formats import mx_quant
from test_fp8 import metric

def case(m,k,n):
    torch.manual_seed(123);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(n,k,device='cuda')/k**.5).requires_grad_();b=torch.randn(n,device='cuda',requires_grad=True)
    q,s=quantize_blocks(x.detach());qr,sr,dx=mx_quant(x.detach().float())
    layout_equal=torch.equal(s.view(torch.uint8),sr.view(torch.uint8));q_equal=torch.equal(q.view(torch.uint8),qr.view(torch.uint8))
    state=MXScalingState();dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    y=linear(x,SimpleNamespace(weight=w,bias=b),state);g=torch.autograd.grad(y,(x,w,b),dy)
    _,_,dw=mx_quant(w.detach());yr=(dx@dw.t()+b.detach().bfloat16().float()).bfloat16()
    _,_,dyd=mx_quant(dy.float());_,_,wtd=mx_quant(w.detach().t())
    dxr=(dyd@wtd.t()).bfloat16()
    pad=(-m)%32
    xp=torch.nn.functional.pad(x.detach(),(0,0,0,pad));dp=torch.nn.functional.pad(dy,(0,0,0,pad))
    _,_,dpt=mx_quant(dp.t().float());_,_,xpt=mx_quant(xp.t().float())
    dwr=(dpt@xpt.t()).bfloat16().float();dbr=dp.float().sum(0)
    r={'m':m,'k':k,'n':n,'scale_layout_equal':layout_equal,'quantized_bytes_equal':q_equal,'output_vs_dequant':metric(y,yr),
       'grads_vs_dequant':{name:metric(a,b) for name,a,b in zip(('dx','dw','db'),g,(dxr,dwr,dbr))},
       'finite':bool(torch.isfinite(y).all() and all(torch.isfinite(v).all() for v in g))}
    r['pass']=layout_equal and q_equal and r['finite'] and r['output_vs_dequant']['cos']>=.999 and all(z['cos']>=.999 for z in r['grads_vs_dequant'].values())
    print(json.dumps(r),flush=True);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    cases=[]
    for shape in ([(137,128,256)] if a.quick else [(2058,1024,4096),(2058,4096,1024),(2058,768,3072),(2058,3072,768)]):
        try:cases.append(case(*shape))
        except Exception as e:
            r={'m':shape[0],'k':shape[1],'n':shape[2],'error':str(e),'pass':False};cases.append(r);print(json.dumps(r),flush=True)
    a.output.write_text(json.dumps({'cases':cases,'pass':all(c['pass'] for c in cases)},indent=2))
    if not all(c['pass'] for c in cases):sys.exit(1)
