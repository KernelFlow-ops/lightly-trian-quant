"""FP32/TF32x3 FA candidate, motivated by the actual FP32 baseline profile."""
import argparse,json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tests'))
from impl.attention import attention,reference
from test_attention import metrics,reference_lse

def case(b,h,n,scale,bm,bn):
    torch.manual_seed(123);x=(torch.randn(b,n,3,h,64,device='cuda')*scale).requires_grad_()
    angle=torch.randn(1024,32,device='cuda');angle=torch.cat((angle,angle),-1);rope=(angle.sin(),angle.cos())
    prefix=n-1024;mask=torch.rand(b,100,1024,device='cuda')>.5 if n==1129 else None
    dy=torch.randn(b,n,h*64,device='cuda')
    y,lse=attention(x,rope,mask,prefix,(bm,bn),return_lse=True);g=torch.autograd.grad(y,x,dy)[0]
    xr=x.detach().clone().requires_grad_();yr=reference(xr,rope,mask,prefix);gr=torch.autograd.grad(yr,xr,dy)[0]
    result={'b':b,'h':h,'n':n,'scale':scale,'bm':bm,'bn':bn,'output':metrics(y,yr),
            'lse':metrics(lse,reference_lse(x.detach(),rope,mask,prefix)),
            'gradients':{name:metrics(g[:,:,j],gr[:,:,j]) for j,name in enumerate(('dq','dk','dv'))}}
    result['pass']=result['output']['cos']>=.999 and all(v['cos']>=.999 for v in result['gradients'].values()) and bool(torch.isfinite(y).all() and torch.isfinite(g).all())
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=1);p.add_argument('--heads',type=int,choices=(12,16),default=12)
    p.add_argument('--bm',type=int,default=32);p.add_argument('--bn',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    cases=[case(a.batch,a.heads,n,s,a.bm,a.bn) for n in (1029,1129) for s in (1.,8.)]
    a.output.write_text(json.dumps({'cases':cases,'pass':all(c['pass'] for c in cases)},indent=2))
    if not all(c['pass'] for c in cases):sys.exit(1)
