import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from impl.rope import rope_qkv,sdpa_rope
from impl.attention import reference

def case(b,h,n,scale,dtype=torch.bfloat16):
    torch.manual_seed(123)
    x=(torch.randn(b,n,3,h,64,device='cuda')*scale).to(dtype).requires_grad_()
    angles=torch.randn(1024,32,device='cuda');angles=torch.cat((angles,angles),-1)
    rope=(angles.sin(),angles.cos())
    y=rope_qkv(x,rope);dy=torch.randn_like(y);gx=torch.autograd.grad(y,x,dy)[0]
    xr=x.detach().clone().requires_grad_();prefix=n-1024
    vals=list(xr.unbind(2))
    for j in (0,1):
        raw=vals[j].float();v=raw[:,prefix:]
        rot=torch.cat((-v[...,32:],v[...,:32]),-1)
        vals[j]=torch.cat((raw[:,:prefix],v*rope[1][:,None]+rot*rope[0][:,None]),1).to(dtype)
    yr=torch.stack(vals,2);gr=torch.autograd.grad(yr,xr,dy)[0]
    r={'b':b,'h':h,'n':n,'scale':scale,'dtype':str(dtype),'forward_equal':torch.equal(y,yr),'grad_equal':torch.equal(gx,gr),
       'forward_max_abs':(y-yr).abs().max().item(),'grad_max_abs':(gx-gr).abs().max().item()}
    r['pass']=r['forward_equal'] and r['grad_equal'];print(json.dumps(r),flush=True);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--heads',type=int,choices=(12,16));p.add_argument('--dtype',choices=('bf16','fp32'),default='bf16');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    dtype=torch.bfloat16 if a.dtype=='bf16' else torch.float32
    cases=[case(a.batch,h,n,s,dtype) for h in ((a.heads,) if a.heads else (12,16)) for n in (1029,1129) for s in (1.,8.)]
    a.output.write_text(json.dumps({'cases':cases,'pass':all(c['pass'] for c in cases)},indent=2))
    if not all(c['pass'] for c in cases):sys.exit(1)
