"""Independent sm_120 current tensor/row/MX/NVFP4 support probe."""
import argparse,json
from pathlib import Path
import torch

# copied from infer-opt/round6_low_precision/mx_test.py (layout utility)
def to_blocked(x):
    rows,cols=x.shape;nr=(rows+127)//128;nc=(cols+3)//4
    pad=torch.zeros(nr*128,nc*4,dtype=x.dtype,device=x.device);pad[:rows,:cols]=x
    return pad.view(nr,128,nc,4).permute(0,2,1,3).reshape(-1,4,32,4).transpose(1,2).reshape(-1,32,16).flatten()

def mx_quant(x):
    z=x.reshape(x.shape[0],-1,32)
    s=torch.exp2(torch.ceil(torch.log2((z.abs().amax(-1)/448).clamp_min(1e-30))))
    q=(z/s[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).reshape(x.shape)
    return q,to_blocked(s.to(torch.float8_e8m0fnu)),(q.float().reshape(z.shape)*s[:,:,None]).reshape(x.shape)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    torch.manual_seed(0);m,n,k=256,512,1024
    x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')
    results=[]
    for mode in ('tensorwise','rowwise','mxfp8','nvfp4'):
        r={'mode':mode,'m':m,'n':n,'k':k}
        try:
            if mode=='mxfp8':qx,sx,dx=mx_quant(x);qw,sw,dw=mx_quant(w)
            elif mode=='nvfp4':
                qx=torch.randint(0,255,(m,k//2),device='cuda',dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
                qw=torch.randint(0,255,(n,k//2),device='cuda',dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
                sx=to_blocked(torch.ones(m,k//16,device='cuda').to(torch.float8_e4m3fn))
                sw=to_blocked(torch.ones(n,k//16,device='cuda').to(torch.float8_e4m3fn))
            else:
                dims=1 if mode=='rowwise' else None
                sx=(x.abs().amax(dim=dims,keepdim=dims is not None)/448).clamp_min(1e-30)
                sw=(w.abs().amax(dim=dims,keepdim=dims is not None)/448).clamp_min(1e-30)
                qx=(x/sx).to(torch.float8_e4m3fn);qw=(w/sw).to(torch.float8_e4m3fn)
                dx=qx.float()*sx;dw=qw.float()*sw
                if mode=='rowwise':sw=sw.t().contiguous()
            y=torch._scaled_mm(qx,qw.t(),scale_a=sx,scale_b=sw,out_dtype=torch.bfloat16)
            torch.cuda.synchronize();r['supported']=True;r['finite']=bool(torch.isfinite(y).all())
            if mode!='nvfp4':r['rel_l2_vs_dequant']=((y.float()-dx@dw.t()).norm()/(dx@dw.t()).norm()).item()
        except Exception as e:r['supported']=False;r['error']=str(e)
        results.append(r);print(json.dumps(r),flush=True)
    a.output.write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'formats':results},indent=2))
