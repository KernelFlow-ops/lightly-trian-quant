"""Locked backend matrix, explicit fwd / bwd / fwd+bwd CUDA event timing."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from impl.attention import attention,reference
from impl.flex import flex
from impl.mask import build_compact

def event_ms(fn,warmup=20,iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times=[]
    for _ in range(3):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters): fn()
        b.record();b.synchronize();times.append(a.elapsed_time(b)/iters)
    return sorted(times)[1]

def benchmark(fn,x,dy):
    f=event_ms(lambda:fn(x))
    y=fn(x)
    b=event_ms(lambda:torch.autograd.grad(y,x,dy,retain_graph=True))
    fb=event_ms(lambda:torch.autograd.grad(fn(x),x,dy))
    return {'fwd_us':f*1000,'bwd_us':b*1000,'fwd_bwd_us':fb*1000}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--flex',action='store_true');args=p.parse_args()
    assert __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD')=='1','Must hold both GPU locks'
    torch.manual_seed(0);results=[]
    for h in (16,12):
        for n,masked in ((1029,False),(1129,True)):
            x=torch.randn(args.batch,n,3,h,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            dy=torch.randn(args.batch,n,h*64,device='cuda',dtype=torch.bfloat16)
            prefix=105 if masked else 5
            angle=torch.randn(n-prefix,32,device='cuda');angle=torch.cat((angle,angle),-1)
            rope=(angle.sin(),angle.cos())
            mask=None
            if masked:
                logits=torch.randn(args.batch,100,128,128,device='cuda',dtype=torch.bfloat16)
                mask=build_compact(logits)
                result={'h':h,'n':n,'batch':args.batch,'variant':'mask_build',
                        'fwd_us':event_ms(lambda:build_compact(logits))*1000,
                        'torch_fwd_us':event_ms(lambda:torch.nn.functional.interpolate(logits,(32,32),mode='bilinear').flatten(2)>0)*1000}
                results.append(result);print(json.dumps(result),flush=True)
            for name,backend in (('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION),('efficient',SDPBackend.EFFICIENT_ATTENTION)):
                result={'h':h,'n':n,'batch':args.batch,'variant':name,'masked':masked}
                try:
                    with sdpa_kernel(backend): result.update(benchmark(lambda z:reference(z,rope,mask,prefix),x,dy))
                except Exception as e:result['error']=str(e)
                results.append(result);print(json.dumps(result),flush=True)
            for bm,bn in ((32,64),(64,64),(64,32)):
                result={'h':h,'n':n,'batch':args.batch,'variant':f'triton_rope_mask_{bm}x{bn}','masked':masked}
                try: result.update(benchmark(lambda z:attention(z,rope,mask,prefix,(bm,bn)),x,dy))
                except Exception as e:result['error']=str(e)
                results.append(result);print(json.dumps(result),flush=True)
            if args.flex:
                result={'h':h,'n':n,'batch':args.batch,'variant':'flex','masked':masked}
                try: result.update(benchmark(lambda z:flex(z,rope,mask,prefix),x,dy))
                except Exception as e:result['error']=str(e)
                results.append(result);print(json.dumps(result),flush=True)
            del x,dy;torch.cuda.empty_cache()
    args.output.write_text(json.dumps({'microbench':results,'torch':torch.__version__,'time':time.strftime('%Y-%m-%dT%H:%M:%S')},indent=2))
