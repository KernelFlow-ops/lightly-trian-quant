import argparse,json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from impl.rope import sdpa_rope
from impl.attention import reference
sys.path.insert(0,str(R/'tests'))
from bench_attention import benchmark

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--heads',type=int,choices=(12,16));p.add_argument('--precision',choices=('bf16','fp32'),default='bf16');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD')=='1'
    torch.manual_seed(0);results=[]
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float32
    for h in ((a.heads,) if a.heads else (16,12)):
        for n in (1029,1129):
            x=torch.randn(a.batch,n,3,h,64,device='cuda',dtype=dtype,requires_grad=True)
            dy=torch.randn(a.batch,n,h*64,device='cuda',dtype=dtype)
            angle=torch.randn(1024,32,device='cuda');angle=torch.cat((angle,angle),-1);rope=(angle.sin(),angle.cos())
            for variant,fn in [('torch_rope_sdpa',lambda z:reference(z,rope)),('fused_rope_sdpa',lambda z:sdpa_rope(z,rope))]:
                result={'h':h,'n':n,'batch':a.batch,'precision':a.precision,'variant':variant,**benchmark(fn,x,dy)}
                results.append(result);print(json.dumps(result),flush=True)
            del x,dy;torch.cuda.empty_cache()
    a.output.write_text(json.dumps({'microbench':results,'condition':'No mask; same SDPA automatic backend; original packed layout'},indent=2))
