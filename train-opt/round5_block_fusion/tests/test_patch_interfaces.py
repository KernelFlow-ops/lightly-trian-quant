"""Small real Block composition acceptance, not the harness level-B gate."""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R.parent/'src'))
from lightly_train._models.dinov3.dinov3_src.layers.attention import SelfAttention
from lightly_train._models.dinov3.dinov3_src.layers.layer_scale import LayerScale
from lightly_train._models.dinov3.dinov3_src.layers.ffn_layers import Mlp
from round4_attention.impl.patch import apply as patch4
from round5_block_fusion.impl.patch import apply as patch5
from round9_low_precision.impl.patch import apply as patch9
from common.rng import RngTape

class Block(torch.nn.Module):
    def __init__(self,d,h):
        super().__init__();self.norm1=torch.nn.LayerNorm(d,eps=1e-6);self.norm2=torch.nn.LayerNorm(d,eps=1e-6)
        self.attn=SelfAttention(d,h,qkv_bias=True,mask_k_bias=True)
        self.attn.qkv.bias_mask.fill_(1);self.attn.qkv.bias_mask[d:2*d]=0
        self.ls1=LayerScale(d);self.ls1.reset_parameters();self.ls2=LayerScale(d);self.ls2.reset_parameters()
        self.mlp=Mlp(d,4*d)

class Train(torch.nn.Module):
    def __init__(self,d,h):
        super().__init__();self.model=torch.nn.Module();self.model.backbone=torch.nn.Module()
        self.model.backbone.blocks=torch.nn.ModuleList([Block(d,h)])
        self.model.backbone.n_storage_tokens=4;self.model.num_queries=100;self.model.num_joint_blocks=1
        self.model.attn_mask_probs=[torch.tensor(1.,device='cuda')]

def make_ctx(model):
    def attn(c,block,x,rope,mask):
        mod=block.attn;b,n,_=x.shape;qkv=mod.qkv(x).reshape(b,n,3,mod.num_heads,64)
        q,k,v=(t.transpose(1,2) for t in qkv.unbind(2))
        q,k=mod.apply_rope(q,k,rope)
        if mask is not None:mask=mask[:,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape_as(x)
        return mod.proj_drop(mod.proj(y))
    def block(c,i,blk,x,rope,mask):
        x=x+blk.ls1(c.ops.attn(c,blk,blk.norm1(x),rope,mask));return x+blk.ls2(blk.mlp(blk.norm2(x)))
    return SimpleNamespace(train_model=model,model=model.model,ops=SimpleNamespace(attn=attn,block=block,build_mask=lambda *a:None),
                           shadow={},meta={},post_opt_hooks=[],rng=RngTape())

def run(d,h,variant,fp8=False):
    torch.manual_seed(123);ref=Train(d,h).cuda();opt=copy.deepcopy(ref);cr,co=make_ctx(ref),make_ctx(opt)
    before={k:tuple(v.shape) for k,v in opt.state_dict().items()};ids={k:id(p) for k,p in opt.named_parameters()}
    patch4(co,variant='sdpa_rope');patch5(co,variant=variant);patch5(co,variant=variant)
    if fp8:patch9(co,variant='fp8_mlp_cur',layers=[0])
    assert before=={k:tuple(v.shape) for k,v in opt.state_dict().items()}
    assert ids=={k:id(p) for k,p in opt.named_parameters()}
    x=torch.randn(1,137,d,device='cuda');angle=torch.randn(132,32,device='cuda');angle=torch.cat((angle,angle),-1);rope=(angle.sin(),angle.cos())
    dy=torch.randn_like(x)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        yr=cr.ops.block(cr,0,ref.model.backbone.blocks[0],x,rope,None)
        yo=co.ops.block(co,0,opt.model.backbone.blocks[0],x,rope,None)
    yr.backward(dy);yo.backward(dy)
    cos=[];missing=[]
    for (name,p),(name2,p2) in zip(ref.named_parameters(),opt.named_parameters()):
        if p.grad is None or p2.grad is None:missing.append(name);continue
        a,b=p.grad.flatten(),p2.grad.flatten()
        if a.norm()==0 and b.norm()==0:c=1.
        else:c=F.cosine_similarity(a,b,dim=0).item()
        cos.append((name,c))
    result={'d':d,'variant':variant,'fp8':fp8,'keys_shapes_preserved':True,'parameter_identity_preserved':True,
            'output_max_abs':(yo-yr).abs().max().item(),'grad_cos_min':min(c for _,c in cos),'missing_grads':missing,
            'shadow_count':len(co.shadow),'grad_cos':dict(cos),'note':'local Block acceptance, not full training level B'}
    result['pass']=not missing and result['grad_cos_min']>=.99
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not __import__('os').environ.get('TRAIN_OPT_GPU_LOCK_HELD'):torch.cuda.set_per_process_memory_fraction(.125)
    cases=[run(d,h,v,f) for d,h in [(768,12),(1024,16)] for v,f in [('ln_fuse',False),('all',False),('all',True)]]
    a.output.write_text(json.dumps({'cases':cases,'pass':all(c['pass'] for c in cases)},indent=2))
    if not all(c['pass'] for c in cases):sys.exit(1)
