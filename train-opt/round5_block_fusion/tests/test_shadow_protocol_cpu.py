"""CPU-only writer ownership, fp32 fallback and serialization contract checks."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R.parent/'src'))
from lightly_train._models.dinov3.dinov3_src.layers.attention import SelfAttention
from lightly_train._models.dinov3.dinov3_src.layers.ffn_layers import Mlp
from round5_block_fusion.impl.shadow import setup

torch.manual_seed(123)
train=torch.nn.Module();train.model=torch.nn.Module();train.model.backbone=torch.nn.Module()
block=torch.nn.Module();block.attn=SelfAttention(64,1,qkv_bias=True,mask_k_bias=True);block.mlp=Mlp(64,256)
block.attn.qkv.bias_mask.fill_(1);block.attn.qkv.bias_mask[64:128]=0
train.model.backbone.blocks=torch.nn.ModuleList([block])
ctx=SimpleNamespace(train_model=train,model=train.model,shadow={},post_opt_hooks=[],meta={})
before={key:tuple(value.shape) for key,value in train.state_dict().items()};ids={name:id(p) for name,p in train.named_parameters()}
x=torch.randn(3,64);expected=block.attn.qkv(x);setup(ctx);actual=block.attn.qkv(x);setup(ctx)
assert torch.equal(actual,expected)
assert len(ctx.post_opt_hooks)==1
assert before=={key:tuple(value.shape) for key,value in train.state_dict().items()}
assert ids=={name:id(p) for name,p in train.named_parameters()}
with torch.no_grad():
    for p in train.parameters():p.add_(0.125)
ctx.post_opt_hooks[0](ctx)
assert all(not v.requires_grad and v.grad_fn is None for v in ctx.shadow.values())
for name,p in train.named_parameters():
    if name in ctx.shadow:assert torch.equal(ctx.shadow[name],p.bfloat16())
saved={name:value.clone() for name,value in ctx.shadow.items()}
ctx.meta['shadow_writer']='round7_optimizer'
with torch.no_grad():
    for p in train.parameters():p.add_(0.125)
ctx.post_opt_hooks[0](ctx)
assert all(torch.equal(ctx.shadow[name],value) for name,value in saved.items())
result={'cpu_fp32_fallback_exact':True,'hook_idempotent':True,'post_opt_refresh_exact':True,
        'refresh_has_no_autograd_history':True,
        'optimizer_writer_ownership_respected':True,'parameter_identity_preserved':True,'state_keys_shapes_preserved':True,
        'shadow_count':len(ctx.shadow),'pass':True}
(Path(__file__).resolve().parents[1]/'profile/shadow_protocol_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
