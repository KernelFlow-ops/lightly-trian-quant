"""Exercise multi-tensor state, missing grads, schedules and CUDA capture."""
from types import SimpleNamespace
import pytest
import torch
from round7_optimizer.impl.patch import apply

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('clip',[0.,.01])
def test_multi_tensor_100_steps_groups_missing_grads_and_resume(clip):
    torch.manual_seed(51)
    model=torch.nn.Module();model.a=torch.nn.Parameter(torch.randn(257,device='cuda'));model.b=torch.nn.Parameter(torch.randn(8197,device='cuda'))
    refs=[p.detach().clone().requires_grad_() for p in model.parameters()]
    groups=[{'params':[model.a],'lr':.001,'weight_decay':.05,'betas':(.9,.99)},{'params':[model.b],'lr':.0003,'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups)
    reference=torch.optim.AdamW([{**g,'params':[refs[i]]} for i,g in enumerate(groups)],foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:.999**s)
    ref_scheduler=torch.optim.lr_scheduler.LambdaLR(reference,lambda s:.999**s)
    model.model_args=SimpleNamespace(gradient_clip_val=clip)
    ctx=SimpleNamespace(train_model=model,optimizer=optimizer,scheduler=scheduler,meta={},ops=SimpleNamespace(),shadow={},post_opt_hooks=[])
    apply(ctx,variant='cuda_multi_tensor')
    for step in range(100):
        for i,(p,r) in enumerate(zip(model.parameters(),refs)):
            if i==1 and step%4==0:p.grad=None;r.grad=None
            else:
                grad=torch.randn_like(p)*.1;p.grad=grad.clone();r.grad=grad.clone()
        if clip>0:expected=torch.nn.utils.clip_grad_norm_(refs,clip)
        actual=ctx.ops.opt_step(ctx).clone();reference.step();reference.zero_grad(set_to_none=True);ref_scheduler.step()
        if clip>0:torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        for p,r in zip(model.parameters(),refs):torch.testing.assert_close(p,r,rtol=1e-6,atol=1e-6)
        if step==50:
            ptrs=[g['lr'].data_ptr() for g in optimizer.param_groups]
            saved=optimizer.state_dict();optimizer.load_state_dict(saved)
            assert [g['lr'].data_ptr() for g in optimizer.param_groups]==ptrs
    for p,r in zip(model.parameters(),refs):
        for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(optimizer.state[p][key],reference.state[r][key].to(p.device),rtol=1e-6,atol=1e-6)
