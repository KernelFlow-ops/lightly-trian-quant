from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from round10_integration.impl.full_graph import FullGraphRunner, padded_batch


def _batch(counts=(1, 2)):
    targets = []
    for i, count in enumerate(counts):
        targets.append({
            "masks": torch.arange(count * 4 * 4, dtype=torch.uint8).reshape(count, 4, 4),
            "labels": torch.arange(count, dtype=torch.long),
        })
    return {"image": torch.randn(len(targets), 3, 8, 8), "binary_masks": targets}


def test_padded_batch_has_static_shapes_and_valid_flags():
    converted = padded_batch(_batch(), torch.device("cpu"), max_targets=3)
    assert converted["image"].shape == (2, 3, 8, 8)
    targets = converted["binary_masks"]
    assert targets["masks"].shape == (2, 3, 4, 4)
    assert targets["labels"].shape == (2, 3)
    assert targets["valid"].tolist() == [[True, False, False], [True, True, False]]
    assert targets["masks"][0, 1:].sum() == 0


def test_padded_batch_rejects_static_capacity_overflow():
    with pytest.raises(ValueError, match="smaller"):
        padded_batch(_batch((3,)), torch.device("cpu"), max_targets=2)


def test_full_graph_contract_reports_cpu_failure_without_mutation():
    parameter = torch.nn.Parameter(torch.ones(()))
    ctx = SimpleNamespace(
        train_model=SimpleNamespace(parameters=lambda: iter((parameter,))),
        rng=SimpleNamespace(mode="off"), meta={},
    )
    runner = FullGraphRunner(ctx)
    report = runner.capture(_batch((1,)))
    assert report.captured is False
    assert "CUDA" in (report.capture_error or "")
    assert runner.ready is False


def test_graph_keys_change_only_at_mask_branch_boundaries():
    assert FullGraphRunner.phase_key((1, .2, 0)) == FullGraphRunner.phase_key((1, .8, 0))
    assert FullGraphRunner.phase_key((1, .8, 0)) != FullGraphRunner.phase_key((1, 0, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_capture_restores_existing_adam_state_rng_and_replay_300_steps():
    from round7_optimizer.impl.patch import apply
    class TinyTrain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=torch.nn.Linear(3,2)
            self.model.register_buffer('attn_mask_probs',torch.ones(3))
            self.model_args=SimpleNamespace(gradient_clip_val=.01,poly_power=.9,
                attn_mask_annealing_steps_start=[10,20,30],attn_mask_annealing_steps_end=[20,30,40])
    train=TinyTrain().cuda()
    optimizer=torch.optim.AdamW(train.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: .999**step)
    ctx=SimpleNamespace(train_model=train,model=train.model,optimizer=optimizer,scheduler=scheduler,
        host_probs=[1.,1.,1.],rng=SimpleNamespace(mode='off'),shadow={},post_opt_hooks=[],meta={},ops=SimpleNamespace(),
        fabric=SimpleNamespace(backward=lambda loss:loss.backward()))
    ctx.meta.update(round3_loss_matcher=object(),round1_annealing_host=True)
    ctx.meta['forward_train_fn']=lambda c,x,**kwargs:([c.model(x)],[])
    ctx.ops.loss_all_layers=lambda c,masks,classes,targets:(masks[0].square().mean(),{})
    apply(ctx,variant='capturable_torch')
    images=torch.randn(2,3,device='cuda')
    batch={'image':images,'binary_masks':[{'masks':torch.ones(1,4,4,device='cuda',dtype=torch.uint8),'labels':torch.zeros(1,device='cuda',dtype=torch.long)} for _ in range(2)]}
    # Initialize nonzero moments before capture to exercise resumed training.
    for _ in range(5):
        train.model(images).square().mean().backward();ctx.ops.opt_step(ctx)
    params=[p for p in train.parameters()]
    before={p:{k:v.clone() for k,v in optimizer.state[p].items()} for p in params}
    model_before={k:v.clone() for k,v in train.state_dict().items()}
    rng_before=torch.cuda.get_rng_state()
    runner=FullGraphRunner(ctx,warmup=2)
    assert runner.capture(batch).captured,runner.report.capture_error
    assert torch.equal(torch.cuda.get_rng_state(),rng_before)
    for key,value in train.state_dict().items():torch.testing.assert_close(value,model_before[key],rtol=0,atol=0)
    for p in params:
        for key,value in optimizer.state[p].items():torch.testing.assert_close(value,before[p][key],rtol=0,atol=0)
    reference=TinyTrain().cuda();reference.load_state_dict(model_before)
    ref_opt=torch.optim.AdamW(reference.parameters(),lr=.001,fused=True)
    ref_opt.load_state_dict(optimizer.state_dict())
    # Load-state takes the original non-capturable optimizer settings.
    for group in ref_opt.param_groups:group['fused']=True;group['foreach']=None
    ref_sched=torch.optim.lr_scheduler.LambdaLR(ref_opt,lambda step:.999**step)
    ref_sched.load_state_dict(scheduler.state_dict())
    for _ in range(300):
        reference.model(images).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),.01)
        ref_opt.step();ref_opt.zero_grad(set_to_none=False);ref_sched.step()
        runner.replay(batch)
    torch.cuda.synchronize()
    for a,b in zip(train.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
    for a,b in zip(train.parameters(),reference.parameters()):
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[a][key],ref_opt.state[b][key].to(a.device),rtol=1e-4,atol=2e-6)
    assert runner.report.replays==300 and runner.report.captures==1
