from __future__ import annotations
from round9_low_precision.impl.fp8 import ScalingState,linear
from round9_low_precision.impl.mxfp8 import MXScalingState

NAME='round9_low_precision'
REQUIRES=()
CONFLICTS=()

def apply(ctx,variant='fp8_mlp_cur',layers=None,units=None,**kw):
    if variant not in ('fp8_mlp_cur','fp8_mlp_delayed','fp8_all','mxfp8_mlp','fp8_mlp_tensor'):
        raise ValueError('Unknown FP8 variant')
    if kw:raise ValueError(f'Unknown low precision options: {sorted(kw)}')
    if ctx.meta.get('round5_block_fusion') in ('lt_erf','lt_tanh'):raise ValueError('FP8 Linear cannot replace the fused Linear+GELU epilogue')
    if ctx.meta.get(NAME)==variant:return {'ops':['block'],'desc':variant}
    n=len(ctx.model.backbone.blocks)
    # First/last two, joint blocks and all heads stay bf16 by default.
    selected=set(range(2,n-max(2,ctx.model.num_joint_blocks))) if layers is None else set(layers)
    types=('qkv','proj','fc1','fc2') if variant=='fp8_all' else ('fc1','fc2')
    if units is not None:
        if any(name not in types for name in units):raise ValueError(f'Units for {variant} must be chosen from {types}')
        types=tuple(units)
    base=ctx.ops.block
    states={}
    for i in selected:
        block=ctx.model.backbone.blocks[i]
        by_name={'qkv':block.attn.qkv,'proj':block.attn.proj,'fc1':block.mlp.fc1,'fc2':block.mlp.fc2}
        for name in types:
            module=by_name[name]
            state=MXScalingState() if variant=='mxfp8_mlp' else ScalingState(delayed=variant=='fp8_mlp_delayed',tensorwise=variant=='fp8_mlp_tensor');states[i,name]=state
            def forward(x,module=module,state=state):return linear(x,module,state)
            module.forward=forward
    # Preserve prior block fusion and attention composition. MLP linears are
    # runtime function overrides; the original Parameter objects are reused.
    ctx.ops.block=base
    ctx.meta[NAME]=variant;ctx.meta['fp8_layers']=sorted(selected)
    ctx.meta['fp8_units']=types
    ctx.meta['fp8_states']=states
    if variant=='fp8_mlp_delayed':ctx.meta['noncapturable_reason']='Delayed scaling is capturable at operator level; full graph scaling-state restoration has not passed C/D'
    return {'ops':['block'],'desc':f'{variant}, layers={sorted(selected)}; experimental until C/D passes'}
