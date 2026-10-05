"""Ahead-of-time FP32 FA compile for sm_120; no GPU execution/context needed."""
import json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from impl.attention import _forward,_backward_q,_backward_kv

result=[]
target=GPUTarget('cuda',120,32)
for bm,bn in ((32,32),(16,32),(32,16)):
    constants={'N':1129,'H':16,'PREFIX':105,'Q':100,'ROPE':True,'MASKED':True,'BM':bm,'BN':bn}
    for fn in (_forward,_backward_q,_backward_kv):
        args=[name for name in fn.arg_names if name not in constants]
        signature={name:('*i1' if name=='MASK' else '*fp32') for name in args}
        row={'kernel':fn.__name__,'bm':bm,'bn':bn}
        try:
            kernel=triton.compile(ASTSource(fn,signature,constants),target=target,
                                 options={'num_warps':4,'num_stages':2,'enable_fp_fusion':False})
            row.update({'compiled':True,'shared_bytes':kernel.metadata.shared,'name':kernel.metadata.name})
        except Exception as e:row.update({'compiled':False,'error':str(e)})
        result.append(row);print(json.dumps(row),flush=True)
(R/'profile/fp32_compile.json').write_text(json.dumps({'compiles':result},indent=2)+'\n')
