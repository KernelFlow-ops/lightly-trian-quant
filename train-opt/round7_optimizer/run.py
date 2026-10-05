"""Common-check entrypoint for capturable optimizer candidates."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.bench import gpu_lock, write_results, empty_result, assert_gpu_exclusive
from common.check import step_equivalence
from common.config import B_AUTO
from common.harness import RunCfg, build, make_loader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('vitb16','vitl16'),required=True)
    parser.add_argument('--variant',choices=('capturable_torch','capturable_shadow','multi_tensor'),default='capturable_torch')
    parser.add_argument('--phase',choices=('early','mid','late'),default='mid')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    ref=RunCfg(model=args.model,phase=args.phase,num_workers=4)
    opt=RunCfg(model=args.model,phase=args.phase,num_workers=4,patches=('round7_optimizer',),patch_kwargs={'round7_optimizer':{'variant':args.variant}})
    with gpu_lock():
        assert_gpu_exclusive()
        loader_ctx = build(ref)
        batch = next(iter(make_loader(loader_ctx, "train")))
        del loader_ctx
        result=step_equivalence(ref,opt,[batch])
    out=args.output or Path(__file__).resolve().parent/'profile'/f'check_{args.model}_{args.variant}_{args.phase}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    row=empty_result(round_name='round7_optimizer',variant=args.variant,model=args.model,batch=B_AUTO[args.model],precision=ref.precision,patches=opt.patches,desc='Capturable Torch fused AdamW fallback')
    row['accuracy']['level_b']={key:result[key] for key in ('loss_rel','grad_cos_min','grad_norm_rel','update_cos_min','pass')}
    row['logs']=[str(out)]
    write_results(Path(__file__).resolve().parent/'results.json',{args.model:{args.variant:row}})
    print(json.dumps(result,indent=2))
    if not result['pass']: raise SystemExit(1)

if __name__=='__main__': main()
