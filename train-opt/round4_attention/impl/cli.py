"""Shared B-owned CLI for r4/r5/r9, wired to the actual common APIs."""
import argparse,gc,json,os,subprocess,sys
from pathlib import Path

def main(round_name,default_variant):
    # Match the original probe/train allocator before any CUDA initialization.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root));sys.path.insert(0,str(root.parent/'src'))
    from common.harness import RunCfg,build,apply_patches,preload_batches,set_phase,train_step
    from common import bench,check
    from common.config import B_AUTO,OPT_EAGER_PATCHES
    from common.rng import RngTape
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('vitl16','vitb16'),default='vitl16')
    p.add_argument('--variant',default=default_variant)
    p.add_argument('--phase',choices=('early','mid','late','all'),default='all')
    p.add_argument('--batch',type=int);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--patches',default=None);p.add_argument('--num-workers',type=int,default=25)
    p.add_argument('--patch-kwargs',default='{}',help='JSON map of base patch kwargs; e.g. r1 autocast_bf16')
    p.add_argument('--check',action='store_true');p.add_argument('--bench',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--e2e',action='store_true')
    p.add_argument('--output',type=Path,default=root/round_name/'results.json')
    a=p.parse_args()
    if not any((a.check,a.bench,a.profile,a.e2e)):p.error('Select --check, --bench, --profile or --e2e')
    batch=a.batch or B_AUTO[a.model]
    if batch is None:raise RuntimeError('B_AUTO is not published; use --batch only for explicitly labeled development checks')
    base=OPT_EAGER_PATCHES if a.patches is None else tuple(v for v in a.patches.split(',') if v)
    patches=tuple(v for v in base if v!=round_name)+(round_name,)
    try:base_kwargs=json.loads(a.patch_kwargs)
    except json.JSONDecodeError as e:p.error(f'--patch-kwargs must be JSON: {e}')
    if not isinstance(base_kwargs,dict):p.error('--patch-kwargs must be an object')
    own_kwargs={**base_kwargs.get(round_name,{}),'variant':a.variant}
    phases=('early','mid','late') if a.phase=='all' else (a.phase,)
    def cfg(phase,optimized):
        return RunCfg(model=a.model,batch_size=batch,seed=a.seed,phase=phase,num_workers=a.num_workers,
            patches=patches if optimized else base,
            patch_kwargs={**base_kwargs,round_name:own_kwargs} if optimized else dict(base_kwargs))
    def create(config):
        c=build(config)
        # build/apply split is the common published contract. Preserve
        # compatibility if A later auto-applies all patches during build.
        if config.patches and not c.meta.get('patches'):apply_patches(c)
        return c
    with bench.gpu_lock():
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
        others=[line for line in apps.splitlines() if line.strip() and int(line.split(',')[0])!=os.getpid()]
        if others:raise RuntimeError('Unknown GPU processes, measurement refused: '+ '; '.join(others))
        result=bench.empty_result(round_name=round_name,variant=a.variant,model=a.model,batch=batch,
            precision='bf16-mixed',patches=patches,desc=f'{round_name} {a.variant}')
        result['patch_kwargs']={**base_kwargs,round_name:own_kwargs}
        result['seed']=a.seed
        result['actual_forward_dtype']='bf16' if base_kwargs.get('round1_torch_cleanup',{}).get('variant')=='autocast_bf16' or 'round1_torch_cleanup' in base else 'fp32'
        if a.check:
            checks={}
            if a.output.is_file():
                previous=json.loads(a.output.read_text()).get(a.model,{}).get(a.variant,{})
                if previous.get('patches')==list(patches) and previous.get('patch_kwargs',{})==result['patch_kwargs'] and previous.get('seed',0)==a.seed:
                    checks.update(previous.get('checks_by_phase',{}))
            for phase in phases:
                c=create(cfg(phase,False));batches=preload_batches(c,k=1);del c;gc.collect()
                __import__('torch').cuda.empty_cache()
                checks[phase]=check.step_equivalence(cfg(phase,False),cfg(phase,True),batches,tape=RngTape('record'))
                del batches;gc.collect();__import__('torch').cuda.empty_cache()
            result['checks_by_phase']=checks
            result['tested_phases']=sorted(checks)
            # Keep a compact, machine-readable gate for every requested phase.
            # This avoids reducing ``--phase all`` to a single aggregate number
            # and makes partial reruns auditable when a phase is still pending.
            result['phase_summary']={phase: {
                'pass': bool(report.get('pass', False)),
                'loss_rel': report.get('loss_rel'),
                'grad_cos_min': report.get('grad_cos_min'),
                'grad_norm_rel': report.get('grad_norm_rel'),
                'update_cos_min': report.get('update_cos_min'),
            } for phase, report in sorted(checks.items())}
            result['all_phases_pass']=bool(set(phases).issubset(checks) and all(
                result['phase_summary'][phase]['pass'] for phase in phases))
            result['accuracy']['level_b']={'loss_rel':max(z['loss_rel'] for z in checks.values()),
                'grad_cos_min':min(z['grad_cos_min'] for z in checks.values()),'grad_norm_rel':max(z['grad_norm_rel'] for z in checks.values()),
                'update_cos_min':min(z['update_cos_min'] for z in checks.values()),
                'pass':result['all_phases_pass'],
                'partial_pass':all(z['pass'] for z in checks.values())}
        if a.bench or a.profile or a.e2e:
            c=create(cfg(phases[0],True));batches=preload_batches(c,k=8)
            if a.bench:
                result['gpu_step']=bench.time_steps(c,batches,phases=phases)
                result['peak_mem_gb']=max(result['gpu_step'][ph]['peak_allocated_gb'] for ph in phases)
            if a.profile:
                c.nvtx=True;c.meta['phase_override']=True
                for phase_idx,phase in enumerate(phases):
                    set_phase(c,phase)
                    for i in range(20):train_step(c,batches[i%len(batches)],phase_idx*20+i)
                def profile_step(i):
                    phase=phases[i//5]
                    if i%5==0:set_phase(c,phase)
                    with bench.nvtx(f'phase_{phase}'):
                        return train_step(c,batches[i%len(batches)],len(phases)*20+i)
                bench.profile_steps(profile_step,warmup=0,steps=5*len(phases))
                result['profiled_phases']=list(phases)
            if a.e2e:
                from common.harness import make_loader
                result['e2e']=bench.time_e2e(c,make_loader(c,'train'))
        result['env']=bench.env_info();result['command']=list(sys.argv)
        # Callers running through gpu_run/tee can provide a durable log path;
        # /proc/<pid>/fd/1 becomes a dead pipe after the process exits.
        result['logs']=[os.environ.get('TRAIN_OPT_RESULT_LOG',
            str(Path(f'/proc/{os.getpid()}/fd/1').resolve()))]
        bench.write_results(a.output,{a.model:{a.variant:result}})
        print(json.dumps(result,indent=2),flush=True)
        if a.check and not result['accuracy']['level_b']['partial_pass']:raise SystemExit(1)
