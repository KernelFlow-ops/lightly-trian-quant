"""Export only this agent's compiled kernels; requires no GPU execution."""
import json,re,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CACHE=Path('/root/autodl-tmp/train-opt-tmp/B/triton')
TOOLS=Path('/usr/local/cuda/bin')
NAMES={
 'round4_attention':{'_forward','_delta','_backward_q','_backward_kv','_rotate','_build'},
 'round5_block_fusion':{'_fwd','_bwd','_gelu','_dgelu'},
 'round9_low_precision':{'_quant'},
}
for round_,names in NAMES.items():
    out=ROOT/round_/'profile/static';out.mkdir(parents=True,exist_ok=True);records=[]
    for p in sorted(CACHE.rglob('*.json')):
        info=json.loads(p.read_text());name=info.get('name')
        if name not in names:continue
        cubin=p.with_suffix('.cubin')
        if not cubin.exists():continue
        ident=f'{name}_{p.parent.name[:12]}'
        resources=subprocess.run([str(TOOLS/'cuobjdump'),'--dump-resource-usage',str(cubin)],capture_output=True,text=True)
        sass=subprocess.run([str(TOOLS/'cuobjdump'),'--dump-sass',str(cubin)],capture_output=True,text=True)
        (out/f'{ident}.resources.txt').write_text(resources.stdout+resources.stderr)
        (out/f'{ident}.sass').write_text(sass.stdout+sass.stderr)
        ptx=p.with_suffix('.ptx')
        if ptx.exists():shutil.copyfile(ptx,out/f'{ident}.ptx')
        reg=re.search(r'REG:(\d+)',resources.stdout);stack=re.search(r'STACK:(\d+)',resources.stdout)
        record={'name':name,'cache':str(p.parent),'num_warps':info['num_warps'],'num_stages':info['num_stages'],
                'shared_bytes':info.get('shared'),'registers':None if reg is None else int(reg.group(1)),
                'stack_bytes':None if stack is None else int(stack.group(1)),
                'sass_counts':{token:len(re.findall(r'\b'+token+r'\b',sass.stdout)) for token in ('HMMA','QMMA','UTMALDG','LDSM','LDGSTS','LDL','STL','ATOM')},
                'notes':'Static counts, not executed instruction counts; occupancy requires launch-specific CUDA API check.'}
        records.append(record)
    (out/'summary.json').write_text(json.dumps({'kernels':records},indent=2))
    print(round_,len(records))
