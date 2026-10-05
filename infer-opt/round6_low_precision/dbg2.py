import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'impl')
import sens, fp8sim, fp8linear as fl, head as hd
ev = sens.Evaluator('vitl16')
u = [tuple(x) for x in json.load(open('greedy_vitl16_row_noproj.json'))['chosen_units']]
def show(tag, r): print(f"{tag:28s} iou {r['pooled_iou']:.4f} min {r['min_iou']:.4f} conf {r['conf_rel_max']*100:6.2f}% mae {r['logit_mae']*1e5:.2f}e-5", flush=True)
show('baseline', ev.eval_cfg([], 'row'))
show('sim', ev.eval_cfg(u, 'row'))
fl.enable(ev.model, u); show('real', ev.run()); fl.disable(ev.model)
hd.enable(ev.model); show('fusedhead only', ev.run())
show('sim+fusedhead', ev.eval_cfg(u, 'row'))
fl.enable(ev.model, u); show('real+fusedhead', ev.run())
# per-frame conf errors
