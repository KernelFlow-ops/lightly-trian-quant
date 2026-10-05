import json, sys
for path in sys.argv[1:]:
    print("==", path)
    for l in open(path):
        r = json.loads(l)
        if "error" in r: print(r["name"], "ERROR", r["error"][-100:]); continue
        m = r["M"]; h = r.get("H")
        s = f"{r['name']:46s} M iou={m['pooled_iou']:.4f} min={m['min_iou']:.3f} conf={m['conf_max']*100:5.2f}% mae={m['logit_mae']:.2e} {'PASS' if m['pass'] else 'fail'}"
        if h: s += f" | H iou={h['pooled_iou']:.4f} min={h['min_iou']:.3f} conf={h['conf_max']*100:5.2f}% >3%:{h['n_conf_gt3']:2d} mae={h['logit_mae']:.2e} {'PASS' if h['pass'] else 'fail'}"
        print(s)
