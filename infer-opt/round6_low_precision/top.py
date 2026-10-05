import json
for m in ("vitl16","vitb16"):
    g=json.load(open(f"greedy_{m}_row.json"))
    d=g["single_mae_delta_e5"]; c=g["single_conf"]
    s=sorted(d.items(), key=lambda kv: kv[1])
    print(m,"most sensitive (unit, dMAE e-5, conf%):", [(k, round(v,2), round(c[k]*100,1)) for k,v in s[-10:]])
    print(m,"least sensitive:", [(k, round(v,2)) for k,v in s[:10]])
    by={}
    for k,v in d.items():
        t=k.split('.')[1]; by.setdefault(t,[]).append(v)
    print(m,"mean dMAE by type", {t: round(sum(v)/len(v),2) for t,v in by.items()})
    L=max(int(k.split('.')[0]) for k in d)+1
    print(m,"mean dMAE by layer third", [round(sum(v for k,v in d.items() if a<=int(k.split('.')[0])<b)/ (4*(b-a)),2) for a,b in [(0,L//3),(L//3,2*L//3),(2*L//3,L)]])
