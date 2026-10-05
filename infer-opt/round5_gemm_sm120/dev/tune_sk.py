"""Stream-K sweep for proj / fc2 (dev).  usage: python tune_sk.py <fc2|proj> <vitl16|vitb16> <M>"""
import sys, itertools, json, time
sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120/dev"); sys.path.insert(0,"/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0,"/root/lightly-train/infer-opt/common")
import torch, mgb
from kt import kt
from multiprocessing import Pool
kind, model, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
d, ff = {"vitl16": (1024, 4096), "vitb16": (768, 3072)}[model]
N, K = (d, ff) if kind == "fc2" else (d, d)
EX = "-DORDER=1 -DUNR=2 -DFRAGALL=1"
def feasible(c):
    NT=c["wm"]*c["wn"]*32
    if c["bm"]%c["wm"] or c["bn"]%c["wn"]: return False
    wtm,wtn=c["bm"]//c["wm"],c["bn"]//c["wn"]
    if wtm%16 or wtn%16: return False
    if c["st"]*(c["bm"]+c["bn"])*c["bk"]*2>98*1024: return False
    if wtm//16*(wtn//8)*4>128: return False
    return 2<=c["wm"]*c["wn"]<=12
def grid():
    out=[]
    for bm,bn in itertools.product((64,80,96,112,128,144,160,192,256),(32,48,64,80,96,112,128,192,256)):
        if bm*bn<4096 or bm*bn>256*128: continue
        for bk,st,wm,wn in itertools.product((32,64),(2,3),(1,2,3,4,5,6,8),(1,2,3,4,6)):
            c=dict(epi=1,rf32=1,bm=bm,bn=bn,bk=bk,st=st,wm=wm,wn=wn,minb=3,streamk=1,skm=0,extra=EX+(" -DNFULL=1" if N%bn==0 else ""))
            if feasible(c): out.append(c)
    return out
def comp(c):
    try: mgb.compile_cubin(c); return None
    except Exception as e: return str(e)[-200:]
if __name__=="__main__":
    cfgs=grid(); print(len(cfgs),"configs",flush=True)
    t0=time.time()
    with Pool(16) as p: errs=p.map(comp,cfgs,chunksize=1)
    print("compile %.0fs fails %d"%(time.time()-t0,sum(1 for e in errs if e)),flush=True)
    dev,bf="cuda",torch.bfloat16
    r=lambda *s: (torch.randn(*s,device=dev)*0.03).to(bf)
    h,W,b=r(M,K),r(N,K),r(N); g=torch.rand(N,device=dev); R=torch.randn(M,N,device=dev); C=torch.empty_like(R)
    rows=[]
    for c,e in zip(cfgs,errs):
        if e: continue
        try:
            k0=mgb.get(c); occ=k0.occupancy()
        except Exception as ex: continue
        for skm in (0,3,4):
            cc=dict(c,skm=skm); k=mgb.get(cc)
            if skm and skm<max(1,occ)-0 and skm not in (2,3): pass
            fn=lambda k=k: k(h,W,b,g,R,C)
            try:
                fn(); torch.cuda.synchronize()
                us=kt(fn,n=10,reps=5)["min"]
            except Exception as ex:
                print("FAIL",cc,str(ex)[:60]); torch.cuda.synchronize() if False else None; continue
            tf=2*M*N*K/us/1e6
            rows.append({"us":us,"tf":tf,"cfg":cc,"occ":occ})
            print(f"{us:8.1f}us {tf:6.1f}TF occ={occ} {cc['bm']}x{cc['bn']}x{cc['bk']} st{cc['st']} w{cc['wm']}x{cc['wn']} skm{skm}",flush=True)
    rows.sort(key=lambda x:x["us"])
    print("TOP10")
    for x in rows[:10]: print(f"{x['us']:8.1f}us {x['tf']:6.1f}TF occ={x['occ']} {x['cfg']}")
    out=f"/root/lightly-train/infer-opt/round5_gemm_sm120/tunesk_{kind}_{model}_M{M}.json"
    json.dump({"rows":rows},open(out,"w"),indent=1)
