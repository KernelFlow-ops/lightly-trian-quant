import sys, re
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
import mgb
EX="-DORDER=1 -DUNR=2 -DFRAGALL=1"
C={"qkv lnf":dict(epi=2,lnf=1,bm=64,bn=128,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX+" -DSTG=1"),
   "qkv plain":dict(epi=2,bm=64,bn=128,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX+" -DSTG=1"),
   "fc1 lnf":dict(epi=0,lnf=1,bm=96,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX+" -DSTG=1"),
   "fc1 plain":dict(epi=0,bm=96,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX+" -DSTG=1"),
   "proj stats":dict(epi=1,stats=1,bm=64,bn=64,bk=32,st=2,wm=1,wn=4,minb=3,extra=EX),
   "proj plain":dict(epi=1,bm=64,bn=64,bk=32,st=2,wm=1,wn=4,minb=3,extra=EX),
   "fc2 stats":dict(epi=1,stats=1,splitk=2,bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX),
   "fc2 plain":dict(epi=1,splitk=2,bm=64,bn=64,bk=32,st=2,wm=2,wn=2,minb=3,extra=EX)}
for k,c in C.items():
    p,log,_=mgb.compile_cubin(c)
    t=open(log).read()
    print("%-11s regs %s spill %s"%(k,re.search(r"Used (\d+) registers",t).group(1), re.search(r"(\d+) bytes spill stores",t).group(1)))
