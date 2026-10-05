"""usage: sass_mix.py file.sass FUNC_SUBSTR  -> opcode histogram of one function (static instruction counts)"""
import re, sys, collections
txt = open(sys.argv[1]).read().split("Function : ")
sel = [t for t in txt if t.startswith(sys.argv[2])]
for t in sel:
    name = t.split("\n")[0]
    c = collections.Counter(); n = 0
    for line in t.split("\n"):
        m = re.match(r"\s+/\*[0-9a-f]{4}\*/\s+(@!?U?P\w\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            op = m.group(2); base = op.split(".")[0]
            c[base] += 1; n += 1
    print(name, "total instr", n)
    print("  " + "  ".join(f"{k}:{v}" for k, v in c.most_common(40)))
