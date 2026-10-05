import json, sys
m, src, n, out = sys.argv[1:5]
d = json.load(open(src))
json.dump({"chosen_units": d["order"][:int(n)]}, open(out, "w"))
