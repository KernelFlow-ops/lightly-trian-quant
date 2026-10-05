"""Re-fill the generated tables of NOTES.md (between <!-- TABLE:name --> markers) from profile/tables_r23.md.  usage: python fill_notes.py <NOTES.md> <tables_r23.md>"""
import re, sys
notes, tables = sys.argv[1], sys.argv[2]
t = open(tables, encoding='utf-8').read(); s = open(notes, encoding='utf-8').read()
def sect(h):
    i = t.index(h); j = t.find('\n### ', i + 5)
    return t[i:j if j > 0 else None].split('\n', 1)[1].strip('\n')
for name, head in (('bench', '### 整步速度表'), ('gate_s', '### 门 S'), ('graph_equiv', '### graph_equiv'), ('gate_b', '### 门 B（同进程'), ('gate_b_extra', '### 门 B 补充')):
    body = sect(head)
    s = re.sub(rf'<!-- TABLE:{name} -->.*?<!-- /TABLE:{name} -->', lambda m: f'<!-- TABLE:{name} -->\n{body}\n<!-- /TABLE:{name} -->', s, flags=re.S)
open(notes, 'w', encoding='utf-8', newline='\n').write(s)
print('filled')
