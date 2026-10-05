"""NOTES.md = NOTES_template.md with {{name}} / {{name:arg}} placeholders replaced by tables.py generators.   python build_notes.py"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
import tables  # noqa: E402

t = (HERE / "NOTES_template.md").read_text(encoding="utf-8")


def sub(m):
    name, _, arg = m.group(1).partition(":")
    fn = tables.GEN[name]
    try:
        return fn(arg) if arg else fn()
    except Exception as ex:  # noqa: BLE001
        return f"(table {m.group(1)} failed: {type(ex).__name__}: {ex})"


out = re.sub(r"\{\{([^}]+)\}\}", sub, t)
(HERE / "NOTES.md").write_text(out, encoding="utf-8")
print("NOTES.md written", len(out))
