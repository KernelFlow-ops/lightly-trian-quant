"""Render the 16 hand-drawn figures used by the top-level README.

    python assets/make_figures.py            # all
    python assets/make_figures.py infer_03   # only figures whose name contains the argument(s)

Needs only matplotlib + numpy and the Comic Sans MS font (any hand-writing font works: edit figlib/style.py).
Every number in the figures is copied from the NOTES / results.json of infer-opt and train-opt; the source of
each figure is printed in its footer.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from figlib import infer_figs  # noqa: E402,F401
from figlib.style import build  # noqa: E402

try:
    from figlib import train_figs  # noqa: E402,F401
except ImportError:  # train figures are optional while developing
    pass

if __name__ == "__main__":
    build(sys.argv[1:] or None)
