"""Reference schemes. Each file is a plain scheme module (see ../scheme.py)."""
from pathlib import Path

HERE = Path(__file__).parent
NAMES = ("baseline", "diskann", "pipeann", "pageann", "haven", "starling", "cagra",
         "compass", "atlas", "router", "reorder", "entry", "batch", "cache")  # the last seven: our graph schemes, best first


def path(name):
    return HERE / (name + ".py")
