"""Reference schemes. Each file is a plain scheme module (see ../scheme.py)."""
from pathlib import Path

HERE = Path(__file__).parent
NAMES = ("baseline", "diskann", "pipeann", "pageann", "haven", "starling", "cagra")


def path(name):
    return HERE / (name + ".py")
