"""Run a Python entry point while loading NumPy-2 pickles on NumPy 1.x.

This only installs in-process module aliases; it does not modify either NumPy
environment or the frozen model checkpoint.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import numpy as np


def install_numpy2_pickle_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    for name in (
        "_multiarray_umath",
        "multiarray",
        "numeric",
        "numerictypes",
        "overrides",
        "shape_base",
        "umath",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: numpy_pickle_compat.py ENTRYPOINT [ARGS ...]")
    install_numpy2_pickle_aliases()
    entrypoint = sys.argv[1]
    sys.path.insert(0, str(Path(entrypoint).resolve().parent))
    sys.argv = sys.argv[1:]
    runpy.run_path(entrypoint, run_name="__main__")
