"""Allow repository CLI scripts to run from an unpacked checkout before installation.

Installed-package execution remains unchanged. This file only prepends ``src`` when
``puma_exploration6`` cannot already be imported.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def bootstrap_repo() -> None:
    if importlib.util.find_spec("puma_exploration6") is not None:
        return
    src = Path(__file__).resolve().parents[1] / "src"
    if not (src / "puma_exploration6").is_dir():
        raise ModuleNotFoundError(f"puma_exploration6 is neither installed nor present under {src}")
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
