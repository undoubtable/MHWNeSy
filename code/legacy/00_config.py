"""Compatibility shim for legacy scripts moved under code/legacy.

Legacy scripts load a sibling 00_config.py via SourceFileLoader. This file
forwards those imports to the GitHub-facing config in code/00_config.py.
"""

from pathlib import Path
import importlib.util

_PARENT_CONFIG = Path(__file__).resolve().parents[1] / "00_config.py"
_SPEC = importlib.util.spec_from_file_location("_mhwneurrl_config", _PARENT_CONFIG)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

for _name in dir(_MOD):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MOD, _name)
