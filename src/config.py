"""Load and expose project configuration from config.yaml.

Every path, parameter, and column name lives in config.yaml so that no
module hardcodes them. Import `CFG` for the loaded dict, or call
`resolve(...)` to turn a config-relative path into an absolute one that
works on any machine.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import yaml

# Project root, derived from __file__, so there are no environment-specific absolute paths.
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | Path = "config.yaml") -> dict:
    # Read and cache config.yaml
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    # Turn a path from config.yaml into an absolute path
    return ROOT / rel_path


# Loaded once on import for convenience: "from .config import CFG"
CFG = load_config()
