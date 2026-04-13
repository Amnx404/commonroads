"""Deprecated: use `uv run python tools/cr_map.py visualise ...`."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__":
    p = Path(__file__).resolve().parent / "cr_map.py"
    spec = importlib.util.spec_from_file_location("_cr_map_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.exit(m.main(["visualise", *sys.argv[1:]]))
