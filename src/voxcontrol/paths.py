"""Where the application reads its bundled files and writes caches, models and results.

The packaged application never writes to AppData: caches live in a subfolder of the
application folder, or in Documents/VoxControlResearch when that folder is read-only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import SOFTWARE_NAME


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Folder with the bundled configs (application folder, or the working directory from source)."""
    return Path(sys.executable).resolve().parent if frozen() else Path.cwd()


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def writable_root() -> Path:
    root = app_root()
    if _writable(root):
        return root
    docs = Path.home() / "Documents" / SOFTWARE_NAME
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def configure_caches() -> Path:
    """Point library caches (matplotlib, speech models) to <writable root>/cache when packaged."""
    cache = writable_root() / "cache"
    if frozen():
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        os.environ.setdefault("HF_HOME", str(cache / "hf"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    return cache
