"""Filesystem locations for the SPoSE dataset and generated artifacts.

The raw MATLAB dataset lives in ``osfstorage-archive/`` at the repository root.
This module locates it robustly whether the code is imported from a notebook,
a script, or an installed copy, and lets the location be overridden on a
cluster via the ``THINGS_ARCHIVE_DIR`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

# python/things_spose/paths.py -> repo root is three levels up.
_PKG_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _PKG_DIR.parent
_REPO_ROOT = _PYTHON_DIR.parent


def _resolve_archive_dir() -> Path:
    """Locate ``osfstorage-archive`` (env override wins)."""
    env = os.environ.get("THINGS_ARCHIVE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.name != "osfstorage-archive" and (p / "osfstorage-archive").is_dir():
            p = p / "osfstorage-archive"
        return p

    candidates = [
        _REPO_ROOT / "osfstorage-archive",
        _PYTHON_DIR / "osfstorage-archive",
        Path.cwd() / "osfstorage-archive",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    # Fall back to the primary expectation so error messages are informative.
    return (_REPO_ROOT / "osfstorage-archive").resolve()


ARCHIVE_DIR: Path = _resolve_archive_dir()
DATA_DIR: Path = ARCHIVE_DIR / "data"
VARIABLE_DIR: Path = ARCHIVE_DIR / "variables"
REFERENCE_MODELS_DIR: Path = ARCHIVE_DIR / "reference_models"

# Generated artifacts (rebuildable) live next to the Python code.
CACHE_DIR: Path = Path(
    os.environ.get("THINGS_CACHE_DIR", _PYTHON_DIR / "cache")
).expanduser().resolve()


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def data(name: str) -> Path:
    """Path to a file in ``osfstorage-archive/data``."""
    return DATA_DIR / name


def variable(name: str) -> Path:
    """Path to a file in ``osfstorage-archive/variables``."""
    return VARIABLE_DIR / name


def check_archive() -> None:
    """Raise a clear error if the dataset cannot be found."""
    if not DATA_DIR.is_dir() or not VARIABLE_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not locate the SPoSE dataset. Expected 'data/' and "
            f"'variables/' under {ARCHIVE_DIR}. Set THINGS_ARCHIVE_DIR to the "
            f"path of your 'osfstorage-archive' directory."
        )
