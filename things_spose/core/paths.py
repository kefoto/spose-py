"""Filesystem locations for the SPoSE dataset and generated artifacts.

The dataset lives under ``data/`` at the repository root: ``data/spose/`` holds the
MATLAB-derived files the analyses read, and ``data/triplet_dataset/`` holds the raw
odd-one-out triplets used for training. This module locates them robustly whether the
code is imported from a notebook, a script, or an installed copy, and lets the location
be overridden on a cluster via the ``THINGS_DATA_DIR`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

# things_spose/core/paths.py -> repo root is three levels up.
_CORE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CORE_DIR.parent
_REPO_ROOT = _PKG_DIR.parent


def _resolve_spose_dir() -> Path:
    """Locate ``data/spose`` (env override wins)."""
    env = os.environ.get("THINGS_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.name != "spose" and (p / "spose").is_dir():
            p = p / "spose"
        return p

    candidates = [
        _REPO_ROOT / "data" / "spose",
        Path.cwd() / "data" / "spose",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    # Fall back to the primary expectation so error messages are informative.
    return (_REPO_ROOT / "data" / "spose").resolve()


SPOSE_DIR: Path = _resolve_spose_dir()
DATA_DIR: Path = SPOSE_DIR / "data"
VARIABLE_DIR: Path = SPOSE_DIR / "variables"
REFERENCE_MODELS_DIR: Path = SPOSE_DIR / "reference_models"

# Raw odd-one-out triplets (train/validation/test splits) used by train.py.
TRIPLET_DIR: Path = Path(
    os.environ.get("THINGS_TRIPLET_DIR", _REPO_ROOT / "data" / "triplet_dataset")
).expanduser().resolve()

# Generated artifacts (rebuildable) live next to the Python code.
CACHE_DIR: Path = Path(
    os.environ.get("THINGS_CACHE_DIR", _REPO_ROOT / "cache")
).expanduser().resolve()


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def data(name: str) -> Path:
    """Path to a file in ``data/spose/data``."""
    return DATA_DIR / name


def variable(name: str) -> Path:
    """Path to a file in ``data/spose/variables``."""
    return VARIABLE_DIR / name


def triplets(name: str) -> Path:
    """Path to a file in ``data/triplet_dataset``."""
    return TRIPLET_DIR / name


def check_data() -> None:
    """Raise a clear error if the dataset cannot be found."""
    if not DATA_DIR.is_dir() or not VARIABLE_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not locate the SPoSE dataset. Expected 'data/' and "
            f"'variables/' under {SPOSE_DIR}. Set THINGS_DATA_DIR to the "
            f"path of your 'spose' data directory."
        )
