"""Rebuildable cache artifacts and their load helpers.

The heavy precompute (49 ``embedding2sim`` evaluations for the Figure-6 ablation,
plus the Figure-4/5 t-SNE layout) is produced once by ``scripts/build_cache.py``
and written under :data:`things_spose.paths.CACHE_DIR` (override with
``THINGS_CACHE_DIR``). This module centralises the filenames and the load side so
notebooks and analysis scripts can pull the artifacts without re-deriving paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import paths

REDUCED_EMB_FILE = "reduced_embeddings.npy"        # (49, N, 49) float32
REDUCED_PAIRVEC_FILE = "reduced_sim_pairvecs.npy"  # (49, n_pairs) float32
TSNE_LAYOUT_FILE = "tsne_layout.npy"               # (N, 2) float64
SPOSE_SIM_FILE = "spose_sim.npy"                   # (N, N) recomputed similarity


def _path(name: str):
    return paths.CACHE_DIR / name


def exists(name: str) -> bool:
    return _path(name).is_file()


@dataclass
class AblationCache:
    reduced_embeddings: np.ndarray   # (49, N, 49)
    reduced_sim_pairvecs: np.ndarray  # (49, n_pairs)


def load_ablation_cache() -> AblationCache:
    """Load the Fig-6 ablation artifacts (raises if the cache is missing)."""
    emb_p, vec_p = _path(REDUCED_EMB_FILE), _path(REDUCED_PAIRVEC_FILE)
    if not emb_p.is_file() or not vec_p.is_file():
        raise FileNotFoundError(
            f"Ablation cache not found under {paths.CACHE_DIR}. "
            f"Run: python scripts/build_cache.py"
        )
    return AblationCache(
        reduced_embeddings=np.load(emb_p),
        reduced_sim_pairvecs=np.load(vec_p),
    )


def load_tsne_layout() -> np.ndarray:
    """Load the cached Figure-4/5 t-SNE layout (raises if missing)."""
    p = _path(TSNE_LAYOUT_FILE)
    if not p.is_file():
        raise FileNotFoundError(
            f"t-SNE layout not found under {paths.CACHE_DIR}. "
            f"Run: python scripts/build_cache.py --tsne"
        )
    return np.load(p)
