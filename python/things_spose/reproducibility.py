"""Extended Data Figure 1: reproducibility of the 49 SPoSE dimensions across 20
independent model fits (ported from ``make_figures_behavsim.m`` lines 1015-1088).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import paths


def _corr_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Full cross-correlation matrix between columns of A (n, p) and B (n, q)."""
    Az = A - A.mean(0)
    Bz = B - B.mean(0)
    Az /= np.linalg.norm(Az, axis=0, keepdims=True)
    Bz /= np.linalg.norm(Bz, axis=0, keepdims=True)
    return Az.T @ Bz


def load_reference_models(sortind0: np.ndarray) -> list[np.ndarray]:
    """Load the last-iteration embedding from each ``reference_models/sNN`` dir,
    drop all-tiny dimensions, and re-sort into the canonical object order."""
    models = []
    for i in range(1, 21):
        d = paths.REFERENCE_MODELS_DIR / f"s{i:02d}"
        txts = sorted(d.glob("*.txt"))
        tmp = np.loadtxt(txts[-1])
        tmp = tmp[:, np.any(tmp > 0.1, axis=0)]   # remove empty dimensions
        models.append(tmp[sortind0])              # apply sortind
    return models


@dataclass
class ReproducibilityResult:
    reproducibility: np.ndarray       # (49, 20) best match per dim per model
    mean: np.ndarray                  # (49,) Fisher-z averaged, back-transformed
    lower: np.ndarray
    upper: np.ndarray
    rank_corr: float                  # corr(dim number, reproducibility rank)
    rank_p: float
    rank_ci95: tuple[float, float]


def dimension_reproducibility(embedding: np.ndarray, sortind0: np.ndarray,
                              seed_perm: int = 1, seed_boot: int = 2,
                              n_perm: int = 100_000) -> ReproducibilityResult:
    models = load_reference_models(sortind0)

    # For each model, the best matching reference dimension for each SPoSE dim.
    repro = np.column_stack([
        _corr_cols(embedding, m).max(axis=1) for m in models
    ])  # (49, 20)

    mean_z = np.arctanh(repro).mean(axis=1)
    ci = 1.96 * np.arctanh(repro).std(axis=1, ddof=0) / np.sqrt(repro.shape[1])
    mean = np.tanh(mean_z)
    lower = np.tanh(mean_z - ci)
    upper = np.tanh(mean_z + ci)

    # Rank of reproducibility vs dimension number (dims are ordered by weight).
    repro_ind = np.argsort(mean)[::-1]                 # descending
    n = mean.size
    rank_corr = float(np.corrcoef(np.arange(1, n + 1), repro_ind + 1)[0, 1])

    rng = np.random.default_rng(seed_perm)
    perms = np.argsort(rng.random((n, n_perm)), axis=0) + 1
    r_perm = _corr_cols(perms.astype(float), (repro_ind + 1).astype(float)[:, None]).ravel()
    rank_p = float((np.concatenate([r_perm, [rank_corr]]) >= rank_corr).mean())

    rng2 = np.random.default_rng(seed_boot)
    idx = rng2.integers(1, n + 1, size=(n, 1000))
    r_boot = np.array([
        np.corrcoef(idx[:, b], (repro_ind + 1)[idx[:, b] - 1])[0, 1] for b in range(1000)
    ])
    s = np.std(np.arctanh(r_boot), ddof=0)
    rank_ci95 = (float(np.tanh(np.arctanh(rank_corr) - 1.96 * s)),
                 float(np.tanh(np.arctanh(rank_corr) + 1.96 * s)))

    return ReproducibilityResult(repro, mean, lower, upper, rank_corr, rank_p, rank_ci95)
