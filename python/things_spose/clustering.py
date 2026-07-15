"""Port of ``clustering_algorithm.m`` -- a visualization-only ordering that groups
objects into "families" by their most dominant SPoSE dimensions. Used to sort the
similarity matrix in Figure 1.
"""
from __future__ import annotations

import numpy as np


def clustering_algorithm(n_iter: int, cutoff: int, embedding: np.ndarray) -> np.ndarray:
    """Return an ordering ``ind`` of objects grouped by dominant dimensions.

    Mirrors the (current, non-commented) MATLAB implementation: build a family
    key from each object's top ``n_iter`` dimensions, then repeatedly collapse
    families with fewer than ``cutoff`` members into coarser groups.
    """
    E = np.asarray(embedding)
    n = E.shape[0]

    # sort ascending; MATLAB `i` are 1-based dimension indices of the sort.
    isort = np.argsort(E, axis=1, kind="stable") + 1          # (n, D) 1-based
    top = np.sort(isort[:, -n_iter:], axis=1)                  # top n_iter dims, ascending

    fams = top[:, n_iter - 1].astype(np.int64).copy()
    for i_iter in range(1, n_iter):
        k_iter = n_iter - i_iter
        fams = fams + top[:, i_iter - 1].astype(np.int64) * 10 ** (2 * k_iter)

    for i_iter in range(1, n_iter + 1):
        ufams, counts = np.unique(fams, return_counts=True)
        small = ufams[counts < cutoff]
        if small.size:
            mult = 10 ** (2 * i_iter)
            add = 99 * 10 ** (2 * (i_iter - 1))
            mask = np.isin(fams, small)
            fams[mask] = (fams[mask] // mult) * mult + add

    # Final pass: push remaining tiny clusters to the very end.
    ufams, counts = np.unique(fams, return_counts=True)
    small = ufams[counts < cutoff]
    if small.size:
        mask = np.isin(fams, small)
        fams[mask] = fams[mask] + 9_000_000

    ind = np.argsort(fams, kind="stable")
    return ind
