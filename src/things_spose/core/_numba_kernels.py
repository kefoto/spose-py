"""Numba-compiled CPU kernels. Imported lazily so the package works without Numba.

The kernel is the parallel fused version of the ``embedding2sim`` triple loop:
``prange`` over rows ``i`` (all cores), inner ``j > i`` and ``k`` loops that
accumulate ``e_ij / (e_ij + e_ik + e_jk)`` with no N*N temporaries, writing the
symmetric pair once.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def emb2sim_kernel(esim):  # pragma: no cover - compiled
    n = esim.shape[0]
    cp = np.zeros((n, n), dtype=esim.dtype)
    inv = 1.0 / (n - 2)
    for i in prange(n):
        row_i = esim[i]
        for j in range(i + 1, n):
            eij = esim[i, j]
            row_j = esim[j]
            s = 0.0
            for k in range(n):
                if k == i or k == j:
                    continue
                s += eij / (eij + row_i[k] + row_j[k])
            v = s * inv
            cp[i, j] = v
            cp[j, i] = v
    return cp
