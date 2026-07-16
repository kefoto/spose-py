"""Benjamini-Hochberg FDR correction, ported from ``fdr_bh.m`` (Groppe).

Only the pieces used by the paper are ported: the ``pdep`` (BH, independence /
positive-dependence) method returning the significance mask and adjusted p-values.
"""
from __future__ import annotations

import numpy as np


def fdr_bh(pvals, q: float = 0.05, method: str = "pdep"):
    """Return ``(h, adj_p)``.

    * ``h``     : boolean mask of hypotheses that survive FDR control at ``q``.
    * ``adj_p`` : BH-adjusted p-values (same shape as input).
    """
    p = np.asarray(pvals, dtype=float).ravel()
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1].")

    m = p.size
    sort_ids = np.argsort(p, kind="stable")
    p_sorted = p[sort_ids]
    unsort_ids = np.argsort(sort_ids, kind="stable")
    ranks = np.arange(1, m + 1)

    if method == "pdep":
        thresh = ranks * q / m
        wtd_p = m * p_sorted / ranks
    elif method == "dep":
        denom = m * np.sum(1.0 / ranks)
        thresh = ranks * q / denom
        wtd_p = denom * p_sorted / ranks
    else:
        raise ValueError("method must be 'pdep' or 'dep'")

    # Monotone step-up adjustment (mirrors the MATLAB nextfill loop).
    adj_p = np.full(m, np.nan)
    wtd_p_sindex = np.argsort(wtd_p, kind="stable")
    wtd_p_sorted = wtd_p[wtd_p_sindex]
    nextfill = 0
    for k in range(m):
        if wtd_p_sindex[k] >= nextfill:
            adj_p[nextfill : wtd_p_sindex[k] + 1] = wtd_p_sorted[k]
            nextfill = wtd_p_sindex[k] + 1
            if nextfill > m - 1:
                break
    adj_p = np.minimum(adj_p[unsort_ids], 1.0)

    rej = p_sorted <= thresh
    h = np.zeros(m, dtype=bool)
    if rej.any():
        max_id = np.max(np.flatnonzero(rej))
        crit = p_sorted[max_id]
        h = p <= crit

    return h.reshape(np.shape(pvals)), adj_p.reshape(np.shape(pvals))
