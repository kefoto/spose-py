"""Leave-one-out category classification, ported from ``predict_category.m``.

Predicts each object's category from the nearest class centroid (Euclidean),
recomputing centroids with the test object held out. Run for both the SPoSE
embedding and the semantic word vectors (the paper's baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 1-based indices from predict_category.m, converted to 0-based below.
_RMCAT = np.array([2, 5, 7, 8, 11, 14, 25]) - 1        # multi-membership categories
_RMCAT2 = np.array([9, 10]) - 1                         # too-few-items (after 1st removal)


@dataclass
class ClassificationResult:
    accuracy_spose: float
    accuracy_wordvec: float
    labels: np.ndarray            # (n,) true category label per retained object
    pred_spose: np.ndarray
    pred_wordvec: np.ndarray
    categories: list[str]


def _reduce(ds):
    catmat = ds.category_mat_manual.astype(bool).copy()
    categories = list(ds.categories27)

    keep1 = [c for c in range(catmat.shape[1]) if c not in set(_RMCAT.tolist())]
    catmat = catmat[:, keep1]
    categories = [categories[c] for c in keep1]

    catmat[catmat.sum(axis=1) > 1, :] = False           # drop multi-category objects

    keep2 = [c for c in range(catmat.shape[1]) if c not in set(_RMCAT2.tolist())]
    catmat = catmat[:, keep2]
    categories = [categories[c] for c in keep2]

    keep_obj = catmat.any(axis=1)
    catmat = catmat[keep_obj]
    labels = (catmat * (np.arange(catmat.shape[1]) + 1)).sum(axis=1)  # 1..K
    return keep_obj, labels.astype(int), categories


def _loo_nearest_centroid(X: np.ndarray, labels: np.ndarray, n_cat: int) -> np.ndarray:
    """Vectorized leave-one-out nearest-centroid prediction.

    Removing object i only shifts its own class centroid, so per-class sums are
    reused: centroid'_c(i) = (sum_c - x_i)/(n_c - 1) for c == label(i), else sum_c/n_c.
    NaNs in the word vectors are handled with nan-aware sums (matches MATLAB nanmean).
    """
    n, d = X.shape
    pred = np.empty(n, dtype=int)
    valid = ~np.isnan(X)
    Xf = np.where(valid, X, 0.0)

    # Per-class nan-aware sums and counts (n_cat, d).
    csum = np.zeros((n_cat, d))
    ccnt = np.zeros((n_cat, d))
    for c in range(n_cat):
        m = labels == (c + 1)
        csum[c] = Xf[m].sum(axis=0)
        ccnt[c] = valid[m].sum(axis=0)

    for i in range(n):
        lab = labels[i] - 1
        s = csum.copy()
        cnt = ccnt.copy()
        s[lab] -= Xf[i]
        cnt[lab] -= valid[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            centroids = s / cnt
        # Euclidean distance ignoring dims that are NaN in the test point.
        diff = centroids - X[i][None, :]
        diff = np.where(np.isnan(diff), 0.0, diff)
        dist = np.sqrt((diff * diff).sum(axis=1))
        pred[i] = np.argmin(dist) + 1
    return pred


def predict_category(ds) -> ClassificationResult:
    keep_obj, labels, categories = _reduce(ds)
    n_cat = len(categories)

    X_spose = ds.embedding[keep_obj]
    X_word = ds.sensevec[keep_obj]

    pred_s = _loo_nearest_centroid(X_spose, labels, n_cat)
    pred_w = _loo_nearest_centroid(X_word, labels, n_cat)

    return ClassificationResult(
        accuracy_spose=float(100.0 * (pred_s == labels).mean()),
        accuracy_wordvec=float(100.0 * (pred_w == labels).mean()),
        labels=labels,
        pred_spose=pred_s,
        pred_wordvec=pred_w,
        categories=categories,
    )
