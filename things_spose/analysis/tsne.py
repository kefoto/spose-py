"""2-D layout for the "crystal" maps of Figures 4 and 5.

Ports ``make_figures_behavsim.m`` lines 697-712, which build the map in three
steps:

1. A metric-MDS 2-D solution from the dissimilarity matrix (MATLAB ``mdscale``
   with the ``metricstress`` criterion) -> :func:`metric_mds`. We use
   ``sklearn.manifold.MDS`` (metric SMACOF); the exact coordinates differ from
   MATLAB (different optimizer + RNG stream) but the global arrangement matches.
2. A *multiscale* affinity matrix ``P = 0.5*(d2p(D, 5) + d2p(D, 30))`` combining
   two perplexities -> :func:`d2p`, a faithful port of van der Maaten's routine.
3. Symmetric t-SNE (:func:`tsne_p`) initialised **from the MDS solution**. When
   an initial solution is supplied the optimiser skips the random init and the
   early-exaggeration ("lying") phase, exactly as the bundled ``tsne_p.m`` does.

The published figure used the same routines; only the RNG differs, so the layout
reproduces qualitatively, not coordinate-for-coordinate. :func:`crystal_layout`
runs the whole pipeline and is what :mod:`things_spose.analysis.viz` / the cache builder
call.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Step 1: metric MDS init (replaces MATLAB mdscale 'metricstress')
# --------------------------------------------------------------------------- #
def metric_mds(dissim: np.ndarray, n_dims: int = 2, seed: int = 42) -> np.ndarray:
    """2-D metric-MDS embedding of a precomputed dissimilarity matrix."""
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=n_dims,
        metric=True,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        max_iter=300,
        normalized_stress=False,
    )
    return mds.fit_transform(np.asarray(dissim, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Step 2: distance -> affinity (perplexity calibration), port of d2p.m
# --------------------------------------------------------------------------- #
def _hbeta(d_row: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    p = np.exp(-d_row * beta)
    sum_p = p.sum()
    if sum_p == 0.0:
        sum_p = 1e-12
    h = np.log(sum_p) + beta * np.sum(d_row * p) / sum_p
    return h, p / sum_p


def d2p(D: np.ndarray, perplexity: float = 15.0, tol: float = 1e-4) -> np.ndarray:
    """Gaussian affinity matrix calibrated to a target ``perplexity``.

    Faithful port of ``d2p.m``: per point, binary-search the precision
    ``beta = 1/(2 sigma^2)`` so the row's Shannon entropy matches
    ``log(perplexity)`` within ``tol`` (max 50 iterations). ``D`` is passed
    through unchanged (the caller normalises it), matching the main script.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    log_u = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        betamin, betamax = -np.inf, np.inf
        idx = np.r_[0:i, i + 1:n]  # all columns except the diagonal
        d_row = D[i, idx]

        h, this_p = _hbeta(d_row, beta)
        h_diff = h - log_u
        tries = 0
        while abs(h_diff) > tol and tries < 50:
            if h_diff > 0:
                betamin = beta
                beta = beta * 2.0 if np.isinf(betamax) else (beta + betamax) / 2.0
            else:
                betamax = beta
                beta = beta / 2.0 if np.isinf(betamin) else (beta + betamin) / 2.0
            h, this_p = _hbeta(d_row, beta)
            h_diff = h - log_u
            tries += 1

        P[i, idx] = this_p
    return P


# --------------------------------------------------------------------------- #
# Step 3: symmetric t-SNE optimiser, port of tsne_p.m
# --------------------------------------------------------------------------- #
def tsne_p(
    P: np.ndarray,
    init: np.ndarray | None = None,
    n_dims: int = 2,
    max_iter: int = 1000,
    seed: int = 1,
) -> np.ndarray:
    """Symmetric t-SNE on a joint-probability affinity matrix ``P``.

    Mirrors ``tsne_p.m`` (lr 500, momentum 0.5->0.8 @ iter 250, delta-bar-delta
    gains, ``min_gain`` 0.01). When ``init`` is given (an ``(n, n_dims)`` array,
    e.g. the MDS solution) the random init and the x4 early-exaggeration are both
    skipped, exactly as the MATLAB ``initial_solution`` branch does.
    """
    P = np.asarray(P, dtype=np.float64).copy()
    n = P.shape[0]
    momentum, final_momentum = 0.5, 0.8
    mom_switch_iter, stop_lying_iter = 250, 100
    epsilon, min_gain = 500.0, 0.01
    tiny = np.finfo(np.float64).tiny

    initial_solution = init is not None
    if initial_solution:
        ydata = np.asarray(init, dtype=np.float64).copy()
        n_dims = ydata.shape[1]

    np.fill_diagonal(P, 0.0)
    P = 0.5 * (P + P.T)
    P = np.maximum(P / P.sum(), tiny)
    const = np.sum(P * np.log(P))
    if not initial_solution:
        P = P * 4.0
        rng = np.random.default_rng(seed)
        ydata = 1e-4 * rng.standard_normal((n, n_dims))

    y_incs = np.zeros_like(ydata)
    gains = np.ones_like(ydata)

    for it in range(1, max_iter + 1):
        sum_ydata = np.sum(ydata ** 2, axis=1)
        num = 1.0 / (1.0 + (sum_ydata[:, None] + sum_ydata[None, :]
                            - 2.0 * (ydata @ ydata.T)))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), tiny)

        L = (P - Q) * num
        y_grads = 4.0 * (np.diag(L.sum(axis=0)) - L) @ ydata

        same = np.sign(y_grads) == np.sign(y_incs)
        gains = (gains + 0.2) * (~same) + (gains * 0.8) * same
        gains[gains < min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        ydata = ydata + y_incs
        ydata = ydata - ydata.mean(axis=0, keepdims=True)

        if it == mom_switch_iter:
            momentum = final_momentum
        if it == stop_lying_iter and not initial_solution:
            P = P / 4.0

    return ydata


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def crystal_layout(
    dissim: np.ndarray,
    perplexity1: float = 5.0,
    perplexity2: float = 30.0,
    mds_seed: int = 42,
    max_iter: int = 1000,
) -> np.ndarray:
    """The Figure-4/5 2-D layout: MDS init -> multiscale affinity -> t-SNE.

    ``dissim`` is the ``1 - spose_sim`` dissimilarity matrix.
    """
    dissim = np.asarray(dissim, dtype=np.float64)
    y2 = metric_mds(dissim, n_dims=2, seed=mds_seed)
    D = dissim / dissim.max()
    P = 0.5 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5))
    return tsne_p(P, init=y2, max_iter=max_iter)
