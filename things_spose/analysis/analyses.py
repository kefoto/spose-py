"""Quantitative analyses behind Figures 2, 6, 7 and 8.

Everything here is fully vectorized -- no Python loop over the 1,854 objects or
the 146,012 test triplets on any hot path. Bootstrap / permutation / randomization
tests build a single ``(n_pairs, n_boot)`` index matrix and evaluate all resamples
with one batched, standardized dot-product instead of looping a ``corr`` call.

A note on reproducibility: MATLAB's exact ``rng`` stream cannot be reproduced, so
we reproduce the *procedure* (seed + resample counts). Numeric ties in the
argmax and the random resamples differ only in the last digits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..core import paths
from .similarity import squareformq

CHANCE = 100.0 / 3.0


# --------------------------------------------------------------------------- #
# Batched correlation helpers (the core of every bootstrap/permutation test)
# --------------------------------------------------------------------------- #
def _zscore_cols(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    sd = np.sqrt((a * a).sum(axis=0, keepdims=True))
    return a / sd


def batched_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between matching columns of A and B (both (m, k))."""
    return (_zscore_cols(A) * _zscore_cols(B)).sum(axis=0)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def fisher_ci(r: float, boot: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """95% CI from bootstrap replicates, averaged in Fisher-z space."""
    s = np.std(np.arctanh(boot), ddof=0)
    return float(np.tanh(np.arctanh(r) - z * s)), float(np.tanh(np.arctanh(r) + z * s))


# --------------------------------------------------------------------------- #
# Figure 2a: triplet prediction accuracy
# --------------------------------------------------------------------------- #
@dataclass
class TripletPrediction:
    accuracy: float               # % of trials where the model's top pair == chosen pair
    accuracy_ci95: float          # 95% CI across objects
    per_object_acc: np.ndarray    # (N,) % predicted correct per object
    per_object_prob: np.ndarray   # (N,) mean choice probability per object
    choice: np.ndarray            # (T,) predicted pair index 1..3
    prob: np.ndarray              # (T,) softmax probability of the chosen pair


def predict_triplets(
    dot_product: np.ndarray,
    triplets: np.ndarray,
    n_objects: int = 1854,
    seed: int = 42,
) -> TripletPrediction:
    """Vectorized port of ``make_figures_behavsim.m`` lines 189-211.

    Column 0 of every test triplet is (by construction) the human-chosen pair's
    shared object; the three candidate pair similarities are (0,1), (0,2), (1,2).
    The model predicts the pair with the largest dot product; accuracy is how
    often that is pair 1 (the (0,1) pair == the human choice).
    """
    t = np.asarray(triplets, dtype=np.int64)
    i, j, k = t[:, 0], t[:, 1], t[:, 2]
    sims = np.stack(
        [dot_product[i, j], dot_product[i, k], dot_product[j, k]], axis=1
    )  # (T, 3)

    mx = sims.max(axis=1, keepdims=True)
    is_max = sims == mx
    n_ties = is_max.sum(axis=1)

    rng = np.random.default_rng(seed)
    # Default winner = first max (argmax). Break genuine ties uniformly at random.
    choice_idx = sims.argmax(axis=1)
    tie_rows = np.where(n_ties > 1)[0]
    for r in tie_rows:  # ties are extremely rare; this touches a handful of rows
        opts = np.flatnonzero(is_max[r])
        choice_idx[r] = opts[rng.integers(opts.size)]

    choice = choice_idx + 1  # 1..3 to match MATLAB
    exp_sims = np.exp(sims)
    prob = exp_sims[np.arange(len(t)), choice_idx] / exp_sims.sum(axis=1)

    correct = choice == 1
    accuracy = 100.0 * correct.mean()

    # Per-object aggregation without a Python loop over objects.
    per_object_acc = np.full(n_objects, np.nan)
    per_object_prob = np.full(n_objects, np.nan)
    hit_sum = np.zeros(n_objects)
    hit_cnt = np.zeros(n_objects)
    prob_sum = np.zeros(n_objects)
    for col in (i, j, k):  # 3 vectorized scatter-adds, not T iterations
        np.add.at(hit_sum, col, correct)
        np.add.at(hit_cnt, col, 1.0)
        np.add.at(prob_sum, col, prob)
    seen = hit_cnt > 0
    per_object_acc[seen] = 100.0 * hit_sum[seen] / hit_cnt[seen]
    per_object_prob[seen] = 100.0 * prob_sum[seen] / hit_cnt[seen]

    accuracy_ci95 = 1.96 * np.nanstd(per_object_acc, ddof=0) / np.sqrt(n_objects)

    return TripletPrediction(
        accuracy=float(accuracy),
        accuracy_ci95=float(accuracy_ci95),
        per_object_acc=per_object_acc,
        per_object_prob=per_object_prob,
        choice=choice,
        prob=prob,
    )


# --------------------------------------------------------------------------- #
# Figure 2a: human noise ceiling from repeated triplets
# --------------------------------------------------------------------------- #
@dataclass
class NoiseCeiling:
    ceiling: float          # mean consistency (%)
    ci95: float
    consistency: np.ndarray  # (1000,) per-triplet consistency in [0, 1]


def noise_ceiling(csv_path=None) -> NoiseCeiling:
    """Port of lines 218-254: for each of the 1000 unique repeated triplets, the
    consistency is the fraction of participants giving the most common response.
    """
    if csv_path is None:
        csv_path = paths.data("triplets_noiseceiling.csv")
    dat = np.loadtxt(csv_path)[:, :4]  # cols: 3 object ids + choice (1..3)

    triplet = dat[:, :3].astype(np.int64)
    choice = dat[:, 3].astype(np.int64)

    # Sort each triplet's three ids and remap the choice to the sorted position,
    # so identical triplets collapse to the same key regardless of presentation
    # order (MATLAB lines 231-234).
    order = np.argsort(triplet, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(triplet, order, axis=1)
    # choice is 1-based index into the original triplet; find its new position.
    new_choice = (order == (choice[:, None] - 1)).argmax(axis=1) + 1

    df = pd.DataFrame(
        {"a": sorted_ids[:, 0], "b": sorted_ids[:, 1], "c": sorted_ids[:, 2],
         "choice": new_choice}
    )
    # consistency per unique triplet = max vote share across the 3 choices.
    counts = (
        df.groupby(["a", "b", "c"])["choice"]
        .apply(lambda s: s.value_counts().max() / len(s))
    )
    consistency = counts.to_numpy()
    ceiling = float(consistency.mean() * 100.0)
    ci95 = float(1.96 * consistency.std(ddof=0) * 100.0 / np.sqrt(consistency.size))
    return NoiseCeiling(ceiling=ceiling, ci95=ci95, consistency=consistency)


def percent_performance(
    per_object_acc: np.ndarray,
    consistency: np.ndarray,
    seed: int = 42,
    n_boot: int = 10000,
) -> tuple[float, float]:
    """Percent of human performance achieved above chance, with bootstrap SE
    (lines 374-382)."""
    acc = np.nanmean(per_object_acc)
    pct = 100.0 * (acc - CHANCE) / (100.0 * consistency.mean() - CHANCE)

    rng = np.random.default_rng(seed)
    n_obj = per_object_acc.shape[0]
    n_ceil = consistency.shape[0]
    valid = per_object_acc[~np.isnan(per_object_acc)]
    bi = rng.integers(0, valid.size, size=(1000, n_boot))
    ci = rng.integers(0, n_ceil, size=(1000, n_boot))
    num = valid[bi].mean(axis=0) - CHANCE
    den = 100.0 * consistency[ci].mean(axis=0) - CHANCE
    se = 100.0 * np.std(num / den, ddof=0)
    return float(pct), float(se)


# --------------------------------------------------------------------------- #
# Figure 2b: model vs behavioral similarity for the 48 objects
# --------------------------------------------------------------------------- #
@dataclass
class Sim48Result:
    r: float
    r_ci95: tuple[float, float]
    reliability: float
    splithalf: float
    variance_explained: float


def sim48_correlation(
    spose_sim48: np.ndarray,
    rdm48: np.ndarray,
    rdm48_split1: np.ndarray,
    rdm48_split2: np.ndarray,
    seed: int = 2,
    n_boot: int = 1000,
) -> Sim48Result:
    """Ports lines 320-367. NOTE: the released data + code yield r ~= 0.87; the
    figure's ``legend('R = 0.90')`` is a static annotation, not this value."""
    c1 = squareformq(spose_sim48)
    c2 = squareformq(1.0 - rdm48)
    r = pearson(c1, c2)

    # Paired bootstrap over object-pairs (not objects, which would be biased).
    rng = np.random.default_rng(seed)
    n_pairs = c1.size
    idx = rng.integers(0, n_pairs, size=(n_pairs, n_boot))
    boot = batched_pearson(c1[idx], c2[idx])
    r_ci95 = fisher_ci(r, boot)

    s1 = squareformq(1.0 - rdm48_split1)
    s2 = squareformq(1.0 - rdm48_split2)
    reliability = pearson(s1, s2)
    splithalf = float(
        np.tanh(np.mean(np.arctanh([pearson(s1, c1), pearson(s2, c1)])))
    )
    variance_explained = splithalf**2 / reliability**2
    return Sim48Result(r, r_ci95, reliability, splithalf, variance_explained)


# --------------------------------------------------------------------------- #
# Figure 7: typicality vs SPoSE dimension
# --------------------------------------------------------------------------- #
@dataclass
class TypicalityResult:
    rho: np.ndarray               # (17,) Spearman rho per category
    p: np.ndarray                 # one-sided p-values
    p_adjusted: np.ndarray        # FDR (BH) corrected
    ci95: list[tuple[float, float]]
    order: np.ndarray             # sort by rho descending (for plotting)
    dim_index: np.ndarray         # matched dimension per category (0-based)


def _spearman_p_one_sided(x, y):
    """Spearman rho and right-tailed p (t-approximation, matches MATLAB corr)."""
    n = len(x)
    rho = pearson(rankdata(x), rankdata(y))
    if n <= 2:
        return rho, np.nan
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho**2))
    from scipy.stats import t as tdist

    p = tdist.sf(t, n - 2)  # right-tailed
    return rho, p


def typicality_correlations(ds, seed: int = 1, n_boot: int = 1000) -> TypicalityResult:
    """Ports lines 849-923: one-sided Spearman between per-object typicality and
    the SPoSE weight on the best-matching dimension, with BH-FDR and bootstrap CI."""
    from ..core.external_fdr import fdr_bh

    sub = ds.category27_subind
    rho = np.zeros(len(sub))
    p = np.zeros(len(sub))
    ci = []
    dim_index = ds.best_match27[sub].astype(int)
    rng = np.random.default_rng(seed)

    for i, s in enumerate(sub):
        typ = ds.typicality_normed[s]
        w = ds.embedding[ds.category27_ind[s], int(ds.best_match27[s])]
        rho[i], p[i] = _spearman_p_one_sided(typ, w)
        nc = typ.size
        idx = rng.integers(0, nc, size=(nc, n_boot))
        boot = np.array([spearman(typ[idx[:, b]], w[idx[:, b]]) for b in range(n_boot)])
        lo, hi = fisher_ci(rho[i], boot, z=1.645)  # one-sided 95%
        ci.append((lo, hi))

    _, p_adjusted = fdr_bh(p)
    order = np.argsort(rho)[::-1]
    return TypicalityResult(rho, p, p_adjusted, ci, order, dim_index)


# --------------------------------------------------------------------------- #
# Figure 8: human dimension ratings -> similarity
# --------------------------------------------------------------------------- #
@dataclass
class HumanRatingResult:
    r: float
    r_ci95: tuple[float, float]
    p_randomization: float
    predicted_sim: np.ndarray     # (20, 20)
    true_sim: np.ndarray          # (20, 20)
    object_index: np.ndarray      # (20,) indices into the 1854 objects


def human_rating_similarity(ds, seed: int = 42, n_shuffle: int = 10000,
                            backend_name: str = "auto") -> HumanRatingResult:
    """Ports lines 929-1009: substitute mean human dimension ratings for 20
    objects into the embedding, recompute similarity, and compare to the model."""
    from .similarity import embedding2sim

    object_names20 = [
        "bazooka", "bib", "crowbar", "crumb", "flamingo", "handbrake", "hearse",
        "keyhole", "palm_tree", "scallion", "sleeping_bag", "spider_web",
        "splinter", "staple_gun", "suitcase", "syringe", "tennis_ball", "woman",
        "workbench", "wreck",
    ]
    uid = ds.unique_id
    ind = np.array([uid.index(n) for n in object_names20])

    # Rt: mean across subjects (axis 2), then the scale adjustment (lines 939-945).
    # ratings_translated_all is (objects=20, dims=49, subjects=20).
    Rt = ds.ratings_translated_all.mean(axis=2)  # (20 objects, 49 dims)
    minRt = Rt.min(axis=0)
    mRt = Rt.mean(axis=0)
    Rt = Rt - mRt
    Rt = (1.0 + minRt) * Rt
    Rt = Rt + mRt

    spose_sub = ds.embedding.copy()
    spose_sub[ind, :] = Rt - Rt.min(axis=0)
    cp = embedding2sim(spose_sub, backend_name=backend_name)

    true_sim = ds.spose_sim[np.ix_(ind, ind)]
    predicted_sim = cp[np.ix_(ind, ind)]

    c1 = squareformq(predicted_sim)
    c2 = squareformq(true_sim)
    r = pearson(c1, c2)

    # Randomization test: shuffle the 20 object labels of predicted_sim.
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((20, n_shuffle)), axis=0)
    r_rand = np.empty(n_shuffle)
    for s in range(n_shuffle):
        pr = predicted_sim[np.ix_(perms[:, s], perms[:, s])]
        r_rand[s] = pearson(squareformq(pr), c2)
    p_rand = float((np.concatenate([r_rand, [r]]) >= r).mean())

    # Bootstrap CI over object-pairs.
    rng2 = np.random.default_rng(1)
    npairs = c1.size
    idx = rng2.integers(0, npairs, size=(npairs, 1000))
    boot = batched_pearson(c1[idx], c2[idx])
    r_ci95 = fisher_ci(r, boot)

    return HumanRatingResult(r, r_ci95, p_rand, predicted_sim, true_sim, ind)


# --------------------------------------------------------------------------- #
# Figure 6: dimension ablation (consumes cached reduced similarity matrices)
# --------------------------------------------------------------------------- #
@dataclass
class AblationResult:
    acc_by_dims: np.ndarray       # (50,) accuracy for 0..49 dims retained
    acc_ci95: np.ndarray          # (49,) CI for the reduced points
    var_by_dims: np.ndarray       # (50,) % variance explained for 0..49 dims
    mindim_acc: int
    maxdim_acc: int
    mindim_var: int
    maxdim_var: int


def build_reduced_embeddings(embedding: np.ndarray) -> np.ndarray:
    """The 49 progressively-reduced embeddings for the Fig-6 ablation.

    Ports ``make_figures`` L395-408: sort each object's weights ascending, then
    cumulatively zero the ``d`` smallest-weight dimensions. ``result[d]`` has the
    ``d+1`` smallest dimensions per object zeroed (so ``result[0]`` = 1 dim
    zeroed, ``result[48]`` = all zeroed), matching MATLAB ``spose_embedding49_reduc``.
    Returns an ``(D, N, D)`` array.
    """
    E = np.asarray(embedding, dtype=np.float64)
    n, d = E.shape
    order = np.argsort(E, axis=1, kind="stable")  # ascending: smallest first
    out = np.empty((d, n, d), dtype=E.dtype)
    cur = E.copy()
    rows = np.arange(n)
    for i_dim in range(d):
        cur[rows, order[:, i_dim]] = 0.0
        out[i_dim] = cur
    return out


def dimension_ablation(
    reduced_embeddings: np.ndarray,   # (49, N, 49) reduced embeddings
    reduced_sims: np.ndarray | None,  # (49, N, N) reduced similarity matrices
    spose_sim: np.ndarray,
    triplets: np.ndarray,
    full_accuracy: float,
    seed: int = 42,
    reduced_sim_pairvecs: np.ndarray | None = None,  # (49, n_pairs) pre-squareformed
) -> AblationResult:
    """Ports lines 421-472. ``reduced_*[d]`` is the embedding/similarity with the
    ``d`` smallest-weight dimensions per object zeroed out (d = 0..48).

    The variance-explained curve only needs the below-diagonal pair-vector of each
    reduced similarity matrix, so ``build_cache.py`` may cache those directly and
    pass them as ``reduced_sim_pairvecs`` instead of the full ``(49, N, N)`` stack.
    Exactly one of ``reduced_sims`` / ``reduced_sim_pairvecs`` must be provided.
    """
    n_obj = spose_sim.shape[0]
    n_reduc = reduced_embeddings.shape[0]

    acc = np.zeros(n_reduc)
    acc_ci = np.zeros(n_reduc)
    for d in range(n_reduc):
        dp = reduced_embeddings[d] @ reduced_embeddings[d].T
        pred = predict_triplets(dp, triplets, n_objects=n_obj, seed=seed)
        acc[d] = pred.accuracy
        acc_ci[d] = 1.96 * np.nanstd(pred.per_object_acc, ddof=0) / np.sqrt(n_obj)

    # Reverse so index = number of dimensions retained (1..49).
    acc = acc[::-1]
    acc_ci = acc_ci[::-1]

    cutoff95 = 0.95 * (full_accuracy - CHANCE) + CHANCE
    cutoff99 = 0.99 * (full_accuracy - CHANCE) + CHANCE
    mindim_acc = int(np.argmax(acc > cutoff95))                 # first exceeding - "0-based count"
    maxdim_acc = int(len(acc) - 1 - np.argmax((acc < cutoff99)[::-1]))

    full_vec = squareformq(spose_sim)
    if reduced_sim_pairvecs is not None:
        r_reduc = np.array(
            [pearson(full_vec, reduced_sim_pairvecs[d]) for d in range(n_reduc)])
    elif reduced_sims is not None:
        r_reduc = np.array(
            [pearson(full_vec, squareformq(reduced_sims[d])) for d in range(n_reduc)])
    else:
        raise ValueError("provide either reduced_sims or reduced_sim_pairvecs")
    r_reduc = r_reduc[::-1]
    var = r_reduc**2
    mindim_var = int(np.argmax(var > 0.95))
    maxdim_var = int(len(var) - 1 - np.argmax((var < 0.99)[::-1]))

    acc_by_dims = np.append(acc, full_accuracy)
    var_by_dims = np.append(100.0 * var, 100.0)
    return AblationResult(
        acc_by_dims, acc_ci, var_by_dims,
        mindim_acc, maxdim_acc, mindim_var, maxdim_var,
    )
