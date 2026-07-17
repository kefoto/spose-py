"""Category-stratified subset selection over the 1854-object SPoSE embedding.

Within each of the 27 manual THINGS categories, objects are projected to 2D via
MDS (preserving pairwise dissimilarity), K-means finds cluster centroids in
that 2D space, and each centroid is snapped to its nearest real object. This
picks a subset that is representative of the full embedding's category and
dimension structure, rather than a uniform random sample.

Validity metrics and how each connects to "representation"
------------------------------------------------------------------------------
`percent`/subset size is only the sweep variable used by
`scripts/select_subsets_by_percent.py` -- the metrics below are the actual
selection criterion, not percent itself.

* `dimension_coverage` (mean/median/min across the 49 SPoSE dims) -- ratio of
  the subset's value-range to the full-set's value-range, per dimension. Each
  dimension is a distinct semantic property (e.g. "round", "made of metal",
  "food-related"); if a dimension's range collapses in the subset, objects
  expressing the extremes of that property are gone and the embedding's
  dimensional structure can no longer be represented. `min` is the worst-case
  (bottleneck) dimension; PRIMARY sizing criterion, monotonic and
  non-trivial.

* `rdm_fidelity` -- Spearman correlation between the shipped, triplet-derived
  dissimilarity matrix (`Dataset.dissim`, the real behavioral/model
  judgments) restricted to subset pairs, and the cosine RDM recomputed from
  the subset's embedding rows alone. Answers "if you only had the subset,
  would pairwise distances computed from it still agree with the real
  similarity judgments for those objects?", using every pair inside the
  subset (O(n^2)), not just complete triplets. PRIMARY sizing criterion --
  the metric closest to whether perceptual/behavioral similarity structure
  survives. Contrast with `rsa_check`, which looks similar but is
  mathematically identical either way (cosine similarity depends only on the
  two vectors involved) and is therefore ~1.0 by construction: an indexing
  sanity check, not a representativeness check.

* `category_centroid_shift` / `weighted_mean_centroid_shift` -- per-category
  (then size-weighted) distance, in the original 49D space, between a
  category's full-set centroid and its subset centroid. Coverage and RDM
  fidelity are global; this is a per-category check -- a category (e.g.
  "animals") can lose its typical objects and drift even while global metrics
  look fine.

* `image_diversity` -- `diversity_ratio` is mean pairwise distance among the
  subset's images (in pixel-PCA space) divided by the full set's; ~1 means
  the subset is about as visually spread out per-object as the full set,
  well below 1 flags visual redundancy. `min_nn_dist`/`median_nn_dist` are
  nearest-neighbor distances within the subset's images -- `min_nn_dist` near
  0 means two selected objects are near-duplicate photos. Coverage alone only
  checks per-axis range and would miss redundant/near-duplicate images
  clustered inside that range; this is the complementary density/diversity
  check on the visual side.

* `image_pca_coverage` -- same coverage check as `dimension_coverage`, but in
  a pixel-PCA space fit on the raw thumbnails, so it tracks visual/image
  representativeness rather than SPoSE structure.

* `evaluate_on_subset` (triplet accuracy) -- odd-one-out prediction accuracy
  on triplets fully contained in the subset. Kept for reference only: stays
  flat (~0.61-0.65) at every size because MDS+K-means always picks
  well-spread points, so it cannot distinguish a good subset from a great
  one and is NOT used for sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

RNG_SEED = 0


def build_category_labels(category_mat_manual: np.ndarray) -> np.ndarray:
    """Collapse the (n_objects, 27) manual category matrix to one label per object.

    Objects in exactly one of the 27 categories get ``category_01``..``category_27``;
    objects in zero or >1 categories are labeled "uncategorized" / "multiple".
    """
    counts = category_mat_manual.sum(axis=1)
    labels = np.full(category_mat_manual.shape[0], "uncategorized", dtype=object)
    single = counts == 1
    cat_idx = category_mat_manual[single].argmax(axis=1)
    labels[single] = [f"category_{c + 1:02d}" for c in cat_idx]
    labels[counts > 1] = "multiple"
    return labels


def select_subset(X: np.ndarray, categories, n_total: int, min_per_category: int = 1) -> np.ndarray:
    """Select ``n_total`` object indices, stratified across ``categories``.

    X: (n_objects, n_dims) embedding
    categories: (n_objects,) array-like of category labels
    Returns: sorted array of selected row indices, length <= n_total
    """
    cats = pd.Series(categories)
    cat_counts = cats.value_counts()

    # proportional allocation, but floor every category at min_per_category
    # so rare/atypical categories aren't wiped out by big ones (e.g. animals)
    raw_alloc = (cat_counts / cat_counts.sum() * n_total).round().astype(int)
    alloc = raw_alloc.clip(lower=min_per_category)

    # fix rounding drift so allocations sum exactly to n_total
    diff = n_total - alloc.sum()
    order = alloc.sort_values(ascending=(diff < 0)).index
    i = 0
    while diff != 0 and len(order):
        c = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if alloc[c] + step >= min_per_category:
            alloc[c] += step
            diff -= step
        i += 1

    selected = []
    for cat, k in alloc.items():
        mask = (cats == cat).values
        idx_in_cat = np.where(mask)[0]
        Xc = X[idx_in_cat]

        k = min(k, len(idx_in_cat))  # can't select more than exist
        if k <= 0:
            continue

        if len(idx_in_cat) <= k:
            selected.extend(idx_in_cat.tolist())
            continue

        chosen_local = _mds_knn_centroids(Xc, k)
        selected.extend(idx_in_cat[chosen_local].tolist())

    return np.array(sorted(set(selected)))[:n_total]


def _mds_knn_centroids(Xc: np.ndarray, k: int) -> np.ndarray:
    """Project Xc to 2D via MDS, K-means-cluster into k clusters, snap each
    centroid to its nearest real item (1-NN in 2D). Returns local indices
    (into Xc), length k (topped up by farthest-point fill on collisions).
    """
    mds = MDS(
        n_components=2,
        dissimilarity="euclidean",
        random_state=RNG_SEED,
        n_init=4,
        normalized_stress="auto",
    )
    coords_2d = mds.fit_transform(Xc)

    km = KMeans(n_clusters=k, n_init="auto", random_state=RNG_SEED).fit(coords_2d)
    nn = NearestNeighbors(n_neighbors=1).fit(coords_2d)
    _, nn_idx = nn.kneighbors(km.cluster_centers_)
    chosen_local = np.unique(nn_idx.flatten())

    if len(chosen_local) < k:
        chosen_local = _farthest_point_fill(coords_2d, chosen_local, k)

    return chosen_local


def _farthest_point_fill(Xc: np.ndarray, chosen_local: np.ndarray, k: int) -> np.ndarray:
    """Greedily add the point farthest (min-distance) from the current set."""
    chosen = list(chosen_local)
    remaining = [i for i in range(len(Xc)) if i not in chosen]
    while len(chosen) < k and remaining:
        d = np.linalg.norm(
            Xc[remaining][:, None, :] - Xc[chosen][None, :, :], axis=-1
        ).min(axis=1)
        pick = remaining[int(np.argmax(d))]
        chosen.append(pick)
        remaining.remove(pick)
    return np.array(chosen)


def dimension_coverage(X: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    """Per-dimension ratio of the subset's value range to the full set's range."""
    full_range = X.max(0) - X.min(0)
    sub_range = X[selected].max(0) - X[selected].min(0)
    coverage = np.divide(sub_range, full_range, out=np.zeros_like(full_range),
                          where=full_range > 0)
    return pd.DataFrame({
        "dim": np.arange(X.shape[1]),
        "full_range": full_range,
        "subset_range": sub_range,
        "coverage_ratio": coverage,
    }).sort_values("coverage_ratio")


def rdm(X: np.ndarray) -> np.ndarray:
    """Pairwise dissimilarity (1 - cosine similarity) matrix."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    return 1 - sim


def rsa_check(X: np.ndarray, selected: np.ndarray) -> float:
    """Spearman correlation between the full RDM's subset submatrix and the RDM
    computed directly on the subset; should be ~1.0 by construction (sanity
    check on indexing only -- cosine similarity depends solely on the two
    vectors involved, so this is identical math either way and cannot detect
    representativeness loss). See :func:`rdm_fidelity` for the real check.
    """
    full_rdm_sub = rdm(X)[np.ix_(selected, selected)]
    direct_rdm = rdm(X[selected])
    iu = np.triu_indices(len(selected), k=1)
    rho, _ = spearmanr(full_rdm_sub[iu], direct_rdm[iu])
    return rho


def rdm_fidelity(dissim_full: np.ndarray, X: np.ndarray, selected: np.ndarray) -> float:
    """Spearman correlation, over every pair within `selected` (not just triplets
    fully contained in it), between the shipped model's dissimilarity matrix
    (``Dataset.dissim``, derived from the triplet-softmax similarity, i.e. the
    actual behavioral/model judgments) and the cosine-based RDM recomputed
    directly from the subset's embedding rows alone.

    Unlike :func:`rsa_check`, these two matrices are NOT the same computation
    -- `dissim_full` comes from `embedding2sim`'s softmax formula over the full
    1854-object embedding, while the comparison RDM only ever sees the subset.
    A high correlation means: if you only had the subset, distances derived
    from it would still agree with the true (full-model) pairwise judgments
    for those same objects -- this is the real "did we lose representational
    structure" check, and it uses O(n^2) pairs instead of the handful of
    complete triplets `evaluate_on_subset` is limited to.
    """
    sub_dissim_full = dissim_full[np.ix_(selected, selected)]
    sub_dissim_direct = rdm(X[selected])
    iu = np.triu_indices(len(selected), k=1)
    rho, _ = spearmanr(sub_dissim_full[iu], sub_dissim_direct[iu])
    return rho


def spose_predict_odd_one_out(x_i: np.ndarray, x_j: np.ndarray, x_k: np.ndarray) -> str:
    """SPoSE odd-one-out decision rule (Hebart et al., 2020, eq. 1-2): the
    predicted odd-one-out is the item NOT in the pair with the highest
    dot-product similarity among the three pairs.
    """
    sims = {
        "k": x_i @ x_j,  # pair (i,j) similar -> odd one out is k
        "j": x_i @ x_k,  # pair (i,k) similar -> odd one out is j
        "i": x_j @ x_k,  # pair (j,k) similar -> odd one out is i
    }
    return max(sims, key=sims.get)


def category_centroid_shift(X: np.ndarray, categories: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    """Per-category distance between the full-set centroid and the subset centroid
    (both in the original embedding space), plus a size-weighted overall summary.

    Categories with zero subset members are dropped (no subset centroid to compare).
    Unlike :func:`rsa_check` (an indexing sanity check that is ~1.0 by
    construction), this measures whether the subset actually preserves each
    category's position in embedding space.
    """
    cats = np.asarray(categories)
    sel_set = set(selected.tolist())
    rows = []
    for cat in np.unique(cats):
        full_idx = np.where(cats == cat)[0]
        sub_idx = np.array([i for i in full_idx if i in sel_set])
        if len(sub_idx) == 0:
            continue
        centroid_full = X[full_idx].mean(axis=0)
        centroid_sub = X[sub_idx].mean(axis=0)
        rows.append({
            "category": cat,
            "n_full": len(full_idx),
            "n_subset": len(sub_idx),
            "centroid_dist": np.linalg.norm(centroid_full - centroid_sub),
        })
    return pd.DataFrame(rows)


def weighted_mean_centroid_shift(centroid_df: pd.DataFrame) -> float:
    """Category-size-weighted mean of `category_centroid_shift`'s `centroid_dist`."""
    if len(centroid_df) == 0:
        return float("nan")
    return float(np.average(centroid_df["centroid_dist"], weights=centroid_df["n_full"]))


def image_pca_coverage(images: np.ndarray, selected: np.ndarray, n_components: int = 20) -> pd.DataFrame:
    """Dimension coverage (see :func:`dimension_coverage`) in pixel-PCA space.

    Flattens each thumbnail, fits PCA on the full image set, and checks how much
    of each principal component's full-set value range the subset still spans --
    an embedding-independent check that the subset also represents visual
    appearance, not just SPoSE structure.
    """
    from sklearn.decomposition import PCA

    flat = images.astype(np.float32).reshape(len(images), -1) if images.dtype != object else np.stack(
        [im.astype(np.float32).ravel() for im in images]
    )
    pca = PCA(n_components=n_components, random_state=RNG_SEED)
    coords = pca.fit_transform(flat)
    return dimension_coverage(coords, selected)


def image_diversity(image_coords: np.ndarray, selected: np.ndarray) -> dict:
    """Visual-diversity check, complementary to `dimension_coverage` (which only
    checks per-axis range and would not notice e.g. many near-duplicate images
    clustered inside that range).

    Computed in the same pixel-PCA coordinate space as `image_pca_coverage`:
    * `diversity_ratio`: mean pairwise distance among the subset's images,
      divided by the mean pairwise distance among all 1854 images. ~1 means
      the subset is about as visually spread out per-object as the full set;
      well below 1 flags a subset of visually redundant/similar images.
    * `median_nn_dist` / `min_nn_dist`: median / smallest nearest-neighbor
      distance within the subset. A `min_nn_dist` near 0 flags that at least
      two selected objects are near-duplicate images.
    """
    from scipy.spatial.distance import pdist, squareform

    full_pd = pdist(image_coords)
    sub_coords = image_coords[selected]
    sub_pd = pdist(sub_coords)

    sub_sq = squareform(sub_pd)
    np.fill_diagonal(sub_sq, np.inf)
    nn_dist = sub_sq.min(axis=1)

    return {
        "mean_pairwise_dist_subset": float(sub_pd.mean()),
        "mean_pairwise_dist_full": float(full_pd.mean()),
        "diversity_ratio": float(sub_pd.mean() / full_pd.mean()),
        "median_nn_dist": float(np.median(nn_dist)),
        "min_nn_dist": float(nn_dist.min()),
    }


def evaluate_on_subset(X: np.ndarray, triplets: np.ndarray, selected: np.ndarray) -> dict:
    """Triplet-prediction accuracy, restricted to triplets fully inside `selected`.

    triplets: (n_triplets, 3) 0-based [i, j, k], where k is the odd-one-out
    (matches Dataset.triplets_test's column convention).
    """
    sel_set = set(selected.tolist())
    mask = np.isin(triplets, list(sel_set)).all(axis=1)
    sub_triplets = triplets[mask]

    if len(sub_triplets) == 0:
        return {"n_triplets": 0, "accuracy": None}

    correct = 0
    for i, j, k in sub_triplets:
        i, j, k = int(i), int(j), int(k)
        pred_label = spose_predict_odd_one_out(X[i], X[j], X[k])
        pred_idx = {"i": i, "j": j, "k": k}[pred_label]
        correct += int(pred_idx == k)

    return {
        "n_triplets": len(sub_triplets),
        "accuracy": correct / len(sub_triplets),
    }
