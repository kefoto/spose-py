"""Matplotlib plotting layer for every figure in ``make_figures_behavsim.m``.

One function per figure (``fig1_*`` ... ``extfig2_*``). Each takes the loaded
:class:`~things_spose.data.dataio.Dataset` (plus, where the computation is expensive,
a precomputed analysis result or the cached t-SNE / ablation artifacts), returns
a Matplotlib ``Figure``, and — when ``save_path`` is given — writes an SVG (the
``dosave`` flag of the original script).

The module imports ``matplotlib.pyplot`` but does **not** force a backend: scripts
that run headless on the cluster set ``matplotlib.use("Agg")`` themselves; a
notebook keeps its inline backend. Heavy analyses are imported lazily so that a
numbers-only run never pulls in this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from . import analyses, classify, clustering, dimlabels, reproducibility, similarity

CHANCE = 100.0 / 3.0
_VIRIDIS = plt.cm.viridis


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _finish(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def crystal_vertices(weights: np.ndarray, scaling: float, center=(0.0, 0.0),
                     radius_offset: float = 0.0):
    """Triangle vertices of one object's 49-wedge "crystal" glyph.

    Ports the ``pol2cart`` rose-plot construction (``make_figures`` L738-745):
    wedge ``i`` spans angles ``[a_i, a_{i+1}]`` at radius ``scaling * weights[i]``
    from ``center``. Returns an ``(D, 3, 2)`` array of triangle vertices plus the
    per-wedge radius (for label gating in Fig 5).
    """
    w = np.asarray(weights, dtype=float)
    d = w.size
    angles = np.linspace(0.0, 2.0 * np.pi, d + 1)
    r = scaling * w + radius_offset
    cx, cy = center
    bx = r * np.cos(angles[:-1]) + cx
    by = r * np.sin(angles[:-1]) + cy
    cx2 = r * np.cos(angles[1:]) + cx
    cy2 = r * np.sin(angles[1:]) + cy
    tris = np.empty((d, 3, 2))
    tris[:, 0, :] = center
    tris[:, 1, 0], tris[:, 1, 1] = bx, by
    tris[:, 2, 0], tris[:, 2, 1] = cx2, cy2
    return tris, r


def _wordcloud_image(dw: dimlabels.DimensionWords, color, width=400, height=400):
    """Render a dimension's word cloud to an RGBA image array (word colour = dim)."""
    from wordcloud import WordCloud

    rgb = tuple(int(255 * c) for c in color[:3])

    def color_func(*args, **kwargs):
        return rgb

    freqs = dw.as_frequencies()
    if not freqs:
        return np.zeros((height, width, 4), dtype=np.uint8)
    wc = WordCloud(
        width=width, height=height, background_color=None, mode="RGBA",
        color_func=color_func, prefer_horizontal=0.9, max_words=60,
    ).generate_from_frequencies(freqs)
    return wc.to_array()


# --------------------------------------------------------------------------- #
# Figure 1: embedding / similarity overview
# --------------------------------------------------------------------------- #
def fig1_similarity_overview(ds, save_path=None, n_montage_rows=30, n_montage_cols=61,
                             thumb_stride=3):
    """Sorted similarity heatmap (via ``clustering_algorithm``) + object montage."""
    ind = clustering.clustering_algorithm(3, 5, ds.embedding)
    sub = ind[::10]
    sim = ds.spose_sim[np.ix_(sub, sub)]

    # Column-major thumbnail montage (make_figures L150-164). Thumbnails are
    # downsampled with ``thumb_stride`` so the preview array stays small enough
    # for memory-constrained cluster nodes (full-res would be ~1 GB).
    tile = ds.images[0][::thumb_stride, ::thumb_stride]
    thumb_h, thumb_w = tile.shape[:2]
    montage = np.ones((n_montage_rows * thumb_h, n_montage_cols * thumb_w, 3))
    cnt = 0
    for c in range(n_montage_cols):
        for r in range(n_montage_rows):
            if cnt >= ds.images.shape[0]:
                break
            img = ds.images[cnt][::thumb_stride, ::thumb_stride].astype(float)
            img = img / 255.0 if img.max() > 1.0 else img
            montage[r * thumb_h:(r + 1) * thumb_h,
                    c * thumb_w:(c + 1) * thumb_w, :] = img[..., :3]
            cnt += 1

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    ax0.imshow(montage)
    ax0.set_title("Object thumbnails")
    ax0.axis("off")
    im = ax1.imshow(sim, vmin=0, vmax=0.9, cmap=_VIRIDIS)
    ax1.set_title("Similarity matrix (sorted by dominant dimensions)")
    ax1.axis("off")
    fig.colorbar(im, ax=ax1, fraction=0.046)
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 2: predict behaviour and similarity
# --------------------------------------------------------------------------- #
def fig2_prediction(ds, pred=None, ceiling=None, sim48=None, save_path=None):
    """Bar-vs-noise-ceiling (2a) + predicted/measured 48x48 matrices & scatter (2b)."""
    if pred is None:
        pred = analyses.predict_triplets(ds.dot_product, ds.triplets_test, ds.n_objects)
    if ceiling is None:
        ceiling = analyses.noise_ceiling()
    spose_sim48 = similarity.embedding2sim48(ds.embedding, ds.wordposition48)
    measured48 = 1.0 - ds.rdm48
    if sim48 is None:
        sim48 = analyses.sim48_correlation(
            spose_sim48, ds.rdm48, ds.rdm48_split1, ds.rdm48_split2)

    c1 = similarity.squareformq(spose_sim48)
    c2 = similarity.squareformq(measured48)

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.7, 1, 1, 1])

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar([0], [pred.accuracy], width=0.6, color="k")
    ax_bar.errorbar([0], [pred.accuracy], yerr=[pred.accuracy_ci95], color="k", capsize=4)
    ax_bar.axhspan(ceiling.ceiling - ceiling.ci95, ceiling.ceiling + ceiling.ci95,
                   color="0.7", alpha=0.6)
    ax_bar.axhline(CHANCE, color="r", lw=2)
    ax_bar.set_xticks([])
    ax_bar.set_ylim(30, 75)
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_title("Prediction\nvs noise ceiling")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(spose_sim48, vmin=0, vmax=1, cmap=_VIRIDIS)
    ax1.set_title("predicted similarity")
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(measured48, vmin=0, vmax=1, cmap=_VIRIDIS)
    ax2.set_title("measured similarity")
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(c1, c2, "o", color="0.5", markersize=3)
    ax3.plot([0, 1], [0, 1], "k")
    ax3.set_xlabel("predicted similarity")
    ax3.set_ylabel("measured similarity")
    ax3.set_aspect("equal")
    ax3.set_title(f"r = {sim48.r:.2f}")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 3: dimension examples (word cloud + top thumbnails)
# --------------------------------------------------------------------------- #
_FIG3_DIMS = np.array([3, 11, 12, 15, 17, 40]) - 1  # 1-based -> 0-based
_FIG3_N_IM = 8


def fig3_dimension_examples(ds, dims=None, n_im=_FIG3_N_IM, save_path=None):
    """For chosen dimensions: a word cloud + the top-``n_im`` example objects."""
    if dims is None:
        dims = _FIG3_DIMS
    dims = np.asarray(dims)
    dimsort = np.argsort(-ds.embedding, axis=0)  # descending rank per dimension

    fig, axes = plt.subplots(len(dims), n_im + 1,
                             figsize=(1.6 * (n_im + 1), 1.6 * len(dims)))
    axes = np.atleast_2d(axes)
    for row, d in enumerate(dims):
        dw = dimlabels.dimension_words(ds.dimlabel_answers, int(d))
        axes[row, 0].imshow(_wordcloud_image(dw, ds.colors[d]))
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(ds.labels_short[d])
        for k in range(n_im):
            obj = dimsort[k, d]
            axes[row, k + 1].imshow(ds.images[obj])
            axes[row, k + 1].axis("off")
    fig.suptitle("Dimension examples")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 4: crystal map over the t-SNE layout
# --------------------------------------------------------------------------- #
def fig4_crystal_map(ds, layout, scaling=2.8, alpha=0.85, save_path=None):
    """All 1,854 crystal glyphs positioned at their t-SNE coordinates."""
    layout = np.asarray(layout, dtype=float)
    polys, facecolors = [], []
    for ii in range(ds.n_objects):
        tris, _ = crystal_vertices(ds.embedding[ii], scaling, center=layout[ii])
        for i in range(ds.n_dims):
            polys.append(tris[i])
            facecolors.append(ds.colors[i])

    fig, ax = plt.subplots(figsize=(11, 11))
    coll = PolyCollection(polys, facecolors=facecolors, edgecolors="none", alpha=alpha)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Crystal map (t-SNE layout)")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 5: single-object crystal close-ups
# --------------------------------------------------------------------------- #
_FIG5_OBJECTS = np.array(
    [2, 171, 351, 601, 745, 898, 923, 1000, 1062, 1131, 1166, 1198, 1259, 1284,
     1321, 1529, 1577, 1787]) - 1  # 1-based -> 0-based


def fig5_crystal_closeup(ds, obj_index, scaling=2.8, label_min_radius=1.5,
                         save_path=None):
    """One large labelled crystal glyph for a single object (0-based index)."""
    w = ds.embedding[obj_index]
    tris, radius = crystal_vertices(w, scaling)

    fig, ax = plt.subplots(figsize=(9, 9))
    coll = PolyCollection(list(tris), facecolors=ds.colors, edgecolors="none", alpha=0.5)
    ax.add_collection(coll)

    angles = np.linspace(0.0, 2.0 * np.pi, ds.n_dims + 1)
    mid = 0.5 * (angles[:-1] + angles[1:])
    for i in range(ds.n_dims):
        if radius[i] < label_min_radius:
            continue
        lx = (radius[i] - 0.05) * np.cos(mid[i])
        ly = (radius[i] - 0.05) * np.sin(mid[i])
        rot = np.degrees(mid[i])
        ha = "right"
        if 90 < rot < 270:
            rot += 180
            ha = "left"
        ax.text(lx, ly, ds.labels_short[i], rotation=rot, fontsize=10,
                ha=ha, va="center", rotation_mode="anchor")

    lim = 1.1 * scaling * max(w.max(), 1e-6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(ds.words[obj_index])
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 6: dimension ablation curves
# --------------------------------------------------------------------------- #
def fig6_ablation(ablation, save_path=None):
    """Accuracy-vs-#dims and variance-explained-vs-#dims (needs an AblationResult)."""
    dims = np.arange(ablation.acc_by_dims.size)  # 0..49
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.axhspan(
        0.95 * (ablation.acc_by_dims[-1] - CHANCE) + CHANCE,
        0.99 * (ablation.acc_by_dims[-1] - CHANCE) + CHANCE,
        color="0.7", alpha=0.3)
    ax0.axvspan(ablation.mindim_acc, ablation.maxdim_acc,
                color=(138 / 255, 204 / 255, 101 / 255), alpha=0.5)
    ax0.plot(dims, ablation.acc_by_dims, "k", lw=3)
    ax0.axhline(CHANCE, color="k", ls="--")
    ax0.set_xlim(0, 49)
    ax0.set_ylim(0, 70)
    ax0.set_xlabel("Number of dimensions retained")
    ax0.set_ylabel("Accuracy (%)")

    ax1.axhspan(95, 99, color="0.7", alpha=0.3)
    ax1.axvspan(ablation.mindim_var, ablation.maxdim_var,
                color=(138 / 255, 204 / 255, 101 / 255), alpha=0.5)
    ax1.plot(dims, ablation.var_by_dims, "k", lw=3)
    ax1.set_xlim(0, 49)
    ax1.set_ylim(0, 108.2)
    ax1.set_xlabel("Number of dimensions retained")
    ax1.set_ylabel("Variance explained (%)")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 7: typicality scatter grid
# --------------------------------------------------------------------------- #
def fig7_typicality(ds, typicality=None, save_path=None):
    """3x6 scatter grid: typicality vs best-dimension weight per category."""
    if typicality is None:
        typicality = analyses.typicality_correlations(ds)

    sub = ds.category27_subind
    order = typicality.order
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    axes = axes.ravel()
    for panel, oi in enumerate(order):
        ax = axes[panel]
        s = sub[oi]
        w = ds.embedding[ds.category27_ind[s], int(ds.best_match27[s])]
        typ = ds.typicality_normed[s]
        ax.plot(w, typ, "o", color=ds.colors[int(ds.best_match27[s])], markersize=7)
        rho = typicality.rho[oi]
        bold = typicality.p_adjusted[oi] < 0.05
        ax.text(0.95, 0.05, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes,
                ha="right", va="bottom",
                fontweight="bold" if bold else "normal")
        ax.set_xlabel(ds.labels_short[int(ds.best_match27[s])], fontsize=9)
        ax.set_ylabel(ds.categories27[s], fontsize=9)
        _no_ticks(ax)
    for extra in range(len(order), len(axes)):
        axes[extra].axis("off")
    fig.suptitle("Typicality vs SPoSE dimension weight")
    fig.tight_layout()
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Figure 8: human dimension ratings vs model similarity
# --------------------------------------------------------------------------- #
def fig8_human_ratings(ds, human=None, save_path=None):
    """Predicted (from ratings) vs reference similarity matrices + scatter."""
    if human is None:
        human = analyses.human_rating_similarity(ds)

    from scipy.cluster.hierarchy import linkage, fcluster

    d = similarity.squareformq(1.0 - human.true_sim)
    z = linkage(d, method="centroid")
    order = np.argsort(fcluster(z, t=8, criterion="maxclust"))

    pred = human.predicted_sim[np.ix_(order, order)]
    true = human.true_sim[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pred, vmin=0, vmax=1, cmap=_VIRIDIS)
    axes[0].set_title("Similarity (dimension ratings)")
    axes[0].axis("off")
    axes[1].imshow(true, vmin=0, vmax=1, cmap=_VIRIDIS)
    axes[1].set_title("Similarity (reference)")
    axes[1].axis("off")
    axes[2].plot(similarity.squareformq(human.predicted_sim),
                 similarity.squareformq(human.true_sim), "o", color="0.5", markersize=3)
    axes[2].plot([0, 1], [0, 1], "k")
    axes[2].set_xlabel("similarity from ratings")
    axes[2].set_ylabel("similarity (reference)")
    axes[2].set_aspect("equal")
    axes[2].set_title(f"r = {human.r:.2f}")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Extended Data Figure 1: dimension reproducibility
# --------------------------------------------------------------------------- #
def extfig1_reproducibility(ds, sortind0, result=None, save_path=None):
    """Per-dimension reproducibility across 20 models (mean + CI band + points)."""
    if result is None:
        result = reproducibility.dimension_reproducibility(ds.embedding, sortind0)

    n = result.mean.size
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, result.lower, result.upper, color="0.7")
    ax.plot(x, result.mean, "k", lw=1)
    for j in range(result.reproducibility.shape[1]):
        ax.plot(x, result.reproducibility[:, j], "o", color="k", markersize=3)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dimension number")
    ax.set_ylabel("Reproducibility (max r)")
    ax.set_title(f"Dimension reproducibility (rank r = {result.rank_corr:.2f})")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Extended Data Figure 2: all 49 dimension word clouds
# --------------------------------------------------------------------------- #
def extfig2_all_dimension_clouds(ds, save_path=None):
    """7x7 grid of word clouds, one per dimension."""
    fig, axes = plt.subplots(7, 7, figsize=(16, 16))
    axes = axes.ravel()
    for d in range(ds.n_dims):
        dw = dimlabels.dimension_words(ds.dimlabel_answers, d)
        axes[d].imshow(_wordcloud_image(dw, ds.colors[d], width=300, height=300))
        axes[d].set_title(f"Dim {d + 1}: {ds.labels[d]}", fontsize=7)
        axes[d].axis("off")
    for extra in range(ds.n_dims, len(axes)):
        axes[extra].axis("off")
    fig.tight_layout()
    return _finish(fig, save_path)
