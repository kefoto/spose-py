"""Load the SPoSE dataset and reproduce the index/order fixes from the MATLAB
preamble of ``make_figures_behavsim.m`` (lines 20-90).

Everything is kept **0-based** for NumPy. Object IDs read from the raw triplet
files (0-based on disk) are remapped into the sorted object order via
``sortind`` exactly as MATLAB does, and MATLAB cell/char arrays are unwrapped to
plain Python lists / NumPy arrays.

The main entry point is :func:`load_dataset`, which returns a :class:`Dataset`
holding every array the analyses need. It is cached, so repeated calls in a
notebook are free.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio

from ..core import paths


# --------------------------------------------------------------------------- #
# Helpers to unwrap MATLAB structures loaded by scipy.io.loadmat
# --------------------------------------------------------------------------- #
def _matload(path) -> dict:
    return sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)


def _cellstr(arr) -> list[str]:
    """Flatten a MATLAB cell-array-of-char (any shape) to a list[str]."""
    out = []
    for el in np.asarray(arr).ravel(order="F"):  # column-major = MATLAB order
        if isinstance(el, np.ndarray):
            el = el.item() if el.size == 1 else "".join(map(str, el.ravel()))
        out.append(str(el))
    return out


def _cell_of_arrays(arr) -> list[np.ndarray]:
    """Flatten a MATLAB cell-array-of-numeric to a list of 1-D float arrays."""
    return [np.asarray(el, dtype=float).ravel() for el in np.asarray(arr).ravel(order="F")]


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    # Core embedding / similarity
    embedding: np.ndarray            # (1854, 49) float64, sorted object order
    dot_product: np.ndarray          # (1854, 1854) = embedding @ embedding.T
    spose_sim: np.ndarray            # (1854, 1854) shipped similarity matrix
    dissim: np.ndarray               # 1 - spose_sim

    # Triplets
    triplets_test: np.ndarray        # (146012, 3) int, 0-based, sorted order

    # 48-object behavioral RDMs
    rdm48: np.ndarray                # (48, 48)
    rdm48_split1: np.ndarray
    rdm48_split2: np.ndarray
    wordposition48: np.ndarray       # (48,) int, indices of words48 within words

    # Typicality (Fig 7)
    categories27: list[str]
    best_match27: np.ndarray         # (27,) float, 0-based dim index or NaN
    category27_ind: list[np.ndarray] # per-category 0-based object indices
    category27_subind: np.ndarray    # (17,) 0-based indices into the 27 categories
    typicality_normed: list[np.ndarray]  # per-category normed ratings (27 entries)

    # Human dimension ratings (Fig 8): (20 subjects, 49 dims, 20 objects)
    ratings_translated_all: np.ndarray

    # Free-text dimension labels (Fig 3 word clouds): (20, 49) object array of str
    dimlabel_answers: np.ndarray

    # Semantic word vectors + category matrix (classification)
    sensevec: np.ndarray             # (1854, 300)
    category_mat_manual: np.ndarray  # (1854, 27) uint8

    # Names / labels / colors / images
    words: list[str]                 # (1854,)
    words48: list[str]               # (48,)
    unique_id: list[str]             # (1854,)
    labels: list[str]                # (49,)
    labels_short: list[str]          # (49,)
    colors: np.ndarray               # (49, 3) RGB in [0, 1]
    images: np.ndarray = field(repr=False)  # (1854,) object array of HxWx3 uint8

    @property
    def n_objects(self) -> int:
        return self.embedding.shape[0]

    @property
    def n_dims(self) -> int:
        return self.embedding.shape[1]


# --------------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------------- #
def load_embedding() -> np.ndarray:
    """The 1854x49 SPoSE embedding (already in sorted object order)."""
    return np.loadtxt(paths.data("spose_embedding_49d_sorted.txt"), dtype=np.float64)


def load_sortind() -> np.ndarray:
    """0-based ``sortind`` permutation (maps original -> sorted object order)."""
    si = _matload(paths.variable("sortind.mat"))["sortind"].ravel().astype(np.int64)
    return si - 1  # MATLAB 1-based -> 0-based


def load_triplets_test(sortind0: np.ndarray | None = None) -> np.ndarray:
    """Test triplets, 0-based, remapped into sorted object order.

    Reproduces ``make_figures_behavsim.m`` lines 30 & 42-46. On disk the values
    are 0-based; MATLAB adds 1 then replaces ``value == sortind(i)`` with ``i``,
    which is exactly the inverse permutation of ``sortind`` applied to the raw
    0-based IDs.
    """
    raw = np.loadtxt(paths.data("data1854_batch5_test10.txt"), dtype=np.int64)  # 0-based
    if sortind0 is None:
        sortind0 = load_sortind()
    inv = np.empty_like(sortind0)
    inv[sortind0] = np.arange(sortind0.size)
    return inv[raw]


def load_colors() -> np.ndarray:
    """Parse ``colors.txt`` and apply the exact reindex/tweak from lines 55-70."""
    with open(paths.variable("colors.txt")) as fh:
        hexes = [ln.strip() for ln in fh if ln.strip()]
    col = np.array(
        [[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=float
    ) / 255.0

    # MATLAB: col(1,:) = [];  then swap rows [1 2 3] -> [2 3 1]  (1-based)
    col = col[1:]                       # drop first row
    col[[0, 1, 2]] = col[[1, 2, 0]]     # 0-based version of col([1 2 3])=col([2 3 1])

    # Handpicked palette (1-based MATLAB indices -> subtract 1)
    pick = np.array(
        [1, 20, 3, 38, 9, 7, 62, 57, 13, 6, 24, 25, 50, 48, 36, 53, 46, 28, 62,
         18, 15, 58, 2, 11, 40, 45, 27, 55, 36, 30, 34, 31, 41, 16, 27, 61, 17,
         36, 57, 25, 63]
    ) - 1
    colors = col[pick]
    # colors(end+1:49,:) = col([8:56-length(colors)],:)  -> extend to 49 rows
    need = 49 - colors.shape[0]
    ext = col[np.arange(8, 8 + need) - 1]  # MATLAB 8:(56-len) inclusive, 1-based
    colors = np.vstack([colors, ext])
    colors[45] = colors[45] - 0.2  # MATLAB colors(46,:) darker
    return np.clip(colors, 0.0, 1.0)


def load_images() -> np.ndarray:
    """Thumbnails aligned to ``words``/``unique_id`` order (lines 82-87)."""
    m = _matload(paths.variable("im.mat"))
    im = m["im"].ravel().copy()          # (1854,) object
    imwords = _cellstr(m["imwords"])
    unique_id = _cellstr(_matload(paths.variable("unique_id.mat"))["unique_id"])

    i = np.argsort(unique_id, kind="stable")
    j = np.argsort(imwords, kind="stable")
    im_sorted = im.copy()
    im_sorted[i] = im[j]                  # im(i) = im(j)
    return im_sorted


# --------------------------------------------------------------------------- #
# Full dataset
# --------------------------------------------------------------------------- #
@functools.cache
def load_dataset() -> Dataset:
    """Load and assemble the entire dataset (cached)."""
    paths.check_data()

    embedding = load_embedding()
    dot_product = embedding @ embedding.T

    spose_sim = _matload(paths.data("spose_similarity.mat"))["spose_sim"].astype(np.float64)
    dissim = 1.0 - spose_sim

    sortind0 = load_sortind()
    triplets_test = load_triplets_test(sortind0)

    rdm = _matload(paths.data("RDM48_triplet.mat"))["RDM48_triplet"]
    rdm_sh = _matload(paths.data("RDM48_triplet_splithalf.mat"))
    rdm48_split1 = rdm_sh["RDM48_triplet_split1"]
    rdm48_split2 = rdm_sh["RDM48_triplet_split2"]

    words = _cellstr(_matload(paths.variable("words.mat"))["words"])
    words48 = _cellstr(_matload(paths.variable("words48.mat"))["words48"])
    unique_id = _cellstr(_matload(paths.variable("unique_id.mat"))["unique_id"])
    labels = _cellstr(_matload(paths.variable("labels.mat"))["labels"])
    labels_short = _cellstr(_matload(paths.variable("labels_short.mat"))["labels_short"])

    # wordposition48: order-stable indices of words48 within words (line 89)
    word_to_idx = {w: k for k, w in enumerate(words)}
    wordposition48 = np.array([word_to_idx[w] for w in words48], dtype=np.int64)

    # Typicality
    typ = _matload(paths.data("typicality_data27.mat"))
    categories27 = _cellstr(typ["categories27"])
    best_match27 = typ["best_match27"].ravel().astype(float) - 1.0  # 0-based dim, NaN preserved
    category27_ind = [a.astype(np.int64) - 1 for a in _cell_of_arrays(typ["category27_ind"])]
    category27_subind = typ["category27_subind"].ravel().astype(np.int64) - 1
    typicality_normed = _cell_of_arrays(typ["category27_typicality_rating_normed"])

    ratings_translated_all = _matload(paths.data("dimension_ratings.mat"))[
        "ratings_translated_all"
    ].astype(np.float64)

    dimlabel_answers = _matload(paths.data("dimlabel_answers.mat"))["dimlabel_answers"]

    sensevec = _matload(paths.data("sensevec_augmented_with_wordvec.mat"))[
        "sensevec_augmented"
    ].astype(np.float64)
    category_mat_manual = _matload(paths.data("category_mat_manual.mat"))[
        "category_mat_manual"
    ]

    colors = load_colors()
    images = load_images()

    return Dataset(
        embedding=embedding,
        dot_product=dot_product,
        spose_sim=spose_sim,
        dissim=dissim,
        triplets_test=triplets_test,
        rdm48=rdm,
        rdm48_split1=rdm48_split1,
        rdm48_split2=rdm48_split2,
        wordposition48=wordposition48,
        categories27=categories27,
        best_match27=best_match27,
        category27_ind=category27_ind,
        category27_subind=category27_subind,
        typicality_normed=typicality_normed,
        ratings_translated_all=ratings_translated_all,
        dimlabel_answers=dimlabel_answers,
        sensevec=sensevec,
        category_mat_manual=category_mat_manual,
        words=words,
        words48=words48,
        unique_id=unique_id,
        labels=labels,
        labels_short=labels_short,
        colors=colors,
        images=images,
    )
