# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the analysis and figure-reproduction codebase for the paper:

> Hebart, M.N., Zheng, C.Y., Pereira, F., & Baker, C.I. (2020). **Revealing the multidimensional mental representations of natural objects underlying human similarity judgments.** *Nature Human Behaviour*, 4, 1173–1185.

The core result is the **SPoSE** (Sparse Positive Similarity Embedding): a 49-dimensional sparse embedding of 1,854 natural objects derived from human odd-one-out triplet judgments.

The **Python port in `src/` is the active codebase** — see `src/README.md`. The original MATLAB release is kept for reference only, in `reference/osfstorage-archive/`, and is not read by any Python code. The MATLAB-specific sections below describe that reference copy.

## Repository Layout

```
things_spose/            # the library, organised by pipeline stage
├── core/                # shared infra: paths, backend, artifacts, fdr
├── data/                # stage 1: dataio (load/assemble), sampling (triplets)
├── training/            # stage 2: model, train, run_log, checkpoint
└── analysis/            # stage 3: similarity, analyses, ..., viz

scripts/                 # orchestration entry points
notebooks/               # 00_overview + one per figure
cache/                   # rebuildable artifacts (gitignored)

data/                    # all data (gitignored)
├── spose/               # the MATLAB-derived dataset the analyses read
│   ├── data/            # embedding, similarity, RDMs, ratings, ...
│   ├── variables/       # sortind, images, words, labels, colors
│   └── reference_models/# s01..s20, for Ext-Fig-1 reproducibility
└── triplet_dataset/     # raw odd-one-out triplets (train/validation/test)

reference/               # not read by any code
├── nihms-1621051.pdf    # the paper
└── osfstorage-archive/  # the original MATLAB source
```

Python code never hardcodes these locations: everything resolves through
`things_spose/core/paths.py` (`paths.data()`, `paths.variable()`,
`paths.triplets()`, `paths.REFERENCE_MODELS_DIR`), overridable via
`THINGS_DATA_DIR` / `THINGS_TRIPLET_DIR` / `THINGS_CACHE_DIR`. Change paths
there, not at call sites. **`paths.py` derives the repo root from its own depth**
(`__file__` three levels up) — moving the file or changing the directory nesting
silently breaks every data path, so update that arithmetic whenever the layout
changes. This has already caused two outages.

Imports follow the stage layout — there is no re-export at the package root:

```python
from things_spose.core import paths, backend
from things_spose.data import dataio, sampling
from things_spose.training import train, checkpoint
from things_spose.analysis import analyses, viz
```

## Running the Analyses (Python)

```bash
python scripts/verify_parity.py        # check the port against MATLAB/paper values
python scripts/build_cache.py --tsne   # one-time heavy precompute
python scripts/run_analyses.py         # numeric spine -> JSON
python scripts/render_figures.py       # all figures -> SVG
python scripts/train_spose.py          # fit a fresh embedding from triplets
```

## Running the Analyses (MATLAB reference)

**Requirements:** MATLAB R2016b or later. The script expects `data/` and `variables/`
as siblings, so point it at `data/spose/` or copy them back next to the `.m` files.

```matlab
% Reproduce all figures and analyses
make_figures_behavsim
```

- First run: ~10–20 minutes (generates and caches `spose_similarity_reduced.mat` and `spose_embedding49_reduced.mat` in `data/`)
- Subsequent runs: ~1–2 minutes (loads cached files)

To run a specific figure section, jump directly into `make_figures_behavsim.m` at the relevant `%% Figure N:` section. All data is loaded at the top of the script and remains in the workspace.

**Saving figures:** Each figure section has a `dosave = 0` flag. Set to `1` to export SVGs.

## Architecture

### Entry Point
`make_figures_behavsim.m` — monolithic script that loads all data, runs all analyses, and generates all figures. Sections are delimited by `%% Figure N:` comments and can be run independently after the data-loading preamble completes.

### Data Flow

```
spose_embedding_49d_sorted.txt   (1854×49 embedding)
        ↓ dot product
dot_product49                    (1854×1854 proximity matrix)
        ↓ triplet softmax (embedding2sim.m)
spose_similarity.mat             (1854×1854 pairwise similarity)
        ↓ compared against
RDM48_triplet.mat                (48×48 behavioral similarity, subset)
```

The triplet softmax: `P(i closer to j than k) = exp(sim(i,j)) / [exp(sim(i,j)) + exp(sim(i,k)) + exp(sim(j,k))]`

### Helper Functions (`reference/osfstorage-archive/helper_functions/`)

| Function | Purpose |
|---|---|
| `embedding2sim.m` | Converts an N×D embedding to an N×N pairwise similarity matrix via the triplet softmax formula. O(N³) — slow for large N. |
| `squareformq.m` | Converts between a square symmetric matrix and its lower-triangular vector form (generalized `squareform`). |
| `clustering_algorithm.m` | Groups objects by their top-K dominant dimensions for visualization only (used to sort the similarity matrix). |
| `predict_category.m` | Leave-one-out category classification using Euclidean distance to per-category centroids; run for both SPoSE and semantic word vectors. |
| `viridis.m` | Viridis colormap. |
| `external/fdr_bh/fdr_bh.m` | Benjamini-Hochberg FDR correction. |

### Key Data Files (`data/spose/data/`, via `paths.data()`)

| File | Contents |
|---|---|
| `spose_embedding_49d_sorted.txt` | Main result: 1854×49 SPoSE embedding (objects sorted by `sortind`) |
| `spose_similarity.mat` | Precomputed 1854×1854 similarity matrix from `embedding2sim` |
| `data1854_batch5_test10.txt` | Triplet test set (0-indexed; script adds +1 on load) |
| `RDM48_triplet.mat` / `_splithalf.mat` | Behavioral similarity for 48-object subset; split-half for reliability |
| `typicality_data27.mat` | Typicality ratings for objects in 27 categories |
| `dimension_ratings.mat` | 20 subjects × 20 objects × 49 dims explicit dimension ratings |
| `dimlabel_answers.mat` | Free-text participant answers used to build word clouds per dimension |
| `sensevec_augmented_with_wordvec.mat` | Semantic word vectors (baseline comparison model) |
| `triplets_noiseceiling.csv` | 1000 repeated triplets for estimating human noise ceiling |
| `category_mat_manual.mat` | Binary matrix: which of 1854 objects belong to which of 27 categories |

### Key Variable Files (`data/spose/variables/`, via `paths.variable()`)

| File | Contents |
|---|---|
| `sortind.mat` | Index array to convert original object order to the sorted order used throughout |
| `im.mat` | Low-resolution thumbnail images for all 1854 objects |
| `words.mat`, `words48.mat` | Object name strings (all 1854; 48-object subset) |
| `labels.mat`, `labels_short.mat` | Human-readable labels for each of the 49 dimensions |
| `colors.txt` | Hex color per dimension (49 colors, parsed into RGB at runtime) |
| `unique_id.mat` | Unique string identifiers matching objects to their image files |

### Reference Models (`data/spose/reference_models/s01/` … `s20/`)

Each subdirectory contains one `.txt` file — a sparse embedding from a single model run at a specific iteration. Used in Extended Data Figure 1 to assess reproducibility of the 49 dimensions across independent model fits.

## Important Implementation Details

- **Object indexing:** Raw data files use 0-based indexing; the script converts to 1-based on load (`+1`). The `sortind.mat` variable is required to map between the original and sorted object orders — always apply it when loading new embeddings.
- **Caching:** The dimension-ablation analysis (Figure 6) generates `spose_similarity_reduced.mat` and `spose_embedding49_reduced.mat` on first run and caches them. Delete these files to force recomputation.
- **t-SNE:** The MATLAB script requires the Van der Maaten t-SNE implementation (not included); download from `https://lvdmaaten.github.io/tsne/#implementations` and place in `reference/osfstorage-archive/`. The Python port does not need it — `things_spose/tsne.py` ports `d2p`/`tsne_p` directly.
- **`subtightplot`:** Used extensively for subplot layout; must be on the MATLAB path.
- **Bug note (patched 2020-11-21):** `ctmp` in `embedding2sim.m` and the inline similarity loop in `make_figures_behavsim.m` was not re-initialized inside the loop in the original release. The current code is fixed; results differ from the original by ~0.001 in r.
