# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the analysis and figure-reproduction codebase for the paper:

> Hebart, M.N., Zheng, C.Y., Pereira, F., & Baker, C.I. (2020). **Revealing the multidimensional mental representations of natural objects underlying human similarity judgments.** *Nature Human Behaviour*, 4, 1173–1185.

All code is written in MATLAB. The core result is the **SPoSE** (Sparse Positive Similarity Embedding): a 49-dimensional sparse embedding of 1,854 natural objects derived from human odd-one-out triplet judgments.

## Running the Analyses

**Requirements:** MATLAB R2016b or later. Navigate to `osfstorage-archive/` before running.

```matlab
% Reproduce all figures and analyses (run from osfstorage-archive/)
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

### Helper Functions (`helper_functions/`)

| Function | Purpose |
|---|---|
| `embedding2sim.m` | Converts an N×D embedding to an N×N pairwise similarity matrix via the triplet softmax formula. O(N³) — slow for large N. |
| `squareformq.m` | Converts between a square symmetric matrix and its lower-triangular vector form (generalized `squareform`). |
| `clustering_algorithm.m` | Groups objects by their top-K dominant dimensions for visualization only (used to sort the similarity matrix). |
| `predict_category.m` | Leave-one-out category classification using Euclidean distance to per-category centroids; run for both SPoSE and semantic word vectors. |
| `viridis.m` | Viridis colormap. |
| `external/fdr_bh/fdr_bh.m` | Benjamini-Hochberg FDR correction. |

### Key Data Files (`data/`)

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

### Key Variable Files (`variables/`)

| File | Contents |
|---|---|
| `sortind.mat` | Index array to convert original object order to the sorted order used throughout |
| `im.mat` | Low-resolution thumbnail images for all 1854 objects |
| `words.mat`, `words48.mat` | Object name strings (all 1854; 48-object subset) |
| `labels.mat`, `labels_short.mat` | Human-readable labels for each of the 49 dimensions |
| `colors.txt` | Hex color per dimension (49 colors, parsed into RGB at runtime) |
| `unique_id.mat` | Unique string identifiers matching objects to their image files |

### Reference Models (`reference_models/s01/` … `s20/`)

Each subdirectory contains one `.txt` file — a sparse embedding from a single model run at a specific iteration. Used in Extended Data Figure 1 to assess reproducibility of the 49 dimensions across independent model fits.

## Important Implementation Details

- **Object indexing:** Raw data files use 0-based indexing; the script converts to 1-based on load (`+1`). The `sortind.mat` variable is required to map between the original and sorted object orders — always apply it when loading new embeddings.
- **Caching:** The dimension-ablation analysis (Figure 6) generates `spose_similarity_reduced.mat` and `spose_embedding49_reduced.mat` on first run and caches them. Delete these files to force recomputation.
- **t-SNE:** Requires the Van der Maaten t-SNE implementation (not included); download from `https://lvdmaaten.github.io/tsne/#implementations` and place in `osfstorage-archive/`.
- **`subtightplot`:** Used extensively for subplot layout; must be on the MATLAB path.
- **Bug note (patched 2020-11-21):** `ctmp` in `embedding2sim.m` and the inline similarity loop in `make_figures_behavsim.m` was not re-initialized inside the loop in the original release. The current code is fixed; results differ from the original by ~0.001 in r.
