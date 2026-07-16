# SPoSE MATLAB → Python Port — Plan

> **Provenance note.** This document was reconstructed from the current state of
> the `src/` package and the agreed next steps, because the original planning
> conversation was not saved to a file. If anything here diverges from what we
> decided, edit this file — it is the source of truth going forward.

> **Status (2026-07-14): COMPLETE.** All four remaining phases are done —
> orchestration scripts (`scripts/build_cache.py`, `run_analyses.py`,
> `render_figures.py`, `verify_parity.py`, `run_notebooks.py`), the visualization
> layer (`things_spose/viz.py`, `tsne.py`, `dimlabels.py`, `artifacts.py`), one
> notebook per figure (`notebooks/`), Slurm scaffolding (`scripts/slurm/`),
> `pyproject.toml`, and `README.md`. Parity verified: `embedding2sim` r=1.0 vs
> shipped, Fig2b r=0.872, triplet acc 64.6% / ceiling 67.2%, classification
> SPoSE 86.4% > wordvec 86.0%, Fig6 ablation 9–14 dims, human-rating r=0.853,
> reproducibility rank r=0.754. All figures render and all notebooks execute
> headlessly with zero errors. See `README.md` for usage.

## Goal

Reproduce the analyses and figures of Hebart, Zheng, Pereira & Baker (2020),
*Nature Human Behaviour* — currently a monolithic MATLAB script
(`reference/osfstorage-archive/make_figures_behavsim.m`) — as a clean, testable Python
package (`things_spose/`) that runs **on a compute cluster** (GPU node or fat CPU
node), with numeric results matching the paper/MATLAB to the documented
tolerances.

## Runtime target: cluster

- **No fussing over PyTorch versioning.** `requirements.txt` pins `torch>=2.0`
  with no upper bound; the cluster's module system / conda env supplies the right
  CUDA wheel. Same for the rest of the scientific stack.
- `backend.py` already adapts to the cluster: honors `SLURM_CPUS_PER_TASK`,
  `CUDA_VISIBLE_DEVICES`, and the `THINGS_DEVICE` / `THINGS_NUM_WORKERS`
  overrides, and calls `configure_threads()` to avoid BLAS oversubscription.
- The one expensive computation — `embedding2sim` (O(N³), ~10–15 min in MATLAB) —
  has three interchangeable backends: **`gpu`** (batched torch, the cluster fast
  path), **`numba`** (parallel fused CPU kernel), **`numpy`** (reference). Chosen
  at runtime via `backend.resolve_backend`.
- Dataset location is overridable with `THINGS_DATA_DIR`; triplets with
  `THINGS_TRIPLET_DIR`; cache location with `THINGS_CACHE_DIR` (see `paths.py`).

## Architecture

```
src/
├── things_spose/          # library (import-light; heavy submodules lazy)
│   ├── paths.py           # dataset + cache locations (env-overridable)
│   ├── backend.py         # device / worker / backend selection for the cluster
│   ├── dataio.py          # load_dataset() → Dataset; ports the MATLAB preamble
│   ├── similarity.py      # embedding2sim (3 backends), embedding2sim48, squareformq
│   ├── _numba_kernels.py  # njit(parallel) fused emb2sim kernel
│   ├── analyses.py        # Figs 2, 6, 7, 8 numeric analyses
│   ├── classify.py        # LOO nearest-centroid category classification
│   ├── reproducibility.py # Extended Data Fig 1
│   ├── clustering.py      # clustering_algorithm.m (Fig 1 ordering)
│   └── external_fdr.py    # Benjamini–Hochberg FDR
├── scripts/               # (empty) orchestration entry points — TO BUILD
├── notebooks/             # (empty) figure notebooks — TO BUILD
└── cache/                 # (empty, gitignored) rebuildable artifacts
```

### Data flow (unchanged from MATLAB)

```
spose_embedding_49d_sorted.txt  (1854×49)
        ↓ dot product
dot_product  (1854×1854)
        ↓ triplet softmax (embedding2sim)
spose_similarity  (1854×1854)
        ↓ compared against
RDM48_triplet  (48×48 behavioral)
```

## Status: what's already ported

The **computational core is essentially complete.** Mapping to the MATLAB
sections of `make_figures_behavsim.m`:

| MATLAB section | Python | Status |
|---|---|---|
| Preamble: load data, `sortind` remap, 0/1-based fixes, colors, images | `dataio.load_dataset` | ✅ Done |
| `embedding2sim.m`, `squareformq.m` | `similarity.py` | ✅ Done (3 backends) |
| Fig 1 object ordering (`clustering_algorithm.m`) | `clustering.py` | ✅ Done (numeric only) |
| Fig 2a triplet prediction | `analyses.predict_triplets` | ✅ Done |
| Fig 2a noise ceiling / % performance | `analyses.noise_ceiling`, `percent_performance` | ✅ Done |
| Fig 2b 48-object similarity correlation | `analyses.sim48_correlation`, `embedding2sim48` | ✅ Done |
| Fig 6 dimension ablation | `analyses.dimension_ablation` | ✅ Done (needs cached reduced matrices) |
| Classification analysis (`predict_category.m`) | `classify.predict_category` | ✅ Done |
| Fig 7 typicality Spearman + FDR + CI | `analyses.typicality_correlations` | ✅ Done |
| Fig 8 human dimension ratings → similarity | `analyses.human_rating_similarity` | ✅ Done |
| Extended Data Fig 1 dimension reproducibility | `reproducibility.dimension_reproducibility` | ✅ Done |

## Remaining work

### Phase 1 — Orchestration / entry points (make it run end-to-end)
- [ ] `scripts/build_cache.py` — the artifact builder referenced by `.gitignore`.
      Recompute `spose_sim` from the embedding (independent of the shipped
      `.mat`) and, for Fig 6, build the **reduced** embeddings/similarity matrices
      `(49, N, D)` / `(49, N, N)` that `dimension_ablation` consumes. Save as
      `.npz`/`.npy`/`.mmap` in `cache/`. Uses the GPU backend when available.
- [ ] `scripts/run_analyses.py` — the equivalent of `make_figures_behavsim.m`'s
      numeric spine: load dataset, run every analysis above, and emit a single
      results summary (JSON + printed table). This is what runs as a Slurm job.

### Phase 2 — Parity verification
- [ ] A `tests/` (or `scripts/verify_parity.py`) harness that asserts Python
      outputs match the MATLAB / paper numbers within tolerance. Targets to lock:
  - `embedding2sim` agrees with shipped `spose_similarity.mat` to ~1e-3 in r
    (per the CLAUDE.md bug-fix note), and the three backends agree with each
    other.
  - Fig 2b: model-vs-behavior **r ≈ 0.87** (note: the figure's `R = 0.90`
    is a static annotation, not the computed value — already documented in
    `sim48_correlation`).
  - Fig 2a triplet-prediction accuracy and human noise ceiling.
  - Classification accuracies (SPoSE vs word-vector baseline).
  - Fig 6 ablation cutoffs; Fig 7 rho/FDR; Fig 8 r + randomization p;
    Ext Data Fig 1 reproducibility.
  - `squareformq` column-major ordering matches MATLAB (matters for bootstrap
    resample alignment).

### Phase 3 — Visualization (`things_spose/viz.py`, not yet created)
Numeric analyses are done; the **plotting layer is not ported**. Needed:
- [ ] Fig 1: sorted similarity matrix heatmap (uses `clustering_algorithm`).
- [ ] Fig 2: prediction/similarity scatter + CI plots.
- [ ] Fig 3: example object images per dimension **+ word clouds** from
      `dimlabel_answers` (the `wordcloud` dependency is already in requirements).
- [ ] Fig 4: "crystal" / dimension-weight bar plots (e.g. abacus example).
- [ ] Fig 5: zoomed single-object examples (microscope/bottle/squid).
- [ ] Fig 6/7/8: ablation curve, typicality bars, human-rating scatter.
- [ ] Extended Data Fig 2: all-dimensions overview (no images variant).
- [ ] **t-SNE**: MATLAB used Van der Maaten's implementation (not included).
      Replace with `sklearn.manifold.TSNE` (or `openTSNE`); flag that the exact
      layout won't match MATLAB, only the qualitative structure.
- [ ] `dosave`-equivalent SVG export flag for each figure.

### Phase 4 — Notebooks & docs
- [ ] `notebooks/` — a driver notebook mirroring the figure sections, calling the
      library so results/figures render inline. `ipykernel` already in reqs.
- [ ] Short README for `src/` (install, `THINGS_*` env vars, how to run on the
      cluster / a sample `sbatch` invocation).

## Open questions / decisions to confirm
- **Reproducibility of RNG:** MATLAB's exact `rng` stream can't be reproduced;
  we reproduce the *procedure* (seeds + resample counts), so bootstrap/permutation
  results match only in the last digits. Confirmed acceptable — parity tolerances
  above are set with this in mind.
- **Recompute vs. trust shipped `.mat`:** `dataio` currently loads the shipped
  `spose_similarity.mat`. `build_cache.py` should *also* be able to recompute it
  from the embedding so the pipeline is self-contained on the cluster. Keep both.
- **Which figures actually need rendering** vs. numbers-only for the cluster run?
  (Plotting can be deferred if the immediate goal is the numeric reproduction.)
