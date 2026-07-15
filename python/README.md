# SPoSE analyses — Python port

Python reimplementation of the MATLAB analysis/figure code for:

> Hebart, M.N., Zheng, C.Y., Pereira, F., & Baker, C.I. (2020). *Revealing the
> multidimensional mental representations of natural objects underlying human
> similarity judgments.* **Nature Human Behaviour**, 4, 1173–1185.

The original monolithic script (`osfstorage-archive/make_figures_behavsim.m`) is
ported to an importable package (`things_spose/`), a set of orchestration scripts
(`scripts/`), and one notebook per figure (`notebooks/`). It is designed to run
on a compute cluster (GPU or fat CPU node).

## Layout

```
python/
├── things_spose/        # library
│   ├── dataio.py        # load_dataset(): ports the MATLAB preamble
│   ├── similarity.py    # embedding2sim (gpu/numba/numpy), squareformq
│   ├── analyses.py      # Figs 2, 6, 7, 8 numerics
│   ├── classify.py      # LOO nearest-centroid classification
│   ├── reproducibility.py, clustering.py, external_fdr.py
│   ├── dimlabels.py     # word-cloud data for Figs 3 / Ext-2
│   ├── tsne.py          # MDS init + faithful d2p/tsne_p for Figs 4/5
│   ├── viz.py           # matplotlib figures fig1..fig8, ext_fig1/2
│   ├── artifacts.py     # cache artifact filenames + loaders
│   └── backend.py, paths.py   # cluster device/worker/path selection
├── scripts/
│   ├── build_cache.py   # one-time heavy precompute (ablation + t-SNE)
│   ├── run_analyses.py  # numeric spine -> JSON summary
│   ├── render_figures.py# all figures -> SVG/PNG (headless)
│   ├── verify_parity.py # assert Python == MATLAB/paper within tolerance
│   ├── run_notebooks.py # execute notebooks headlessly (papermill)
│   └── slurm/*.sbatch   # example cluster jobs
├── notebooks/           # 00_overview + one per figure
└── cache/               # rebuildable artifacts (gitignored)
```

## Install

```bash
cd python
pip install -r requirements.txt
```

`numba`, `torch`, `wordcloud`, and `papermill` are used but the core analyses
degrade gracefully (e.g. the NumPy similarity backend if Numba/torch are absent).
On a cluster, prefer the module system / conda env to supply the right CUDA wheel;
`requirements.txt` pins no upper bound on `torch`.

## Environment variables

| Variable | Purpose |
|---|---|
| `THINGS_ARCHIVE_DIR` | Path to `osfstorage-archive/` (default: repo root). |
| `THINGS_CACHE_DIR`   | Where cache artifacts are written (default: `python/cache/`). |
| `THINGS_DEVICE`      | `cpu` / `cuda` / `cuda:N` / `mps` / `auto` for the GPU backend. |
| `THINGS_NUM_WORKERS` | CPU thread count (else `SLURM_CPUS_PER_TASK`, else `os.cpu_count()`). |

`CUDA_VISIBLE_DEVICES` and `SLURM_CPUS_PER_TASK` are honoured automatically.

## Quick start

```bash
# 1) Verify the port matches MATLAB/paper values
python scripts/verify_parity.py                 # add --backends to compare gpu/numba/numpy

# 2) Build the cache once (Fig-6 ablation + Fig-4/5 t-SNE layout); ~4 min CPU / seconds on GPU
python scripts/build_cache.py --tsne            # THINGS_DEVICE=cuda ... --backend gpu on a GPU node

# 3) Run every numeric analysis -> JSON + printed table
python scripts/run_analyses.py --out results.json

# 4) Render every figure headlessly
python scripts/render_figures.py --out figs --ext svg

# 5) Or work interactively
jupyter lab notebooks/          # 00_overview.ipynb first, then figN.ipynb
```

Each `figN.ipynb` has a `SAVE = False` flag — set it `True` to export SVGs (the
`dosave` equivalent from the MATLAB script).

## Running on a cluster (Slurm)

```bash
cd python
cache_job=$(sbatch --parsable scripts/slurm/build_cache.sbatch)
sbatch --dependency=afterok:$cache_job scripts/slurm/run_analyses.sbatch
sbatch --dependency=afterok:$cache_job scripts/slurm/render_notebooks.sbatch
```

Edit the `module load` / env-activation lines and partition names in each
`.sbatch` to match your site.

## Notes on parity

- `embedding2sim` reproduces the shipped `spose_similarity.mat` to r > 0.999
  (see the bug-fix note in the repo `CLAUDE.md`); the three backends agree to ~1e-3.
- Figure 2b model-vs-behaviour correlation is **r ≈ 0.87** for the released data;
  the figure's `R = 0.90` in the paper is a static annotation.
- Bootstrap / permutation / t-SNE results reproduce the *procedure* (same seeds
  and resample counts), not MATLAB's exact RNG stream, so they match to the last
  digits only. t-SNE reproduces the layout qualitatively, not coordinate-for-coordinate.
