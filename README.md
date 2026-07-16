# SPoSE analyses — Python port

Python reimplementation of the MATLAB analysis/figure code for:

> Hebart, M.N., Zheng, C.Y., Pereira, F., & Baker, C.I. (2020). *Revealing the
> multidimensional mental representations of natural objects underlying human
> similarity judgments.* **Nature Human Behaviour**, 4, 1173–1185.

The original monolithic script (`reference/osfstorage-archive/make_figures_behavsim.m`)
is ported to an importable package (`things_spose/`), a set of orchestration scripts
(`scripts/`), and one notebook per figure (`notebooks/`). It is designed to run
on a compute cluster (GPU or fat CPU node).

## Data

`/data` folder contains the data for triplet decisions and computed spose embeddings

`/triplet_dataset`:

https://osf.io/f5rn6/files/osfstorage

`/spose` folder contains `/data`, `/reference_models`, and `variables`:

https://osf.io/z2784/overview provide the initial placeholders

## Layout

The library is organised by pipeline stage: **data → training → analysis**, over a
shared **core**.

```
.
├── things_spose/
│   ├── core/                # shared infra (depends on nothing else here)
│   │   ├── paths.py         # dataset + cache locations (env-overridable)
│   │   ├── backend.py       # device / worker / backend selection
│   │   ├── artifacts.py     # cache artifact filenames + loaders
│   │   ├── external_fdr.py  # Benjamini-Hochberg FDR
│   │   └── _numba_kernels.py
│   ├── data/                # stage 1 — load / assemble / sample
│   │   ├── dataio.py        # load_dataset(): ports the MATLAB preamble
│   │   └── sampling.py      # triplet loading + train/val splitting
│   ├── training/            # stage 2 — fit an embedding
│   │   ├── model.py         # the SPoSE model (nn.Linear) + l1_regularization
│   │   ├── train.py         # objective + training loop
│   │   ├── run_log.py       # per-run logging
│   │   └── checkpoint.py    # embedding + model/optimizer state
│   └── analysis/            # stage 3 — numbers + figures from an embedding
│       ├── similarity.py    # embedding2sim (gpu/numba/numpy), squareformq
│       ├── analyses.py      # Figs 2, 6, 7, 8 numerics
│       ├── classify.py      # LOO nearest-centroid classification
│       ├── reproducibility.py, clustering.py
│       ├── dimlabels.py     # word-cloud data for Figs 3 / Ext-2
│       ├── tsne.py          # MDS init + faithful d2p/tsne_p for Figs 4/5
│       └── viz.py           # matplotlib figures fig1..fig8, ext_fig1/2
├── scripts/
│   ├── build_cache.py   # one-time heavy precompute (ablation + t-SNE)
│   ├── run_analyses.py  # numeric spine -> JSON summary
│   ├── render_figures.py# all figures -> SVG/PNG (headless)
│   ├── verify_parity.py # assert Python == MATLAB/paper within tolerance
│   ├── train_spose.py   # fit a fresh embedding from triplets
│   ├── run_notebooks.py # execute notebooks headlessly (papermill)
│   └── slurm/*.sbatch   # example cluster jobs
├── notebooks/           # 00_overview + one per figure
└── cache/               # rebuildable artifacts (gitignored)
```

Import submodules from their stage:

```python
from things_spose.core import paths, backend
from things_spose.data import dataio, sampling
from things_spose.training import train, checkpoint
from things_spose.analysis import analyses, viz
```

Data and the original MATLAB release sit alongside the package, and no code
reads from `reference/`:

```
data/
├── spose/               # the MATLAB-derived dataset the analyses read
│   ├── data/            # embedding, similarity, RDMs, ratings, ...
│   ├── variables/       # sortind, images, words, labels, colors
│   └── reference_models/# s01..s20, for Ext-Fig-1 reproducibility
└── triplet_dataset/     # raw odd-one-out triplets (train/validation/test)

reference/               # not read by any code
├── nihms-1621051.pdf    # the paper
└── osfstorage-archive/  # the original MATLAB source
```

## Install

```bash
pip install -r requirements.txt
```

`numba`, `torch`, `wordcloud`, and `papermill` are used but the core analyses
degrade gracefully (e.g. the NumPy similarity backend if Numba/torch are absent).
On a cluster, prefer the module system / conda env to supply the right CUDA wheel;
`requirements.txt` pins no upper bound on `torch`.

## Environment variables

| Variable | Purpose |
|---|---|
| `THINGS_DATA_DIR`    | Path to the `spose/` data directory (default: `data/spose/`). |
| `THINGS_TRIPLET_DIR` | Path to the raw triplets (default: `data/triplet_dataset/`). |
| `THINGS_CACHE_DIR`   | Where cache artifacts are written (default: `cache/`). |
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

# 6) Fit a fresh embedding from the raw triplets (not needed for the figures)
python scripts/train_spose.py --epochs 200 --out spose_embedding_trained.txt

#    ...with a log file and resumable checkpoints every 10 epochs
python scripts/train_spose.py --epochs 200 \
    --checkpoint-dir runs/fit1 --checkpoint-every 10 --log-file runs/fit1/train.log

#    resume that run after an interruption (picks up the newest checkpoint)
python scripts/train_spose.py --epochs 200 --checkpoint-dir runs/fit1 \
    --log-file runs/fit1/train.log --resume
```

`--out` writes the fitted embedding (the result); `--checkpoint-dir` writes
`model_epoch*.tar` with model+optimizer state (the run), which is what `--resume`
reads. Resuming appends to the log rather than truncating it.

Each `figN.ipynb` has a `SAVE = False` flag — set it `True` to export SVGs (the
`dosave` equivalent from the MATLAB script).

## Running on a cluster (Slurm)

```bash
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
- `train.py` follows the reference SPoSE implementation (ViCCo-Group/SPoSE): the
  embedding is an `nn.Linear` fed one-hot triplets, initialised `normal(0.1, 0.01)`,
  with non-negativity enforced by a penalty rather than a hard projection. A fresh
  fit reproduces the *set* of dimensions, not the published embedding's exact
  values or dimension order (see the `train.py` module docstring). Scoring the
  shipped embedding through this code reproduces the paper's 64.60% test accuracy.
