#!/bin/bash
#SBATCH --job-name=spose-notebooks
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/render_notebooks_%j.out
#
# Execute every figure notebook headlessly with papermill, plus render all
# figures to SVG. Assumes the cache already exists (submit build_cache first,
# e.g. with --dependency=afterok:<cacheJobId>).
#     sbatch scripts/slurm/render_notebooks.sbatch
set -euo pipefail

module load python 2>/dev/null || true
# source .venv/bin/activate

# export THINGS_ARCHIVE_DIR=/path/to/osfstorage-archive
# export THINGS_CACHE_DIR=$SCRATCH/spose_cache
export MPLBACKEND=Agg

mkdir -p logs
python scripts/render_figures.py --out figs --ext svg
python scripts/run_notebooks.py --out notebooks/executed
