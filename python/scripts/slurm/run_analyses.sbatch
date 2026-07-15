#!/bin/bash
#SBATCH --job-name=spose-analyses
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/run_analyses_%j.out
#
# Run the full numeric analysis spine and write a JSON summary.
# backend.num_workers() picks up SLURM_CPUS_PER_TASK automatically.
#     sbatch scripts/slurm/run_analyses.sbatch
set -euo pipefail

module load python 2>/dev/null || true
# source .venv/bin/activate

# export THINGS_ARCHIVE_DIR=/path/to/osfstorage-archive
# export THINGS_CACHE_DIR=$SCRATCH/spose_cache

mkdir -p logs results
python scripts/run_analyses.py --backend numba --out results/summary_${SLURM_JOB_ID:-local}.json
