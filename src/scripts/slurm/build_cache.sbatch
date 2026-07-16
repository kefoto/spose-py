#!/bin/bash
#SBATCH --job-name=spose-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_cache_%j.out
#
# Build the rebuildable cache (Fig-6 ablation + Fig-4/5 t-SNE layout) once.
# Uses the GPU embedding2sim backend. Submit from the src/ directory:
#     sbatch scripts/slurm/build_cache.sbatch
set -euo pipefail

module load python cuda 2>/dev/null || true   # adapt to your cluster's modules
# source .venv/bin/activate                    # or activate your env

export THINGS_DEVICE=cuda
# export THINGS_DATA_DIR=/path/to/data/spose
# export THINGS_CACHE_DIR=$SCRATCH/spose_cache

mkdir -p logs
python scripts/build_cache.py --backend gpu --tsne --recompute-sim
