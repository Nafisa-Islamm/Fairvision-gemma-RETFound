#!/usr/bin/env bash
# The small routing policy trains on cached expert outputs; no GPU is requested.
#SBATCH --job-name=fairroute-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${REPO_ROOT:?Export REPO_ROOT before submission}"
python_bin="${VLM_ROUTER_PYTHON:-$HOME/.venvs/fairvision-gpu-py311/bin/python}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" TOKENIZERS_PARALLELISM=false
"$python_bin" -u equi-agent/scripts/fairroute_grpo.py "$@"
