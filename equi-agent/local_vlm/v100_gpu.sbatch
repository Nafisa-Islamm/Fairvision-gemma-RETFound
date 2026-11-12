#!/usr/bin/env bash
# Experimental V100 route; test each model before replacing H100 jobs.
#SBATCH --job-name=fairvision-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${REPO_ROOT:?Export REPO_ROOT}"
: "${VLM_OUTPUT_ROOT:?Choose a separate V100 precision-specific output root}"
model="${1:?Pass gemma, medgemma, or gemma4-it}"
case "$model" in
  gemma|medgemma) export VLM_VENV="${V100_VENV:-$HOME/.venvs/fairvision-gpu-py311}" ;;
  gemma4-it) export VLM_VENV="${V100_GEMMA4_VENV:-$HOME/.venvs/fairvision-gemma4-py311}" ;;
  *) echo "Use gemma, medgemma, or gemma4-it; unquantized Muse does not fit one 32 GB V100." >&2; exit 1 ;;
esac
export VLM_DTYPE="${V100_DTYPE:-float16}"
case "$VLM_DTYPE" in float16|float32) ;; *) echo "V100 template requires explicit float16 or float32; native BF16 is unavailable." >&2; exit 1 ;; esac
if [[ "$model" == "gemma4-it" && "$VLM_DTYPE" == "float32" ]]; then
  echo "Gemma 4 12B FP32 weights exceed 32 GB; this template requires FP16 for its experimental fit test." >&2
  exit 1
fi
export SMOKE="${SMOKE:-1}"
printf 'V100 experiment: %s; dtype=%s; SMOKE=%s\n' "$model" "$VLM_DTYPE" "$SMOKE"
exec bash "$REPO_ROOT/equi-agent/local_vlm/cluster_gpu.sbatch" "$model"
