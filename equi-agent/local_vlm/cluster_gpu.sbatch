#!/usr/bin/env bash
# Stokes/Newton: pass the real partition and any required account to sbatch.
#SBATCH --job-name=fairvision-vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${REPO_ROOT:?Export the absolute RETFound checkout path before submission}"
: "${FAIRVISION_DATA_ROOT:?Export the directory containing FairVision data}"
model="${1:?Pass medgemma or gemma}"
case "$model" in medgemma|gemma) ;; *) echo "Pass medgemma or gemma" >&2; exit 1 ;; esac
cd "$REPO_ROOT"
venv="${VLM_VENV:-$REPO_ROOT/.venv}"
output_root="${VLM_OUTPUT_ROOT:-$REPO_ROOT/equi-agent/outputs/local_vlm}"
snapshot_file="$output_root/${model}_snapshot.txt"
[[ -f "$snapshot_file" ]] || { echo "Download $model first (see README)." >&2; exit 1; }
IFS= read -r snapshot < "$snapshot_file"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
args=(--manifest "${FAIRVISION_MANIFEST:-$output_root/cohort.csv}"
      --data-root "$FAIRVISION_DATA_ROOT" --model "$model" --model-path "$snapshot"
      --device cuda --dtype "${VLM_DTYPE:-bfloat16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  args+=(--limit-per-task 2 --out-dir "$output_root/${model}_smoke")
else
  args+=(--out-dir "$output_root/$model")
fi
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then args+=(--retry-errors); fi
"$venv/bin/python" equi-agent/scripts/fairvision_local_vlm.py infer "${args[@]}"
