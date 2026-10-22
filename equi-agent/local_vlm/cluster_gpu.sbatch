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
model="${1:?Pass medgemma, gemma, or gemma4}"
case "$model" in medgemma|gemma|gemma4) ;; *) echo "Pass medgemma, gemma, or gemma4" >&2; exit 1 ;; esac
cd "$REPO_ROOT"
venv="${VLM_VENV:-$REPO_ROOT/.venv}"
output_root="${VLM_OUTPUT_ROOT:-$REPO_ROOT/equi-agent/outputs/local_vlm}"
snapshot_file="$output_root/${model}_snapshot.txt"
[[ -f "$snapshot_file" ]] || { echo "Download $model first (see README)." >&2; exit 1; }
IFS= read -r snapshot < "$snapshot_file"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
printf 'GPU job node: %s\n' "$(hostname)"
printf 'Slurm job: %s; assigned GPUs: %s\n' "${SLURM_JOB_ID:-unset}" "${SLURM_JOB_GPUS:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "nvidia-smi failed; checking CUDA inside the job step." >&2
  fi
else
  echo "nvidia-smi is unavailable; checking CUDA inside the job step." >&2
fi
# Enter a Slurm job step so the scheduler applies task GPU visibility/binding.
launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  command -v srun >/dev/null || { echo "srun is required inside a Slurm allocation." >&2; exit 1; }
  launcher=(srun --ntasks=1)
fi
"${launcher[@]}" "$venv/bin/python" - <<'PY'
import os
import torch

for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
    print(f"{name}: {os.environ.get(name, 'unset')}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}; GPU count: {torch.cuda.device_count()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible inside the allocated job step. Check the Slurm allocation/node/driver before model inference.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}; VRAM: {props.total_memory / 2**30:.1f} GiB", flush=True)
PY
args=(--manifest "${FAIRVISION_MANIFEST:-$output_root/cohort.csv}"
      --data-root "$FAIRVISION_DATA_ROOT" --model "$model" --model-path "$snapshot"
      --device cuda --dtype "${VLM_DTYPE:-bfloat16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  args+=(--limit-per-task 2 --out-dir "$output_root/${model}_smoke")
else
  args+=(--out-dir "$output_root/$model")
fi
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then args+=(--retry-errors); fi
"${launcher[@]}" "$venv/bin/python" equi-agent/scripts/fairvision_local_vlm.py infer "${args[@]}"
