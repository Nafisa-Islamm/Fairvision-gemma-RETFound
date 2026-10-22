#!/usr/bin/env bash
# Stokes/Newton: pass the real partition and any required account to sbatch.
#SBATCH --job-name=fairvision-vlm-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${REPO_ROOT:?Export the absolute RETFound checkout path}"
cd "$REPO_ROOT"
venv="${VLM_CPU_VENV:-${VLM_VENV:-$REPO_ROOT/.venv}}"
output_root="${VLM_OUTPUT_ROOT:-$REPO_ROOT/equi-agent/outputs/local_vlm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
stage="${1:-check}"
case "$stage" in
  check)
    : "${FAIRVISION_DATA_ROOT:?Export your FairVision data directory}"
    "$venv/bin/python" equi-agent/scripts/fairvision_local_vlm.py prepare-manifest --output "$output_root/cohort.csv"
    "$venv/bin/python" equi-agent/scripts/fairvision_local_vlm.py check \
      --manifest "${FAIRVISION_MANIFEST:-$output_root/cohort.csv}" --data-root "$FAIRVISION_DATA_ROOT"
    ;;
  report)
    run_dirs=()
    read -r -a report_models <<< "${VLM_REPORT_MODELS:-medgemma gemma}"
    for report_model in "${report_models[@]}"; do
      case "$report_model" in medgemma|gemma|gemma4) ;; *) echo "Unknown report model: $report_model" >&2; exit 1 ;; esac
      run_dirs+=("$output_root/$report_model")
    done
    [[ ${#run_dirs[@]} -gt 0 ]] || { echo "No report models selected" >&2; exit 1; }
    "$venv/bin/python" equi-agent/scripts/report_fairvision_local_vlm.py \
      --run-dirs "${run_dirs[@]}" --out-dir "$output_root/table"
    ;;
  *) echo "Pass check or report" >&2; exit 1 ;;
esac
