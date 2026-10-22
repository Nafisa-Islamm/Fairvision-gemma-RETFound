# Gemma 4 12B on Newton

This workflow uses the requested `google/gemma-4-12B` checkpoint, `AutoModelForMultimodalLM`, the existing FairVision manifest, and the same prompts and image grids. It adds a `gemma4` model identity and report label. The architecture must be `gemma4_unified` and the checkpoint must provide a chat template; if either check fails, stop and confirm the intended checkpoint rather than inventing a template. No fine-tuning is performed.

Keep the old environments and `fairvision-vlm-results-eager` unchanged. The separate environment pins the inspected official Transformers commit `f348d3dcfa9317bed603e01b8ca22b11fe59615c` (5.21.0.dev0); do not install the old `requirements-gpu.txt` over it. Compilation remains disabled. The old 4.53.3 runtime is insufficient for Gemma 4.

Verified locally: existing 10 inference/report tests, Gemma 4 loader selection and identity in saved output, and an additional real two-image generation/reload test on a tiny random Gemma 4 Unified model with PyTorch 2.6 CPU. The pretrained 12B weights and Newton GPU execution require the smoke test below. The installer has not been executed on Newton by the assistant. A fresh dependency lock is saved inside the environment after installation.

## 1. Reconnect and update

In Windows PowerShell, if disconnected:

```powershell
ssh na942507@newton.ist.ucf.edu
```

In the Newton login shell, run each command separately:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export VLM_VENV="$HOME/.venvs/fairvision-gemma4-py311"
export VLM_CPU_VENV="$HOME/.venvs/fairvision-cpu-py311"
export VLM_OUTPUT_ROOT="$HOME/fairvision-vlm-results-gemma4"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export PYTHON_BIN="$(command -v python3.11)"
mkdir -p "$VLM_OUTPUT_ROOT"
cp -n "$HOME/fairvision-vlm-results-eager/cohort.csv" "$VLM_OUTPUT_ROOT/cohort.csv"
```

`PYTHON_BIN` must resolve to Python >=3.10. If it is empty, use the known available `/usr/bin/python3.11` after checking its version, or load the cluster's Python 3.11 module. Exports must be restored after reconnecting. Leave existing pending/running jobs alone.

## 2. Install using a CPU allocation

This requests no GPU. Wait for completion before continuing:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=12G --time=00:45:00 bash equi-agent/local_vlm/install_gemma4.sh
```

Expected final message: `Gemma 4 environment: ...`. If installation fails, stop and inspect the error. Do not replace the old GPU environment. The new installer refuses an unrelated existing environment.

## 3. Download the requested checkpoint

Existing Hugging Face login may be reused. Verify it, then download in a CPU allocation (not the GPU allocation):

```bash
"$VLM_VENV/bin/hf" auth whoami
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=8G --time=02:00:00 "$VLM_VENV/bin/hf" download google/gemma-4-12B --local-dir "$HOME/fairvision-models/gemma-4-12B"
```

If authentication fails, run `"$VLM_VENV/bin/hf" auth login`. Enter credentials only at the secure prompt. Gated-access errors require accepting the model's terms in the browser with the same account. Retry an interrupted download with the same command. Confirm enough filesystem quota before downloading: 12 billion BF16 parameters alone are approximately 24 GB in decimal units, with extra space for metadata, cache and partial files; inspect the actual model repository for exact size.

After successful download:

```bash
printf '%s\n' "$HOME/fairvision-models/gemma-4-12B" > "$VLM_OUTPUT_ROOT/gemma4_snapshot.txt"
"$VLM_VENV/bin/python" -c 'import os; from transformers import AutoConfig, AutoProcessor; p=os.path.expanduser("~/fairvision-models/gemma-4-12B"); c=AutoConfig.from_pretrained(p, local_files_only=True); processor=AutoProcessor.from_pretrained(p, local_files_only=True); print("Architecture:", c.model_type); print("Chat template present:", bool(processor.chat_template)); assert c.model_type == "gemma4_unified" and processor.chat_template, "Stop: confirm checkpoint architecture and chat template"'
```

This final command loads configuration/processor only, not the 12B weights. Do not run full model loading on the login node.

## 4. Six-case H100 smoke test

Use the existing working GPU type and exclude evc26. One H100 has about 80 GB VRAM; actual memory use must be verified by the test. Request 64 GB CPU memory for the larger checkpoint:

```bash
GEMMA4_TEST_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=64G --time=00:30:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/cluster_gpu.sbatch gemma4)
echo "$GEMMA4_TEST_JOB_ID"
squeue -u "$USER"
tail -n 100 "$REPO_ROOT/slurm-fairvision-vlm-${GEMMA4_TEST_JOB_ID}.out"
sacct -j "$GEMMA4_TEST_JOB_ID" --format=JobID,State,Elapsed,NodeList,ExitCode
```

If sbatch returns an error or an empty ID, stop. A log may not exist until the job starts. Proceed only after `COMPLETED`, exit `0:0`, and `Complete: 6/6 rows; 0 errors`. This checks execution, not diagnostic accuracy. Save the job ID before disconnecting.

## 5. Full evaluation (only after a successful smoke test)

```bash
GEMMA4_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=64G --time=06:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/cluster_gpu.sbatch gemma4)
echo "$GEMMA4_JOB_ID"
```

Six hours is a time allowance, not a measured estimate. Read the actual job log. Expected completion is 750/750 rows and zero errors. The existing resume checks work only with unchanged code, data, environment and options. Do not submit two writers to the same output folder.

## 6. CPU report

After the full run completes:

```bash
export VLM_REPORT_MODELS=gemma4
GEMMA4_REPORT_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --cpus-per-task=1 --mem=4G --time=00:10:00 --export=ALL equi-agent/local_vlm/cluster_cpu.sbatch report)
echo "$GEMMA4_REPORT_JOB_ID"
```

After checking that reporting completed with exit 0:

```bash
cat "$VLM_OUTPUT_ROOT/table/table1_local_models.md"
```

Predictions: `$HOME/fairvision-vlm-results-gemma4/gemma4/predictions.csv`. Tables and subgroup scores: `$HOME/fairvision-vlm-results-gemma4/table/`. Existing Gemma 3 and MedGemma runs stay in their old directory. Comparisons must document the different architecture, parameter count and library version, despite matching cases, prompts and image grids.
