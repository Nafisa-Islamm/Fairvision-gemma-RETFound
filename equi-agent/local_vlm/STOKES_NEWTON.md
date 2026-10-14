# Gemma and MedGemma: Newton GPU, Stokes CPU

Goal: standalone Gemma 3 4B and original MedGemma 4B on the same 750 selected
FairVision patients (250 glaucoma, 250 AMD, 250 DR). No existing foundation
model retraining or paid API calls. Produce macro-F1, worst-group macro-F1,
sensitivity, specificity and balanced accuracy in CSV/Markdown/LaTeX.

Dataset supplied by the user:
`/home/na942507/datasets/FairVision30K/raw`
with disease directories `Amd`, `Dr`, `Glaucoma`.
The updated loader supports those names and case variations in Test directories.
An rsync broken-pipe message was shown: existence of folders does not prove that
all selected images transferred. Complete the image check before inference.

## 1. Clone the code directly from your GitHub fork

The new scripts are on the fairvision-local-gemma branch of your fork.
Clone into a new directory on Stokes and/or Newton; existing checkouts are preserved:

```bash
git clone --branch fairvision-local-gemma --single-branch https://github.com/Nafisa-Islamm/Fairvision-gemma-RETFound.git "$HOME/Fairvision-gemma-RETFound"
cd "$HOME/Fairvision-gemma-RETFound"
```

If that directory already contains this clone, inspect `git status --short` first,
then use `git pull --ff-only` when there are no local changes. Do not reset local
edits. All required source files, installers, job templates, tests and this guide
are included in Git; no encoded installer or laptop software is required.

## 2. Stokes: request CPU only if not already in a CPU allocation

The supplied Stokes inventory showed account ytian and partition normal.

```bash
srun --partition=normal --account=ytian --ntasks=1 --cpus-per-task=2 --mem=8G --time=01:00:00 --pty bash
```

Skip this request if already inside your allocated Stokes CPU shell.
Set paths in that shell:

```bash
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export VLM_CPU_VENV="$HOME/.venvs/fairvision-cpu-py311"
export VLM_OUTPUT_ROOT="$HOME/fairvision-vlm-results"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export PYTHON_BIN=/usr/bin/python3.11
mkdir -p "$VLM_OUTPUT_ROOT"
cd "$REPO_ROOT"
bash equi-agent/local_vlm/install_cpu.sh
```

This installs only NumPy, Pillow, scikit-learn and their dependencies.
No PyTorch, Transformers or GPU allocation is needed for CPU checks/reporting.

## 3. Stokes: prepare the patient list

```bash
"$VLM_CPU_VENV/bin/python" equi-agent/scripts/fairvision_local_vlm.py prepare-manifest --output "$VLM_OUTPUT_ROOT/cohort.csv"
```

This preserves the original 750 patients. A source-metadata-verified -1 glaucoma
label is recovered to 0 in the NEW copy and recorded in cohort.audit.json;
the original repository CSV is unchanged. Expected: glaucoma 124 positive/
126 negative, AMD 125/125, DR 125/125.

## 4. Stokes: validate all selected NPZs when data are visible there

```bash
find "$FAIRVISION_DATA_ROOT" -maxdepth 3 -type d
"$VLM_CPU_VENV/bin/python" equi-agent/scripts/fairvision_local_vlm.py check --manifest "$VLM_OUTPUT_ROOT/cohort.csv" --data-root "$FAIRVISION_DATA_ROOT"
```

Expect 750 readable cases and the class counts above. The check reads both image
arrays and hashes each NPZ. Missing files, corrupt ZIPs or wrong array dimensions
are blockers to inference. Training images are not needed.

If the dataset is accessible only on Newton, this image check must run on an
allocated Newton CPU node using the same lightweight installer, or the selected
data must be made accessible to Stokes. Do not reserve a GPU for a file check.

## 5. Newton: confirm resources and storage before installation/submission

```bash
hostname
sinfo -o "%P %a %l %G"
sacctmgr -n show assoc where user="$USER" format=Cluster,Account,Partition
command -v python3.11 python3.12
ls "$HOME/fairvision-models/medgemma-4b-it/config.json"
ls "$HOME/fairvision-models/gemma-3-4b-it/config.json"
ls "$HOME/Fairvision-gemma-RETFound/equi-agent/scripts/fairvision_local_vlm.py"
ls "$HOME/datasets/FairVision30K/raw"
```

Newton's actual GPU partition/account have not been supplied. Use its real
authorized values rather than Anvil or Stokes settings. Prefer a suitable GPU
with 40 GB VRAM. The job script requests one GPU and 48 GB host RAM.

Stokes and Newton are not assumed to share storage or compatible environments.
If any path above is missing, transfer the relevant source/data/model directories
using your institution's supported transfer service or SFTP. The original
MedGemma/Gemma weights have already been downloaded on Stokes, ~8.1 GB each;
reuse a verified copy instead of downloading again.

## 6. Newton: install the GPU environment without reserving a GPU

On an allocated Newton CPU node with network access, after the source files
are available, set:

```bash
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export VLM_VENV="$HOME/.venvs/fairvision-gpu-py311"
export VLM_OUTPUT_ROOT="$HOME/fairvision-vlm-results"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export FAIRVISION_MODEL_ROOT="$HOME/fairvision-models"
export PYTHON_BIN="$(command -v python3.11)"
mkdir -p "$VLM_OUTPUT_ROOT"
cd "$REPO_ROOT"
bash equi-agent/local_vlm/install.sh
```

If Newton lacks Python 3.11, select an available Python >=3.10 and set PYTHON_BIN
to its full executable path. GPU wheels are CUDA 12.4/Torch 2.6.0; verify driver
compatibility inside the GPU smoke allocation. Do not transfer a virtual
environment from an incompatible cluster.

The cohort CSV prepared on Stokes must be visible at the chosen output root.
Copy it there if storage is separate; do not select a new cohort. If needed, the
same prepare-manifest command can recreate the identical selected manifest from
the same unchanged repository inputs on Newton's CPU.

Record already downloaded model directories:

```bash
printf '%s\n' "$FAIRVISION_MODEL_ROOT/medgemma-4b-it" > "$VLM_OUTPUT_ROOT/medgemma_snapshot.txt"
printf '%s\n' "$FAIRVISION_MODEL_ROOT/gemma-3-4b-it" > "$VLM_OUTPUT_ROOT/gemma_snapshot.txt"
```

## 7. Newton: smoke tests, one model at a time

Once the inventory identifies authorized GPU settings, set the real values:

```bash
read -r -p "Newton GPU partition: " GPU_PARTITION
read -r -p "Newton account (blank if your site uses a default): " GPU_ACCOUNT
gpu_options=(--partition="$GPU_PARTITION")
if [[ -n "$GPU_ACCOUNT" ]]; then gpu_options+=(--account="$GPU_ACCOUNT"); fi
```

Submit MedGemma:

```bash
sbatch "${gpu_options[@]}" --cpus-per-task=2 --time=00:20:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/cluster_gpu.sbatch medgemma
```

Wait for completion, inspect the job log and six predicted rows, then submit Gemma:

```bash
sbatch "${gpu_options[@]}" --cpus-per-task=2 --time=00:20:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/cluster_gpu.sbatch gemma
```

Both must complete with zero errors. If the GPU lacks bfloat16 support, use
`VLM_DTYPE=float16` consistently in both smoke and full exports. If it cannot
fit unquantized 4B inference with the two images, choose a larger GPU; this baseline
does not silently quantize or reduce imaging inputs.

## 8. Newton: full evaluation

Use smoke elapsed_seconds × 750 plus loading/walltime headroom to select a
walltime within the cluster's limit. Four hours below is an example, not an
estimate guaranteed for Newton.

```bash
sbatch "${gpu_options[@]}" --cpus-per-task=2 --time=04:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/cluster_gpu.sbatch medgemma
```

After MedGemma completes, run:

```bash
sbatch "${gpu_options[@]}" --cpus-per-task=2 --time=04:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/cluster_gpu.sbatch gemma
```

Successful rows are saved after every patient; re-submit the same job to resume
walltime interruption. Smoke directories are separate from full runs.
Raw responses, errors, all attempts, software settings and cohort/data hashes
are retained. Never run two jobs against the same model output simultaneously.
Do not change model snapshots/prompts/data inside an existing run directory.

## 9. Stokes: generate the table on CPU

Ensure both complete result directories are visible on Stokes. If storage is
separate, transfer the two run directories containing predictions.csv,
run_config.json and cohort.json, preferably with attempts.jsonl as well.
Model weights and clinical images are not needed for reporting.

In an allocated Stokes CPU shell with the variables from step 2:

```bash
cd "$REPO_ROOT"
"$VLM_CPU_VENV/bin/python" equi-agent/scripts/report_fairvision_local_vlm.py --run-dirs "$VLM_OUTPUT_ROOT/medgemma" "$VLM_OUTPUT_ROOT/gemma" --out-dir "$VLM_OUTPUT_ROOT/table"
cat "$VLM_OUTPUT_ROOT/table/table1_local_models.md"
```

Outputs: table1_local_models.csv, .md, .tex; subgroup_metrics.csv;
report_config.json. The table has two new model rows per disease and the five
requested metrics. Incomplete/error cohorts produce no headline scores.

Worst-group F1 uses marginal race, ethnicity, sex and age groups, fixed labels
[0,1], zero_division=0, age bins <50/50–69/>=70. All observed nonmissing groups
qualify by default, including single-class groups; support is exported. The
paper's minimum support convention is not fully specified, so matching existing
worst-group numbers requires the same policy. Confidence words are not converted
to probabilities or AUROC. Numerical results come from your model predictions.

## Validation evidence

Ten targeted tests passed after adding Amd/Dr directory support. The lightweight
CPU installer ran in the development workspace and prepared the complete cohort.
Full pretrained GPU inference and scheduler submission remain to be validated
on Newton with your dataset and actual authorized resources.
