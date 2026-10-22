# Gemma 4 12B instruction-tuned: Newton recovery

The first downloaded `google/gemma-4-12B` processor had no chat template. The official Transformers Gemma 4 Unified usage example (inspected at commit `f348d3dcfa9317bed603e01b8ca22b11fe59615c`, `docs/source/en/model_doc/gemma4_unified.md`) uses `google/gemma-4-12B-it`. Use its native processor/template for instruction-based diagnosis. This is a different checkpoint and is saved as `gemma4-it` in provenance and reports. Do not relabel base weights as instruction tuned. No change to the already-installed Gemma 4 environment is needed.

Run the following on the Newton login shell, one line at a time:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export VLM_VENV="$HOME/.venvs/fairvision-gemma4-py311"
export VLM_CPU_VENV="$HOME/.venvs/fairvision-cpu-py311"
export VLM_OUTPUT_ROOT="$HOME/fairvision-vlm-results-gemma4-it"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
mkdir -p "$VLM_OUTPUT_ROOT"
cp -n "$HOME/fairvision-vlm-results-eager/cohort.csv" "$VLM_OUTPUT_ROOT/cohort.csv"
```

Check filesystem quota before downloading another checkpoint. Keep the existing base download until the intended replacement has been verified; removing files is unnecessary for this recovery.

## Verify the processor before downloading weights

Only download processor/configuration files first; this uses no GPU:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00 "$VLM_VENV/bin/hf" download google/gemma-4-12B-it --local-dir "$HOME/fairvision-models/gemma-4-12B-it" --include '*.json' '*.jinja' '*.model' '*.txt'
"$VLM_VENV/bin/python" -c 'import os; from transformers import AutoConfig, AutoProcessor; p=os.path.expanduser("~/fairvision-models/gemma-4-12B-it"); c=AutoConfig.from_pretrained(p, local_files_only=True); processor=AutoProcessor.from_pretrained(p, local_files_only=True); print("Architecture:", c.model_type); print("Chat template present:", bool(processor.chat_template)); assert c.model_type == "gemma4_unified" and processor.chat_template, "STOP: architecture/template check failed"'
```

Expected: `Architecture: gemma4_unified` and `Chat template present: True`, with no traceback. If files, access or the template are missing, stop and inspect the actual error before allocating a GPU. A repository name in documentation is not a substitute for this runtime check.

## Download the remaining files

After the check passes:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=8G --time=02:00:00 "$VLM_VENV/bin/hf" download google/gemma-4-12B-it --local-dir "$HOME/fairvision-models/gemma-4-12B-it"
printf '%s\n' "$HOME/fairvision-models/gemma-4-12B-it" > "$VLM_OUTPUT_ROOT/gemma4-it_snapshot.txt"
```

The full download command uses already-downloaded metadata. Do not reuse the base-model snapshot file. Do not reinstall or upgrade the separate environment solely for this missing-template error.

## Six-case smoke test

```bash
GEMMA4_IT_TEST_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=64G --time=00:30:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/cluster_gpu.sbatch gemma4-it)
echo "$GEMMA4_IT_TEST_JOB_ID"
squeue -u "$USER"
tail -n 100 "$REPO_ROOT/slurm-fairvision-vlm-${GEMMA4_IT_TEST_JOB_ID}.out"
sacct -j "$GEMMA4_IT_TEST_JOB_ID" --format=JobID,State,Elapsed,NodeList,ExitCode
```

Wait for a log to exist after the job starts. Stop on any submission error. Require `Complete: 6/6 rows; 0 errors`, `COMPLETED`, and exit `0:0` before proceeding. No pretrained IT-checkpoint GPU validation has been performed by the assistant.

## Full evaluation and CPU reporting

Only after successful smoke testing:

```bash
GEMMA4_IT_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=64G --time=06:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/cluster_gpu.sbatch gemma4-it)
echo "$GEMMA4_IT_JOB_ID"
```

Six hours is an allowance, not a measured estimate. After all 750 cases complete with zero errors:

```bash
export VLM_REPORT_MODELS=gemma4-it
GEMMA4_IT_REPORT_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --cpus-per-task=1 --mem=4G --time=00:10:00 --export=ALL equi-agent/local_vlm/cluster_cpu.sbatch report)
echo "$GEMMA4_IT_REPORT_JOB_ID"
```

Check that reporting completed with exit `0:0`, then read:

```bash
cat "$VLM_OUTPUT_ROOT/table/table1_local_models.md"
```

## Recover unquoted confidence formatting without rerunning inference

Observed in the first 750-case IT run: 748 valid predictions, two JSON errors where
the complete response used `"confidence": high` instead of `"confidence": "high"`.
The diagnoses were already explicit integers. Do not edit diagnoses, discard patients,
change ground truth, or resubmit all cases. The following narrowly defined recovery
changes only that confidence formatting at a JSON parser error, validates the resulting
response with the original parser, and saves a NEW derived run. All raw responses and
the entire source directory remain unchanged. Other kinds of errors remain errors.

After updating the code, use CPU resources:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/recover_fairvision_json.py --run-dir "$HOME/fairvision-vlm-results-gemma4-it/gemma4-it" --out-dir "$HOME/fairvision-vlm-results-gemma4-it/gemma4-it-recovered"
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/report_fairvision_local_vlm.py --run-dirs "$HOME/fairvision-vlm-results-gemma4-it/gemma4-it-recovered" --out-dir "$HOME/fairvision-vlm-results-gemma4-it/table-recovered"
cat "$HOME/fairvision-vlm-results-gemma4-it/table-recovered/table1_local_models.md"
```

Require `Valid predictions: 750/750; remaining errors: 0`. An existing destination is
refused; do not delete source results or overwrite the original CSV. The recovery audit
records each repaired response and original error, source hashes and a new run fingerprint.
The reporter verifies the audit checksum and labels formatting recovery in the Markdown
and report metadata. Report both the initial 2/750 formatting-failure rate and the use of
this recovery rule; for fair comparisons, use the same rule for every evaluated model.
This operation does not generate new model predictions or validate clinical correctness.
