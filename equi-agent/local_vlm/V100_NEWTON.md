# Experimental V100 alternative to queued H100 inference

Use a **32 GB V100**, requested as `gpu:tesla_v100-pcie-32gb:1`. Do not silently
replace the existing BF16 experiments: V100 has no native BF16 support. This template
uses FP16 by default, writes to a separate result root, and starts in smoke mode.

| Expert | Existing runtime selected by the wrapper | Status |
|---|---|---|
| Gemma 3 4B | `~/.venvs/fairvision-gpu-py311` | Candidate for a 32 GB FP16 smoke test |
| MedGemma 4B | `~/.venvs/fairvision-gpu-py311` | Candidate for a 32 GB FP16 smoke test |
| Gemma 4 12B IT | `~/.venvs/fairvision-gemma4-py311` | Experimental; 32 GB fit and FP16 numerical behavior must be tested |
| Muse Glimmer 30B | Not supported by this single-GPU template | BF16/FP16 weights alone exceed 32 GB |

These runtimes use PyTorch 2.6 / CUDA 12.4. The wrapper does not upgrade environments,
install packages, quantize weights, or split models across GPUs. Requesting two V100s
does not automatically combine their memory: the current inference loader uses one GPU.
No pretrained V100 run has been performed by the assistant. FP16 may be numerically
unstable for some checkpoints; a valid-looking six-case run is not an accuracy study
or a guarantee that every longer case will succeed.

## 1. Update and configure separate training outputs

Run on Newton's login node, one line at a time. Existing H100 jobs are not cancelled:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export FAIRVISION_MANIFEST="$HOME/fairvision-routing-research/manifests-v1/training.csv"
export VLM_SPLIT=training
export VLM_OUTPUT_ROOT="$HOME/fairvision-routing-research/cache-v100-fp16/training"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$VLM_OUTPUT_ROOT"
printf '%s\n' "$HOME/fairvision-models/gemma-3-4b-it" > "$VLM_OUTPUT_ROOT/gemma_snapshot.txt"
printf '%s\n' "$HOME/fairvision-models/medgemma-4b-it" > "$VLM_OUTPUT_ROOT/medgemma_snapshot.txt"
printf '%s\n' "$HOME/fairvision-models/gemma-4-12B-it" > "$VLM_OUTPUT_ROOT/gemma4-it_snapshot.txt"
```

The wrapper selects the right existing environment for each model even if the login
shell's `VLM_VENV` still points to a different expert. Explicit overrides are
`V100_VENV` for the 4B experts and `V100_GEMMA4_VENV` for Gemma 4.

## 2. Test six cases per expert before replacing H100 jobs

```bash
V100_GEMMA_TEST=$(sbatch --parsable --account=ytian --partition=normal --export=ALL,SMOKE=1 equi-agent/local_vlm/v100_gpu.sbatch gemma)
echo "$V100_GEMMA_TEST"
V100_MEDGEMMA_TEST=$(sbatch --parsable --account=ytian --partition=normal --export=ALL,SMOKE=1 equi-agent/local_vlm/v100_gpu.sbatch medgemma)
echo "$V100_MEDGEMMA_TEST"
```

Optional, separate larger-model test; no memory-fit guarantee:

```bash
V100_GEMMA4_TEST=$(sbatch --parsable --account=ytian --partition=normal --time=01:00:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/v100_gpu.sbatch gemma4-it)
echo "$V100_GEMMA4_TEST"
```

Inspect the queue, logs and accounting. Logs use a different name from the H100 template:

```bash
squeue -u "$USER"
tail -n 100 "$REPO_ROOT/slurm-fairvision-v100-${V100_GEMMA_TEST}.out"
sacct -j "$V100_GEMMA_TEST,$V100_MEDGEMMA_TEST" --format=JobID,State,Elapsed,NodeList,ExitCode
```

Require six valid predictions, zero errors and exit `0:0` for each expert you move.
On NaNs, invalid JSON, CUDA kernel errors or out-of-memory, stop and inspect the actual
error. Do not reinterpret invalid outputs as negative diagnoses. A 4B FP32 experiment
is an optional separately tested fallback (`V100_DTYPE=float32`) with a NEW FP32
output root; it is slower and needs more memory. The wrapper rejects 12B FP32 on 32 GB.

## 3. Cancel only the H100 jobs whose V100 replacement passed

First inspect current states:

```bash
squeue -j 1200114,1200115,1200117
```

If the Gemma and MedGemma V100 tests passed and their corresponding H100 jobs are
still pending, cancellation commands for the previously reported IDs are:

```bash
scancel 1200114 1200115
```

Keep Gemma 4 job 1200117 unless its own alternative passed. If a job already started,
inspect its progress before discarding that allocation. Do not cancel unrelated jobs.

## 4. Full V100 generation after success

Keep the same V100 precision-specific configuration and submit the selected model
with `SMOKE=0` and an appropriate time allowance. For example:

```bash
V100_GEMMA_TRAIN=$(sbatch --parsable --account=ytian --partition=normal --time=2-00:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/v100_gpu.sbatch gemma)
echo "$V100_GEMMA_TRAIN"
```

This still selects 18,000 training cases; a different GPU does not change the sample
count. A shorter queue is not a faster GPU: V100 inference may be slower than H100.
Input preparation still uses CPU/I/O and prints progress. Resume requires unchanged
source/settings and the same model's output folder. No speed, memory or total-runtime
guarantee follows from the resource request.

## 5. Keep precision consistent for the GRPO study

For each expert moved to FP16, also generate its **validation and existing 750-case
test outputs at FP16** before building GRPO bundles. Do not train the router on FP16
expert outputs and evaluate that same expert using its earlier BF16 outputs. Individual
experts may use different precisions if each expert's protocol is consistent across
all splits; e.g. FP16 Gemma/MedGemma with BF16 Gemma 4 is explicitly recorded.

- Validation manifest: `~/fairvision-routing-research/manifests-v1/validation.csv`;
  `VLM_SPLIT=validation`, separate output `.../cache-v100-fp16/validation`.
- Test manifest: `~/fairvision-vlm-results-eager/cohort.csv`; `VLM_SPLIT=test`,
  separate output `.../cache-v100-fp16/test`.

Create snapshot pointers in each output root as above. Keep the original BF16 results.
The same patient cohort is reused for precision-controlled comparison; the test labels
remain excluded from training.

Router bundles/checkpoints now use format version 2. Per-expert checkpoint, precision
and recorded runtime/generation settings are checked across splits and at test time.
Regenerate old version-1 bundles with the current `prepare` command into new files;
this alone does not require rerunning expert inference if precision/protocol is unchanged.
Preserve old router artifacts; a new router is needed for new precision experiments.
