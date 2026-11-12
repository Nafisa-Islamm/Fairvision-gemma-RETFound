# FairRoute: actual GRPO routing on the existing FairVision code

**Dataset-size options:** this guide covers the full-source training/validation
workflow. A [540-training / 320-validation pilot](GRPO_PILOT.md) is also available
when a smaller inference budget is preferred. Existing test cases are not used for
training in either workflow.

This implementation adapts the supplied example into a trainable **small neural routing policy**. Frozen Gemma/MedGemma experts are not fine-tuned. It is categorical, one-step contextual-bandit GRPO with a separate differentiable fairness regularizer, not token-level GRPO training of a language model or a complete reproduction of RetinAgent. It has been tested on explicitly synthetic fixtures; no real FairVision routing model has yet been trained by the assistant.

**Precision compatibility:** the current bundle/checkpoint format is version 2 and
records per-expert checkpoint/precision/runtime settings. Existing version-1 bundles
must be regenerated using `prepare` into new files. FP16 and BF16 outputs for the same
expert cannot be mixed across training/validation/test. See [the V100 workflow](V100_NEWTON.md)
before replacing queued H100 jobs; preserve existing BF16 experiment outputs.

The full-data workflow is appropriate when its inference budget is available: it uses
18,000 training and 3,000 validation cases per expert, keeping the existing 750-case
test cohort separate. GPU batch jobs now print preparation progress every 100 cases
(`PREP_PROGRESS_EVERY` can change this interval). Initial NPZ image/hash preparation
still uses CPU and disk I/O before model loading; allocating a GPU does not accelerate
that stage. Each model then performs its diagnostic inference on the requested GPU.

## What changed from the example

| Example issue | Implemented behavior |
|---|---|
| Assumed expert probabilities and 768D embeddings | Reads actual binary labels and categorical confidence; optional real case-keyed embeddings; no probability/embedding fabrication |
| Broadcast fairness reward cancels in within-case normalization | Expected racial FNR hinge loss is applied separately with its own differentiable gradient |
| Clipping setting unused | Frozen behavior log-probabilities, sampled action groups, detached normalized advantages, clipped importance-ratio objective and multiple inner updates |
| No reference policy | Frozen initial router with exact categorical KL regularization |
| Entropy computed without gradients | Entropy term participates in the gradient |
| Missing racial groups treated as zero FNR | Unsupported groups are excluded from that estimate and their support is recorded; no unsupported fairness guarantee |
| Unknown race mapped to White | Unknown category is explicit; race-blind policy input is default |
| Test data could be randomly split for learning | Predefined training/validation roles, exact patient/case/image overlap checks, train-only feature normalization, validation-only checkpoint/baseline selection |
| All expert hints used while claiming cheaper calls | `arbitration` and `prequery` modes are distinct; selected-expert cost penalties are rejected in arbitration mode |
| Disease-pooled F1 | Metrics are reported for each disease and each race, with fixed binary macro-F1 |

**GRPO's group means several sampled expert actions for the same case, not racial groups.** The fairness extension estimates expected FNR over supported training race-by-disease groups. Its squared hinge penalizes gaps beyond the configured threshold. This is a soft regularizer, not a hard fairness constraint or a guarantee for greedy test decisions. Validation selection uses mean disease macro-F1 minus the supported FNR-gap penalty, never test labels.

## Files

- `../scripts/fairroute_grpo.py`: prepare bundles, GRPO/supervised training, frozen held-out evaluation.
- `../scripts/fairroute_data.py`: real-output adapter, feature construction and leakage validation.
- `../scripts/make_fairroute_manifests.py`: explicit training/validation manifests from predefined source summaries.
- `router_cpu.sbatch`: CPU-only routing training/evaluation using the already installed Torch environment.
- `../scripts/audit_expert_routing.py`: exploratory audit, bootstrap intervals and nonlearned baselines for existing test outputs.
- `../tests/test_fairroute_grpo.py`: numerical clipping/advantage tests, nonzero fairness gradients, feature/split guards, real training/checkpoint/evaluation tests for both algorithms.

## Data required: existing test results are insufficient for training

Existing test expert directories remain:

```text
~/fairvision-vlm-results-eager/gemma
~/fairvision-vlm-results-eager/medgemma
~/fairvision-vlm-results-gemma4-it/gemma4-it-recovered
```

You also need the same experts' outputs on **training and validation** patients. The source repository summaries contain 6,000 training, 1,000 validation and 3,000 test records per disease. Generating new expert caches, not training the small router, is the main GPU expense. Full train/validation caches require 21,000 predictions per expert: 42,000 for two experts or 63,000 for three. No such jobs have been submitted by the assistant.

Patient identity is mandatory. Provide a real `task,case_id,patient_id` CSV, or, only after checking dataset documentation, explicitly assert that every task/case represents an independent patient across diseases and splits. The flag does not verify that assertion. Different image IDs for the same patient cannot be detected without a true patient map. Matching patient IDs, task/case keys and NPZ hashes across splits are rejected.

Because the existing 750 test cases have already informed this research design, treat their further comparisons as exploratory. Reserve additional untouched patients for confirmatory evaluation. Do not split these same 750 patients into a new training/validation set.

## 1. Update and prepare manifests: CPU only

Run one line at a time on Newton:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/make_fairroute_manifests.py --out-dir "$HOME/fairvision-routing-research/manifests-v1"
cat "$HOME/fairvision-routing-research/manifests-v1/manifest_config.json"
```

This writes metadata CSVs only. No models or training jobs run. The configuration records race/class support and source hashes. An optional `--max-per-task` creates a uniformly sampled pilot within each source split, without using expert outputs; small pilots may have too little positive-group support for meaningful fairness training.

Labels are the same as the existing cohort: glaucoma yes/no; AMD any non-normal stage; DR **vision-threatening** versus non-vision-threatening. These source mappings were checked against all AMD/DR labels in the existing 250-case manifests. They must not be silently changed to a different clinical endpoint.

**Clinical-protocol review before new inference:** the inherited DR prompt asks broadly about DR, whereas the repository's positive label is vision-threatening DR. Preserve this distinction in interpretation. The framework keeps the existing prompt for comparability, but its alignment with the intended clinical endpoint should be reviewed before spending GPU time on new caches. If prompts are revised, predefine the revision and regenerate every split/expert under that protocol; do not combine new-prompt training caches with old-prompt test outputs. The bundle checks reject prompt fingerprint mismatches.

The inference/check CLI now accepts explicit `--split training` or `--split validation`; the default remains test. Non-test manifests need a matching split column. The resolver checks disease/split casing and rejects paths pointing to a different split.

## 2. Generate frozen expert caches: GPU required, budget first

Check that the actual NPZ files exist before committing GPU time:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/fairvision_local_vlm.py check --manifest "$HOME/fairvision-routing-research/manifests-v1/training.csv" --split training --data-root "$HOME/datasets/FairVision30K/raw"
```

For training caches, set:

```bash
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export FAIRVISION_MANIFEST="$HOME/fairvision-routing-research/manifests-v1/training.csv"
export VLM_SPLIT=training
export VLM_OUTPUT_ROOT="$HOME/fairvision-routing-research/cache/training"
mkdir -p "$VLM_OUTPUT_ROOT"
printf '%s\n' "$HOME/fairvision-models/gemma-3-4b-it" > "$VLM_OUTPUT_ROOT/gemma_snapshot.txt"
printf '%s\n' "$HOME/fairvision-models/medgemma-4b-it" > "$VLM_OUTPUT_ROOT/medgemma_snapshot.txt"
printf '%s\n' "$HOME/fairvision-models/gemma-4-12B-it" > "$VLM_OUTPUT_ROOT/gemma4-it_snapshot.txt"
export VLM_VENV="$HOME/.venvs/fairvision-gpu-py311"
```

Use `cluster_gpu.sbatch gemma` or `medgemma` in a six-case smoke allocation (`SMOKE=1`) first. After success, the following starts **one** full cache-generation job; it does not train a router:

```bash
TRAIN_GEMMA_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=64G --time=2-00:00:00 --export=ALL,SMOKE=0 equi-agent/local_vlm/cluster_gpu.sbatch gemma)
echo "$TRAIN_GEMMA_JOB_ID"
```

Repeat for MedGemma using the same original GPU environment. For Gemma 4 IT, first set `VLM_VENV` to `$HOME/.venvs/fairvision-gemma4-py311`, then select `gemma4-it`. Confirm cohort counts, completion and any parsing errors. Freeze code and dependencies during a run; resuming requires matching provenance. Two days is a time allowance, not a runtime guarantee.

Repeat the setup for validation with `FAIRVISION_MANIFEST=.../validation.csv`, `VLM_SPLIT=validation`, and a separate `VLM_OUTPUT_ROOT=$HOME/fairvision-routing-research/cache/validation`; recreate its snapshot files. Do not reuse training output folders. These cache jobs may be expensive: pilot and verify support before choosing the full experiment size.

## 3. Convert caches into router bundles: no GPU

Use the existing environment that already contains Torch; naming it `fairvision-gpu-py311` does not allocate a GPU. The following commands assume the one-independent-case-per-patient assertion has been checked. Otherwise replace the assertion flag with `--patient-map` pointing to your verified mapping CSV. Optional real embeddings can be supplied as JSON records containing `task`, `case_id`, and a finite fixed-length `embedding` vector; no embedding is invented if omitted.

```bash
export VLM_ROUTER_PYTHON="$HOME/.venvs/fairvision-gpu-py311/bin/python"
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:15:00 "$VLM_ROUTER_PYTHON" equi-agent/scripts/fairroute_grpo.py prepare --run-dirs "$HOME/fairvision-routing-research/cache/training/gemma" "$HOME/fairvision-routing-research/cache/training/medgemma" "$HOME/fairvision-routing-research/cache/training/gemma4-it" --split train --out-file "$HOME/fairvision-routing-research/train.json" --confirm-one-case-per-patient
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:15:00 "$VLM_ROUTER_PYTHON" equi-agent/scripts/fairroute_grpo.py prepare --run-dirs "$HOME/fairvision-routing-research/cache/validation/gemma" "$HOME/fairvision-routing-research/cache/validation/medgemma" "$HOME/fairvision-routing-research/cache/validation/gemma4-it" --split validation --out-file "$HOME/fairvision-routing-research/validation.json" --confirm-one-case-per-patient
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:15:00 "$VLM_ROUTER_PYTHON" equi-agent/scripts/fairroute_grpo.py prepare --run-dirs "$HOME/fairvision-vlm-results-eager/gemma" "$HOME/fairvision-vlm-results-eager/medgemma" "$HOME/fairvision-vlm-results-gemma4-it/gemma4-it-recovered" --split test --out-file "$HOME/fairvision-routing-research/test.json" --confirm-one-case-per-patient
```

To start with two experts, omit the Gemma 4 directory from **all three** bundle commands. The expert order must match. Source test caches cannot be relabeled as training/validation. Missing/failed predictions, inconsistent embeddings and incompatible prompts are rejected. Use the same formatting-recovery rule across experts and disclose repairs.

## 4. Train real GRPO routing parameters: CPU is sufficient for this small network

```bash
GRPO_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --export=ALL equi-agent/local_vlm/router_cpu.sbatch train --train "$HOME/fairvision-routing-research/train.json" --validation "$HOME/fairvision-routing-research/validation.json" --out-dir "$HOME/fairvision-routing-research/grpo-fair-seed2026" --algorithm grpo --mode arbitration --fairness-weight 0.5 --fnr-gap-threshold 0.15 --seed 2026)
echo "$GRPO_JOB_ID"
```

This uses default group size 8, four clipped inner updates, a frozen reference, entropy with gradients, a small 64-hidden-unit network and train-only scaling. The race-blind default omits race from policy inputs while race is still used in training fairness estimates. `--race-aware` is a separate ablation. With no embeddings, arbitration is conditioned on metadata and the experts' image-derived diagnoses/confidence, not a new learned image encoder.

Training loss = clipped GRPO policy loss + reference KL penalty - entropy bonus + fairness-weight × expected-FNR-gap hinge². Task/label balancing weights are applied outside group normalization, so they do not cancel. The reference and behavior policies have distinct roles. A zero-advantage fraction is logged because identical expert outcomes cannot supply a relative GRPO signal.

Only adequately supported training race/disease groups enter the fairness loss (default at least 20 positives and at least two races for a disease). Support is saved; the run refuses to pretend to optimize fairness if no disease is supported. It may leave some diseases unsupported. More samples, improved experts, or an unconstrained pilot may be necessary. Full-model agreement on wrong diagnoses cannot be repaired by routing alone.

For required baselines, submit fresh output directories with:

- Unconstrained GRPO: same command, `--fairness-weight 0`.
- Supervised full-information policy: `--algorithm supervised --fairness-weight 0`.
- Supervised plus fairness: `--algorithm supervised --fairness-weight 0.5`.
- Race-aware/race-blind ablations, and multiple prespecified seeds.

Both algorithms use the same number of inner optimizer updates for a given setting.
The supervised baseline directly optimizes expected utility from all frozen expert
outcomes; it is not fine-tuning any diagnostic expert. Logged group-advantage diagnostics
are used by GRPO only.

Use validation data to choose hyperparameters; never select the experiment using the 750 test outcomes. Default clinical reward bonuses/penalties are hypotheses, not validated clinical utilities. Report sensitivity analyses and accuracy/fairness tradeoffs.

For a genuine pre-query variant, select `--mode prequery`: expert predictions/confidence disappear from policy features. Optional embeddings must be available before any expert call. Without embeddings this is a metadata-only routing baseline. Measured expert costs can then be supplied with `--expert-costs` in expert order and `--cost-weight`; there are no fabricated default speed estimates. In arbitration mode cost-weight must remain zero because all experts were already queried.

## 5. Evaluate the frozen router on test cases: CPU only

After the training job completes successfully:

```bash
EVAL_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --export=ALL equi-agent/local_vlm/router_cpu.sbatch evaluate --checkpoint "$HOME/fairvision-routing-research/grpo-fair-seed2026/best_router.pt" --data "$HOME/fairvision-routing-research/test.json" --out-dir "$HOME/fairvision-routing-research/grpo-fair-seed2026-test" --verification disagreement)
echo "$EVAL_JOB_ID"
```

Use `squeue -u "$USER"`, then `sacct -j "$EVAL_JOB_ID" --format=JobID,State,Elapsed,ExitCode`. Job logs are `slurm-fairroute-grpo-JOBNUMBER.out` in the repository; use the printed actual job number, not a literal placeholder.

Outputs include per-disease `metrics.csv`, `race_metrics.csv`, per-case `routing_decisions.csv`, and evaluation provenance. Baselines include each expert, odd-count majority vote, validation-selected global expert, validation-selected race expert, and validation-shrunken race reliability. The latter is an explicit simple baseline, not a reproduction of the full RetinAgent algorithm, whose probability-based reliability terms are unavailable here.

The optional `--verification disagreement` is a deterministic defer recommendation using all cached experts, allowed only in arbitration mode. It writes `verification_decisions.csv` and conditional `selective_metrics.csv` with per-race coverage. It is not an LLM verification agent, and neither a correct clinician diagnosis nor a benefit to clinical workload is assumed for deferred cases. Expert-selection probability is not disease probability.

Patient/case/image overlap with training or validation is rejected at test time. Test labels are used only for metrics. The router checkpoint, source hashes, seed and validation-selected epoch remain frozen. No real FairVision performance gain, fairness improvement, training runtime or novelty claim is established by synthetic tests.
