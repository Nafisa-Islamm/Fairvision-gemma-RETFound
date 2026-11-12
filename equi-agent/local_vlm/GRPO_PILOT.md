# Smaller GRPO pilot: preserve the existing 750 test cases

The first 18,000-case CPU image/hash check timed out after 30 minutes. A subsequent
six-case training check passed. Do not launch full expert-cache jobs just to test the
new routing pipeline. Start with a smaller, explicitly stratified pilot instead.

## Selection

Use seed 2026 and sample without replacement from each disease's predefined splits:

- Training: at most 30 cases per race x binary label cell.
- Validation: at most 20 per cell; if the source has fewer, retain those cases once.
- Test: retain the existing 750 evaluation cases unchanged.

With the repository's current metadata this gives:

| Disease | Training | Validation |
|---|---:|---:|
| Glaucoma | 180 | 120 |
| AMD | 180 | 98 |
| DR | 180 | 102 |
| Total | 540 | 320 |

Each training race/disease has 30 positives and 30 negatives, meeting the trainer's
default minimum of 20 positive cases per group. Validation has only two Asian DR
positives, 11 Asian AMD positives and seven Black AMD positives. These counts cannot
be increased by sampling without replacement. The code records below-target cells
instead of duplicating patients or borrowing from training/test splits.

This is a pipeline/learning pilot, not evidence of reliable population fairness.
Stratified caps change demographic/class proportions; validation F1 and checkpoint
selection are conditional on that pilot distribution. Some validation fairness terms
will be unsupported at the default threshold. Keep the support warnings, and do not
claim fairness certification from this experiment. Verify actual patient identities
and review the DR endpoint/prompt alignment described in GRPO_NEWTON.md before new
expert inference.

## Create separate pilot manifests: CPU only

Run one line at a time on the Newton login shell:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/make_fairroute_manifests.py --out-dir "$HOME/fairvision-routing-research/manifests-pilot-v1" --training-per-cell 30 --validation-per-cell 20
```

An existing destination is refused. Keep previous manifests/results and use a new
pilot name for a deliberately changed seed/design. The original full `manifests-v1`
directory is not modified. Confirm counts and support in:

```bash
cat "$HOME/fairvision-routing-research/manifests-pilot-v1/manifest_config.json"
```

## Check pilot files with visible progress

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/fairvision_local_vlm.py check --manifest "$HOME/fairvision-routing-research/manifests-pilot-v1/training.csv" --split training --data-root "$FAIRVISION_DATA_ROOT" --progress-every 50
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/fairvision_local_vlm.py check --manifest "$HOME/fairvision-routing-research/manifests-pilot-v1/validation.csv" --split validation --data-root "$FAIRVISION_DATA_ROOT" --progress-every 50
```

This validates all selected pilot images/hashes. It is not a trained router, and the
earlier six-case success did not validate all these files. Optional progress messages
do not change cohort identities or hashes; omitting the flag retains JSON-only output.

## Later stages

After successful data checks and the protocol/identity review, follow GRPO_NEWTON.md
with these separate pilot paths throughout:

- Manifests: `$HOME/fairvision-routing-research/manifests-pilot-v1/{training,validation}.csv`.
- New expert caches: `$HOME/fairvision-routing-research/pilot-cache/{training,validation}`.
- Bundles: `$HOME/fairvision-routing-research/pilot-{train,validation,test}.json`.
- Router output: `$HOME/fairvision-routing-research/grpo-pilot-seed2026`.

Freeze expert versions, prompts, preprocessing and code before generating caches.
There are 860 new cases per expert, so three experts need 2,580 new predictions;
two experts need 1,720. Existing test predictions are reusable and are never training
targets. The large models still require GPU inference, while the small GRPO policy
can train on CPU from their frozen outputs. Do not use smaller sample counts as a
guarantee of runtime, diagnostic benefit, fairness improvement or reduced clinical cost.
