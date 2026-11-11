# Fairness-aware external-expert routing: extension of the existing FairVision code

This research builds on the completed Gemma 3 4B, MedGemma 4B and Gemma 4 12B IT experiments. Muse can be added only after it completes the same cohort without unresolved errors. Model generation remains unchanged; no paid API, expert retraining or training on the 750 evaluated test cases is introduced by this first stage.

**GRPO implementation added:** see [the runnable GRPO/Newton guide](GRPO_NEWTON.md).
The following audit remains the exploratory starting point. The new small-policy GRPO
trainer, explicit split support and auxiliary fairness loss implement the next stage,
but still require real, separate training and validation expert caches.

The supplied proposal is a research hypothesis. Its named MoE-GRPO, FairGE, Fair-FedMOE and EQPO references have not been independently verified here, and the combination is not claimed to be novel or effective. Compare the eventual method with the paper's actual reliability-based arbitration and simpler learned baselines before making those claims.

## Implemented now: CPU-only frozen-expert audit

`../scripts/audit_expert_routing.py` reads existing run directories through the same provenance/formatting-recovery validator used for paper tables. It rejects mismatched patients/images/prompts, duplicate expert identities, missing cases and failed predictions. Originals are read-only, output directories must be new, and metadata records all input hashes, repaired-result provenance, the audit code hash and bootstrap settings.

Outputs:

| File | Purpose |
|---|---|
| `summary.md` | Overall macro-F1, worst-race macro-F1 and race F1 gap per expert and fixed majority vote |
| `overall_and_race_gaps.csv` | Confusion counts, sensitivity/specificity and racial F1, sensitivity and false-positive-rate gaps |
| `race_metrics.csv` | Every observed known race, counts and class-support warnings |
| `confidence_intervals.csv` | Exploratory 95% intervals for macro-F1, worst-race F1 and race F1 gap |
| `paired_differences.csv` | Paired method differences with 95% intervals; left minus right |
| `expert_disagreement.csv` | Both correct, only one correct, both wrong, and disagreement counts |
| `expert_complementarity.csv` | Cases where every expert is right, every expert is wrong, or correctness differs |
| `selective_coverage.csv` | Coverage and accepted-case metrics for the fixed rule: unanimous diagnosis, otherwise defer |
| `audit_config.json` | Full provenance, resampling settings and limitations |

Majority vote is evaluated only for an odd number of experts; the tool does not invent a tie-breaking rule for even expert sets. Unanimous/defer is reported separately with coverage per race; deferred cases are not counted as correct or negative. Neither consensus nor a confidence statement proves clinical correctness.

### Interpretation limits

- Worst-race F1 uses race alone. It differs from the paper table's worst-group F1 across race, ethnicity, sex and age.
- Race F1 gap is max racial F1 minus min racial F1. Smaller is not automatically better if stronger groups deteriorate. Examine weakest-group improvement, overall F1, sensitivity and specificity together.
- Gaps need at least two known groups with a defined metric. Unknown race cases remain in overall scores but are excluded from racial gaps. Single-class subgroups retain the original fixed-label macro-F1 convention; undefined sensitivity/specificity are never silently set to zero.
- Bootstrap resampling is stratified by observed race and diagnosis within each disease, conditional on the cohort's observed race/class counts. Every method uses the same sampled cases. This assumes one independent case per patient per disease; use a verified patient-level cluster key before analyzing repeated eyes/visits. It does not measure variation in model training or generalize the balanced subset to population prevalence.
- Groups with fewer than 20 positives or 20 negatives are flagged, not silently removed. Intervals may be unstable or degenerate for tiny/singleton strata. These exploratory intervals are not multiple-comparison-adjusted significance tests.
- Correctness differences are diagnostic evidence of complementary errors, not an implementable oracle policy. No "best model per race" is selected using test labels.
- All experts have already run. This stage evaluates retrospective arbitration and does not demonstrate fewer expert calls, runtime savings, learned routing, GRPO, calibrated probabilities or AUROC.
- Two Gemma 4 formatting repairs remain disclosed in provenance. They did not change generated diagnoses.

## Run on Newton using the existing CPU environment

Run one line at a time after reconnecting. No GPU allocation or new dependency installation is needed:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 "$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/audit_expert_routing.py --run-dirs "$HOME/fairvision-vlm-results-eager/gemma" "$HOME/fairvision-vlm-results-eager/medgemma" "$HOME/fairvision-vlm-results-gemma4-it/gemma4-it-recovered" --out-dir "$HOME/fairvision-routing-research/audit-001" --bootstrap 2000
cat "$HOME/fairvision-routing-research/audit-001/summary.md"
```

If `audit-001` exists, keep it and choose `audit-002`. Missing files or incompatible provenance must be investigated; do not alter cohort hashes to bypass checks. The assistant has validated this tool on synthetic fixtures, not run it against the user's Newton predictions.

## Next stage: data preparation for learned routing

1. Inspect the dataset's real patient identifiers and predefined train/validation/test metadata. Verify patient/eye separation before producing prediction caches. Filenames reused across diseases are not sufficient evidence of a shared patient identity.
2. Generate frozen-expert predictions on training and validation data with the same diagnosis protocol. The current standalone inference script intentionally accepts test manifests only; a split-aware extension must preserve that default and explicitly record any training/validation mode. Do not relabel the 750 test cases as training data or remove the safeguard blindly.
3. Freeze expert versions, image settings, prompts and recovery rules. Record class and demographic support by split. Use train labels for fitting and validation labels for policy/hyperparameter selection.
4. Keep the existing 750-case evaluation exploratory: its outcomes have already informed the research idea. Predefine an additional untouched evaluation population if making confirmatory claims, with sensitivity analysis at natural prevalence.

## Learned policies to compare before claiming an RL contribution

Use the same frozen experts and patient splits for:

1. Each individual expert (report all; select any preferred expert on validation only).
2. Fixed majority vote and unanimous/defer.
3. Validation-selected global expert per disease.
4. Validation-selected expert per race with a predefined sparse-group fallback. Treat it as a baseline, not a preferred demographic assignment.
5. RetinAgent-style validation reliability arbitration, with shrinkage for sparse groups. Existing categorical confidences do not supply AUROC/ECE probabilities.
6. A supervised case-reliability router, plus a race-blind input ablation.
7. Contextual-bandit / GRPO routing without fairness penalties.
8. Fairness-constrained routing and a separately evaluated escalation extension.

A router using all experts' predictions and confidences is an **arbitrator**: all inference costs have already been incurred. A pre-inference router can use task, permitted metadata and image-derived features available before calling experts. A sequential router can request another expert after observing the first. Do not give it unqueried expert outputs or the true diagnosis as input. Cache-based experiments simulate such calls; actual speed savings require later timing real execution.

## Fairness objective and GRPO scope

Start with a small supervised/contextual-bandit router, because selecting among three experts does not inherently require a language model. Estimate diagnosis loss and racial false-negative/false-positive gaps using training groups, and choose fairness strength on validation data. Compare race-aware and race-blind inputs; race-blind features can still contain demographic proxies.

An illustrative objective is diagnostic error plus a query-cost penalty, subject to prespecified bounds on racial FNR/FPR gaps and minimum overall/per-race accepted coverage. Compute fairness constraints over adequately supported groups using training batches or running estimates, not a single patient's reward. Do not optimize away disease detection by predicting everyone positive or negative, or optimize away workload by escalating everyone. Report accuracy/coverage tradeoffs, not an invented perfect clinician outcome.

GRPO would sample several routing/action sequences for the **same training case**, calculate outcome/cost rewards and group-relative advantages, then update a policy with the appropriate clipped objective/reference regularization. Here "group" means the sampled candidate responses, **not a racial group**. Demographic fairness needs a separate population-level constraint/penalty mechanism. This stage is not implemented in the audit script, and should demonstrate benefit over simpler supervised/bandit routing before an expensive LLM-router run.

Avoid claiming that post hoc racial F1 differences alone establish bias, that demographic metadata is anatomical evidence, or that smaller gaps ensure improved patient outcomes. The eventual study is retrospective and requires uncertainty estimates and external/clinical validation beyond this prototype.
