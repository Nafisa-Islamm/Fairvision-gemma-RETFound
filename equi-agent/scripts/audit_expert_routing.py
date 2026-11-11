"""CPU-only exploratory audit of frozen experts; never fits a policy on test labels."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from fairvision_local_vlm import MODEL_LABELS, TASKS, atomic_json, binary, digest, write_csv
from report_fairvision_local_vlm import UNKNOWN, load_run

AUDIT_METRICS = ("f1_macro", "worst_race_f1_macro", "race_f1_gap")


def scores(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    n = len(truth)
    if not n:
        return dict(n=0, positive=0, negative=0, tn=0, fp=0, fn=0, tp=0,
                    f1_macro=None, sensitivity=None, specificity=None, balanced_accuracy=None)
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    tp = int(((truth == 1) & (prediction == 1)).sum())
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(n=n, positive=tp + fn, negative=tn + fp, tn=tn, fp=fp, fn=fn, tp=tp,
                f1_macro=(f1_pos + f1_neg) / 2, sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)


def racial_summary(truth, prediction, races):
    groups = {race: scores(truth[races == race], prediction[races == race])
              for race in sorted(set(races) - UNKNOWN)}
    f1s = [row["f1_macro"] for row in groups.values()]
    sensitivities = [row["sensitivity"] for row in groups.values() if row["sensitivity"] is not None]
    fprs = [1 - row["specificity"] for row in groups.values() if row["specificity"] is not None]
    result = scores(truth, prediction)
    result.update(known_races=len(groups), unknown_race_cases=int(sum(race in UNKNOWN for race in races)),
                  worst_race_f1_macro=min(f1s) if f1s else None,
                  race_f1_gap=max(f1s) - min(f1s) if len(f1s) >= 2 else None,
                  race_sensitivity_gap=max(sensitivities) - min(sensitivities) if len(sensitivities) >= 2 else None,
                  race_false_positive_rate_gap=max(fprs) - min(fprs) if len(fprs) >= 2 else None)
    return result, groups


def align_runs(paths, allow_small_cohort=False):
    runs = [load_run(path) for path in paths]
    names = [config["model"] for config, _, _ in runs]
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Supply at least two distinct expert models")
    if len({config["cohort_sha256"] for config, _, _ in runs}) != 1:
        raise ValueError("Experts do not have identical cohort/data fingerprints")
    if len({digest(config["prompts"]) for config, _, _ in runs}) != 1 or len({config["oct_slices"] for config, _, _ in runs}) != 1:
        raise ValueError("Experts used different prompts or image selection")
    cohort = runs[0][1]
    keys = [(case["task"], case["case_id"]) for case in cohort]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cohort cases")
    if not allow_small_cohort and Counter(case["task"] for case in cohort) != Counter({task: 250 for task in TASKS}):
        raise ValueError("Expected 250 cases per disease; small fixtures require --allow-small-cohort")
    columns = {}
    for name, (_, _, rows) in zip(names, runs):
        indexed = {(row["task"], row["case_id"]): row for row in rows}
        if set(indexed) != set(keys) or any(row["status"] != "ok" for row in rows):
            raise ValueError(f"Missing or failed predictions for {name}; do not exclude patients")
        columns[name] = np.asarray([binary(indexed[key]["y_pred"]) for key in keys])
    return runs, cohort, columns


def audit(args):
    output = Path(args.out_dir)
    if output.exists():
        raise ValueError("Choose a NEW output directory; refusing to overwrite an audit")
    runs, cohort, all_predictions = align_runs(args.run_dirs, args.allow_small_cohort)
    tables, race_rows, intervals, differences, pair_rows, complementarity, selective = [], [], [], [], [], [], []
    experts = list(all_predictions)
    for task_index, task in enumerate(TASKS):
        selected = np.asarray([i for i, case in enumerate(cohort) if case["task"] == task], dtype=int)
        if not len(selected):
            continue
        cases = [cohort[i] for i in selected]
        truth = np.asarray([binary(case["y_true"]) for case in cases])
        races = np.asarray([case["race"] for case in cases])
        predictions = {name: values[selected] for name, values in all_predictions.items()}
        matrix = np.stack(list(predictions.values()))
        if len(experts) % 2:
            predictions["majority_vote"] = (matrix.sum(axis=0) > len(experts) // 2).astype(int)
        estimates = {}
        for method, prediction in predictions.items():
            estimates[method], groups = racial_summary(truth, prediction, races)
            tables.append(dict(task=task, method=method, **estimates[method]))
            race_rows.extend(dict(task=task, method=method, race=race,
                                  sparse_class_support=row["positive"] < args.support_warning or row["negative"] < args.support_warning,
                                  **row) for race, row in groups.items())
        for left, right in combinations(experts, 2):
            a, b = predictions[left] == truth, predictions[right] == truth
            pair_rows.append(dict(task=task, left=left, right=right, n=len(truth),
                                  both_correct=int((a & b).sum()), only_left_correct=int((a & ~b).sum()),
                                  only_right_correct=int((~a & b).sum()), both_wrong=int((~a & ~b).sum()),
                                  disagreements=int((predictions[left] != predictions[right]).sum())))
        correctness = matrix == truth[None, :]
        complementarity.append(dict(task=task, n=len(truth), all_experts_correct=int(correctness.all(axis=0).sum()),
                                    all_experts_wrong=int((~correctness.any(axis=0)).sum()),
                                    mixed_correctness=int((correctness.any(axis=0) & ~correctness.all(axis=0)).sum())))
        accepted = (matrix == matrix[0]).all(axis=0)
        for race in ["ALL", *sorted(set(races) - UNKNOWN)]:
            population = np.ones(len(truth), dtype=bool) if race == "ALL" else races == race
            keep = accepted & population
            selective.append(dict(task=task, policy="unanimous_else_defer", race=race,
                                  total_cases=int(population.sum()), accepted_cases=int(keep.sum()),
                                  deferred_cases=int((population & ~accepted).sum()), coverage=float(keep.sum() / population.sum()),
                                  **scores(truth[keep], matrix[0][keep])))
        # Same race x label resampling indices for every method: differences stay paired.
        strata = [np.asarray([i for i, (r, y) in enumerate(zip(races, truth)) if (r, y) == key])
                  for key in sorted(set(zip(races, truth)))]
        rng = np.random.default_rng(args.seed + task_index)
        samples = {name: {metric: [] for metric in AUDIT_METRICS} for name in predictions}
        for _ in range(args.bootstrap):
            indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
            for name, prediction in predictions.items():
                summary, _ = racial_summary(truth[indices], prediction[indices], races[indices])
                for metric in AUDIT_METRICS:
                    if summary[metric] is not None:
                        samples[name][metric].append(summary[metric])
        for name in predictions:
            for metric in AUDIT_METRICS:
                values = samples[name][metric]
                lower, upper = np.quantile(values, [.025, .975]).tolist() if values else (None, None)
                intervals.append(dict(task=task, method=name, metric=metric, estimate=estimates[name][metric],
                                      ci_lower=lower, ci_upper=upper, valid_bootstrap_replicates=len(values)))
        for left, right in combinations(predictions, 2):
            for metric in AUDIT_METRICS:
                a, b = samples[left][metric], samples[right][metric]
                values = np.asarray(a) - np.asarray(b) if a and b else []
                lower, upper = np.quantile(values, [.025, .975]).tolist() if len(values) else (None, None)
                estimate = estimates[left][metric] - estimates[right][metric] if len(values) else None
                differences.append(dict(task=task, left=left, right=right, metric=metric,
                                        difference_left_minus_right=estimate, ci_lower=lower, ci_upper=upper))
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("overall_and_race_gaps.csv", tables), ("race_metrics.csv", race_rows),
                           ("confidence_intervals.csv", intervals), ("paired_differences.csv", differences),
                           ("expert_disagreement.csv", pair_rows), ("expert_complementarity.csv", complementarity),
                           ("selective_coverage.csv", selective)):
        write_csv(output / filename, rows)
    atomic_json(output / "audit_config.json", dict(
        stage="exploratory_test_audit_no_training", experts=experts,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy=np.__version__,
        source_runs=[dict(path=str(Path(path).resolve()), fingerprint=digest(run[0]),
                          predictions_sha256=hashlib.sha256((Path(path) / "predictions.csv").read_bytes()).hexdigest(),
                          postprocessing=run[0].get("postprocessing")) for path, run in zip(args.run_dirs, runs)],
        cohort_sha256=runs[0][0]["cohort_sha256"], bootstrap=args.bootstrap, seed=args.seed,
        ci_method="paired race-by-label stratified case bootstrap; percentile 95% intervals",
        bootstrap_unit="case; assumes one independent case per patient within each disease",
        support_warning=args.support_warning, f1="macro; fixed labels [0,1]; undefined class F1=0",
        unknown_race_excluded_from_gaps=True, majority_vote="odd expert counts only; no ties",
        escalation="defer disagreements; accepted-case metrics do not measure clinician performance",
        limitations=["Conditional on observed race/class counts and selected balanced cohort.",
                     "Bootstrap cannot compensate for tiny groups; intervals may be degenerate with singleton strata.",
                     "No probability, calibration, AUROC, learned routing, cost saving or GRPO claim.",
                     "All experts were already queried; this is retrospective arbitration, not pre-inference routing.",
                     "No policy or demographic expert assignment is selected using test labels."]))
    lines = ["# Exploratory frozen-expert audit", "",
             "| Task | Method | Macro-F1 | Worst-race F1 | Race F1 gap |", "|---|---|---:|---:|---:|"]
    fmt = lambda value: "NA" if value is None else f"{value:.3f}"
    for row in tables:
        label = MODEL_LABELS.get(row["method"], row["method"])
        lines.append(f"| {row['task']} | {label} | {fmt(row['f1_macro'])} | {fmt(row['worst_race_f1_macro'])} | {fmt(row['race_f1_gap'])} |")
    lines.extend(["", "Worst-race F1 differs from worst-group F1 across all demographic attributes.",
                  "This audit does not train or validate a routing policy. Use separate training/validation predictions for learning.",
                  "Read audit_config.json for sparse-group, bootstrap and selective-coverage limitations."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved audit: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--support-warning", type=int, default=20)
    parser.add_argument("--allow-small-cohort", action="store_true", help="Synthetic tests only")
    args = parser.parse_args()
    if args.bootstrap < 2 or args.support_warning < 1:
        parser.error("bootstrap must be >=2 and support-warning >=1")
    audit(args)
