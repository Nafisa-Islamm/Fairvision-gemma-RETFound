"""Produce paper-style macro-F1 tables; never omit failed or missing patients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

from fairvision_local_vlm import TASKS, atomic_json, binary, digest, read_csv, write_csv

ATTRIBUTES = ("race", "ethnicity", "sex_gender", "age_group")
UNKNOWN = {"", "unknown", "nan", "none", "not available", "missing"}
METRICS = ("f1_macro", "worst_group_f1_macro", "sensitivity", "specificity", "balanced_accuracy")


def metrics(rows):
    truth = [binary(r["y_true"]) for r in rows]
    prediction = [binary(r["y_pred"]) for r in rows]
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(n=len(rows), positive=sum(truth), negative=len(rows) - sum(truth),
                tn=tn, fp=fp, fn=fn, tp=tp,
                f1_macro=float(f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0)),
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None)


def summarize(rows, min_group_size=1):
    result = metrics(rows)
    subgroups = []
    for attribute in ATTRIBUTES:
        groups = sorted({r[attribute] for r in rows} - UNKNOWN)
        for group in groups:
            selected = [r for r in rows if r[attribute] == group]
            eligible = len(selected) >= min_group_size
            subgroups.append(dict(attribute=attribute, group=group, eligible=eligible, **metrics(selected)))
    eligible = [r["f1_macro"] for r in subgroups if r["eligible"]]
    result["worst_group_f1_macro"] = min(eligible) if eligible else None
    result["eligible_subgroups"] = len(eligible)
    return result, subgroups


def load_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "run_config.json").read_text())
    cohort = json.loads((directory / "cohort.json").read_text())
    if digest(cohort) != config["cohort_sha256"]:
        raise ValueError(f"Cohort digest mismatch: {directory}")
    expected = {(c["task"], c["case_id"]): c for c in cohort}
    rows = read_csv(directory / "predictions.csv")
    seen = set()
    for row in rows:
        key = (row["task"], row["case_id"])
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        seen.add(key)
        if row["run_fingerprint"] != digest(config) or row["model_name"] != config["model"]:
            raise ValueError("Prediction model/configuration fingerprint mismatch")
        for field, item in expected[key].items():
            if str(row[field]) != str(item):
                raise ValueError(f"Prediction/cohort metadata mismatch: {key}/{field}")
        if row["status"] not in ("ok", "error"):
            raise ValueError(f"Unknown prediction status: {key}")
        if row["status"] == "ok":
            binary(row["y_pred"])
    return config, cohort, rows


def report(args):
    runs = [load_run(path) for path in args.run_dirs]
    if len({config["model"] for config, _, _ in runs}) != len(runs):
        raise ValueError("Supply at most one run directory per model")
    if len({config["cohort_sha256"] for config, _, _ in runs}) != 1:
        raise ValueError("Models were not evaluated on the identical cohort/data")
    if len({config["oct_slices"] for config, _, _ in runs}) != 1 or len({digest(config["prompts"]) for config, _, _ in runs}) != 1:
        raise ValueError("Models used different inputs or prompts")
    table, subgroups, incomplete = [], [], False
    for config, cohort, rows in runs:
        for task in TASKS:
            expected = [c for c in cohort if c["task"] == task]
            if not expected:
                continue
            selected = [r for r in rows if r["task"] == task]
            valid = [r for r in selected if r["status"] == "ok"]
            complete = len(valid) == len(expected)
            paper_size = len(expected) == 250
            item = dict(task=task, model=config["model"], modality="OCT/SLO",
                        expected_cases=len(expected), completed_cases=len(valid),
                        error_cases=len(selected) - len(valid), missing_cases=len(expected) - len(selected),
                        status="complete" if complete else "incomplete",
                        min_group_size=args.min_group_size,
                        **{key: None for key in ("n", "positive", "negative", "tn", "fp", "fn", "tp", *METRICS, "eligible_subgroups")})
            if complete and (paper_size or args.allow_small_cohort):
                aggregate, groups = summarize(valid, args.min_group_size)
                item.update(aggregate)
                subgroups.extend(dict(task=task, model=config["model"], **g) for g in groups)
            else:
                incomplete = True
                if complete:
                    item["status"] = "not_250_cases"
            table.append(item)
    out = Path(args.out_dir)
    table.sort(key=lambda row: (TASKS.index(row["task"]), row["model"]))
    write_csv(out / "table1_local_models.csv", table)
    if subgroups:
        write_csv(out / "subgroup_metrics.csv", subgroups)
    else:
        (out / "subgroup_metrics.csv").unlink(missing_ok=True)
    header = ["Task", "Model", "Modality", "F1", "Worst-group F1", "Sensitivity", "Specificity", "Balanced acc.", "Status"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    latex = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Model & Modality & F1 & Worst-group F1 & Sensitivity & Specificity & Balanced acc. \\",
        r"\hline",
    ]
    previous_task = None
    for row in table:
        scores = ["--" if row[key] is None else f"{row[key]:.3f}" for key in METRICS]
        label = "MedGemma 4B" if row["model"] == "medgemma" else "Gemma 3 4B"
        lines.append("| " + " | ".join([row["task"], label, "OCT/SLO", *scores, row["status"]]) + " |")
        if row["task"] != previous_task:
            title = {"glaucoma": "Glaucoma", "amd": "AMD", "dr": "Diabetic retinopathy"}[row["task"]]
            latex.append(r"\multicolumn{7}{l}{\textit{" + title + r"}} \\")
            previous_task = row["task"]
        latex.append(" & ".join([label, "OCT/SLO", *scores]) + r" \\")
    latex.extend([r"\hline", r"\end{tabular}"])
    (out / "table1_local_models.md").write_text("\n".join(lines) + "\n")
    (out / "table1_local_models.tex").write_text("\n".join(latex) + "\n")
    atomic_json(out / "report_config.json", dict(
        runs=[str(Path(p).resolve()) for p in args.run_dirs], min_group_size=args.min_group_size,
        age_bins="<50, 50–69, >=70", attributes=ATTRIBUTES, f1_average="macro",
        fixed_labels=[0, 1], zero_division=0, unknown_groups_excluded=True,
        subgroup_rule="All observed marginal groups meeting min_group_size, including single-class groups.",
        complete=not incomplete, cohort_sha256=runs[0][0]["cohort_sha256"],
        note="No probability/AUROC inferred from verbal confidence. Incomplete cohorts have no headline metrics."))
    print("\n".join(lines))
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-group-size", type=int, default=1,
                        help="Paper gives no support cutoff; default includes all observed marginal groups")
    parser.add_argument("--allow-small-cohort", action="store_true", help="Smoke reporting only")
    args = parser.parse_args()
    if args.min_group_size < 1:
        parser.error("min-group-size must be positive")
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
