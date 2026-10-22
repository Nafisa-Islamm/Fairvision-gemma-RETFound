"""Audit narrowly recoverable JSON formatting errors into a NEW result directory."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from fairvision_local_vlm import atomic_json, digest, parse_prediction, write_csv
from report_fairvision_local_vlm import load_run


def quote_confidence(raw, task):
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        token = re.match(r"(low|moderate|high)(?=\s*[,}])", text[error.pos:])
        if (error.msg != "Expecting value" or token is None or
                not re.search(r'"confidence"\s*:\s*$', text[:error.pos])):
            raise ValueError("Not an unquoted confidence enum") from error
        repaired = text[:error.pos] + json.dumps(token.group(1)) + text[error.pos + len(token.group(1)):]
        return repaired, parse_prediction(repaired, task)
    raise ValueError("Response already has valid JSON; no formatting repair applies")


def recover(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Choose a NEW sibling output directory, outside the source run")
    if destination.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")
    config, cohort, rows = load_run(source)
    if "postprocessing" in config:
        raise ValueError("Use an original inference run, not an already recovered run")
    if not rows:
        raise ValueError("Source has no prediction rows")
    repairs = []
    for row in rows:
        if row["status"] != "error":
            continue
        try:
            repaired, (prediction, confidence, reasoning) = quote_confidence(row["raw_response"], row["task"])
        except (ValueError, TypeError, AttributeError):
            continue
        repairs.append(dict(task=row["task"], case_id=row["case_id"],
                            original_error=row["error"], prediction=prediction,
                            raw_response_sha256=hashlib.sha256(row["raw_response"].encode()).hexdigest(),
                            repaired_response=repaired))
        # Preserve original raw_response and timing; recover only structured fields.
        row.update(y_pred=prediction, confidence=confidence, reasoning=reasoning, status="ok", error="")
    if not repairs:
        raise ValueError("No matching formatting errors found; no files were written")
    audit = dict(method="quote_unquoted_confidence_v1", repairs=repairs,
                 recovery_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 note="Only quotes around an existing confidence enum were inserted. Diagnoses were not changed. Original raw responses remain unchanged.")
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "recovery_audit.json", audit)
    original_config = dict(config)
    config["postprocessing"] = dict(
        method=audit["method"], source_run=str(source), source_run_fingerprint=digest(original_config),
        source_predictions_sha256=hashlib.sha256((source / "predictions.csv").read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256((destination / "recovery_audit.json").read_bytes()).hexdigest())
    for row in rows:
        row["run_fingerprint"] = digest(config)
    atomic_json(destination / "source_run_config.json", original_config)
    atomic_json(destination / "run_config.json", config)
    atomic_json(destination / "cohort.json", cohort)
    write_csv(destination / "predictions.csv", rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(f"Recovered formatting errors: {len(repairs)}")
    print(f"Valid predictions: {len(rows) - errors}/{len(cohort)}; remaining errors: {errors}")
    print(f"Saved derived results: {destination}")
    print("Original run unchanged. Disclose formatting recovery when reporting these metrics.")
    return 2 if errors or len(rows) != len(cohort) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(recover(args.run_dir, args.out_dir))
