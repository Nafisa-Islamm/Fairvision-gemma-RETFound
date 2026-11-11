"""Prepare strictly separated, probability-free caches for routing experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fairvision_local_vlm import TASKS, atomic_json, binary, digest, read_csv
from report_fairvision_local_vlm import load_run

ROLES = {"train": "training", "validation": "validation", "test": "test"}
CONFIDENCE = ("low", "moderate", "high", "unknown")
RACES = ("asian", "black", "white", "unknown")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(run_dirs, split, output, patient_map=None, confirm_one_case_per_patient=False, embeddings=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Bundle output exists; choose a new path")
    if patient_map is None and not confirm_one_case_per_patient:
        raise ValueError("Provide --patient-map or explicitly --confirm-one-case-per-patient after checking dataset identities")
    runs = [load_run(path) for path in run_dirs]
    names = [config["model"] for config, _, _ in runs]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("At least two distinct experts are required")
    if len({config["cohort_sha256"] for config, _, _ in runs}) != 1:
        raise ValueError("Expert cohort/data mismatch")
    if len({digest(config["prompts"]) for config, _, _ in runs}) != 1 or len({config["oct_slices"] for config, _, _ in runs}) != 1:
        raise ValueError("Expert prompts or image selection mismatch")
    if any(config.get("data_split", "test") != ROLES[split] for config, _, _ in runs):
        raise ValueError("Run split differs from bundle split; existing test runs cannot be called training data")
    cohort = runs[0][1]
    keys = [(c["task"], c["case_id"]) for c in cohort]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cohort keys")
    tables = []
    for _, _, rows in runs:
        table = {(r["task"], r["case_id"]): r for r in rows}
        if set(table) != set(keys) or any(r["status"] != "ok" for r in rows):
            raise ValueError("Every expert must have a valid prediction for every case")
        tables.append(table)
    patients = {}
    if patient_map:
        for row in read_csv(patient_map):
            key = row["task"], row["case_id"]
            if key in patients or not row["patient_id"].strip():
                raise ValueError("Duplicate/missing patient identity")
            patients[key] = row["patient_id"].strip()
        if not set(keys).issubset(patients):
            raise ValueError("Patient map does not cover every case")
    vectors = {}
    if embeddings:
        for row in json.loads(Path(embeddings).read_text()):
            key = row["task"], row["case_id"]
            vector = row["embedding"]
            if key in vectors or not vector or not np.isfinite(vector).all():
                raise ValueError("Duplicate, empty or nonfinite embedding")
            vectors[key] = vector
        if set(vectors) != set(keys) or len({len(v) for v in vectors.values()}) != 1:
            raise ValueError("Embeddings must cover exact case keys with a fixed dimension")
    cases = []
    for case, key in zip(cohort, keys):
        record = dict(case)
        record.update(patient_id=patients[key] if patient_map else f"{key[0]}:{key[1]}",
                      expert_preds=[binary(table[key]["y_pred"]) for table in tables],
                      expert_confidence=[table[key].get("confidence", "unknown") for table in tables],
                      embedding=vectors.get(key, []))
        cases.append(record)
    data = dict(format_version=1, split=split, experts=names, cases=cases,
                provenance=dict(identity="patient_map" if patient_map else "asserted_one_case_per_patient",
                                patient_map_sha256=file_hash(patient_map) if patient_map else None,
                                embedding_sha256=file_hash(embeddings) if embeddings else None,
                                prompts_sha256=digest(runs[0][0]["prompts"]), oct_slices=runs[0][0]["oct_slices"],
                                runs=[dict(path=str(Path(path).resolve()), config_fingerprint=digest(run[0]),
                                           predictions_sha256=file_hash(Path(path) / "predictions.csv"))
                                      for path, run in zip(run_dirs, runs)]))
    validate_bundle(data, split)
    atomic_json(output, data)
    return data


def validate_bundle(data, expected_split=None):
    if data.get("format_version") != 1 or data.get("split") not in ROLES:
        raise ValueError("Unsupported bundle format or split")
    if expected_split and data["split"] != expected_split:
        raise ValueError(f"Expected {expected_split} bundle, got {data['split']}")
    experts = data["experts"]
    if len(experts) < 2 or len(experts) != len(set(experts)) or not data["cases"]:
        raise ValueError("Need distinct experts and nonempty cases")
    seen, dimensions = set(), set()
    for case in data["cases"]:
        key = case["task"], case["case_id"]
        if case["task"] not in TASKS or key in seen or not str(case["patient_id"]).strip():
            raise ValueError("Invalid task, duplicate case or missing patient ID")
        image_hash = case.get("npz_sha256", "")
        if len(image_hash) != 64 or any(char not in "0123456789abcdef" for char in image_hash):
            raise ValueError("Missing/invalid image hash; preserve NPZ provenance from inference caches")
        seen.add(key)
        binary(case["y_true"])
        if len(case["expert_preds"]) != len(experts) or len(case["expert_confidence"]) != len(experts):
            raise ValueError("Missing expert outputs; no synthetic probabilities or default predictions are allowed")
        for value in case["expert_preds"]:
            binary(value)
        if any(conf not in CONFIDENCE for conf in case["expert_confidence"]):
            raise ValueError("Confidence must be low/moderate/high/unknown, not a probability")
        vector = case.get("embedding", [])
        dimensions.add(len(vector))
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite embedding")
    if len(dimensions) != 1:
        raise ValueError("Embedding dimensions differ")
    return data


def load_bundle(path, expected_split=None):
    return validate_bundle(json.loads(Path(path).read_text()), expected_split)


def check_separation(*bundles):
    if len({tuple(b["experts"]) for b in bundles}) != 1:
        raise ValueError("Expert identities/order differ between splits")
    if len({len(b["cases"][0].get("embedding", [])) for b in bundles}) != 1:
        raise ValueError("Embedding dimensions differ between splits")
    if len({b["provenance"]["identity"] for b in bundles}) != 1:
        raise ValueError("Patient identity conventions differ between splits")
    if len({(b["provenance"].get("prompts_sha256"), b["provenance"].get("oct_slices")) for b in bundles}) != 1:
        raise ValueError("Prompt/image protocol differs between splits")
    for i, left in enumerate(bundles):
        for right in bundles[i + 1:]:
            for field in ("patient_id", "npz_sha256"):
                a = {c.get(field) for c in left["cases"]} - {None, ""}
                b = {c.get(field) for c in right["cases"]} - {None, ""}
                if a & b:
                    raise ValueError(f"Split leakage: overlapping {field}")
            a = {(c["task"], c["case_id"]) for c in left["cases"]}
            b = {(c["task"], c["case_id"]) for c in right["cases"]}
            if a & b:
                raise ValueError("Split leakage: overlapping task/case_id")


def feature_matrix(data, race_aware=False, mode="arbitration"):
    rows = []
    for case in data["cases"]:
        values = [float(case["task"] == task) for task in TASKS]
        try:
            age = float(case.get("age", "nan"))
            missing = not np.isfinite(age) or age < 0
        except (TypeError, ValueError):
            age, missing = 0., True
        values += [0. if missing else age / 100, float(missing)]
        for field, categories in (("sex_gender", ("female", "male", "unknown")),
                                  ("ethnicity", ("hispanic", "non-hispanic", "unknown"))):
            value = case.get(field, "unknown")
            value = value if value in categories else "unknown"
            values.extend(float(value == category) for category in categories)
        if race_aware:
            race = case.get("race", "unknown")
            race = race if race in RACES else "unknown"
            values.extend(float(race == category) for category in RACES)
        if mode == "arbitration":
            values.extend(float(binary(value)) for value in case["expert_preds"])
            for confidence in case["expert_confidence"]:
                values.extend(float(confidence == category) for category in CONFIDENCE)
        elif mode != "prequery":
            raise ValueError("Unknown routing mode")
        values.extend(case.get("embedding", []))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)
