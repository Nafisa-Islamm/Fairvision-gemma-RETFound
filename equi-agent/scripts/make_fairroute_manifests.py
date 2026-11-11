"""Create explicit training/validation manifests from the original dataset summary splits."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fairvision_local_vlm import ROOT, atomic_json, read_csv, write_csv
from fairroute_data import file_hash


def make_manifests(metadata_root, out_dir, max_per_task=0, seed=2026):
    metadata_root, out_dir = Path(metadata_root), Path(out_dir)
    if out_dir.exists():
        raise ValueError("Output directory exists; choose a new path")
    manifests, sources = {"training": [], "validation": []}, []
    rng = np.random.default_rng(seed)
    allowed = {"glaucoma": {"no": 0, "yes": 1},
               "amd": {"normal": 0, "early amd": 1, "intermediate amd": 1, "late amd": 1},
               "dr": {"non-vision threatening dr": 0, "vision threatening dr": 1}}
    for folder, task in (("Glaucoma", "glaucoma"), ("AMD", "amd"), ("DR", "dr")):
        source = metadata_root / folder / f"data_summary_{task}.csv"
        rows = read_csv(source)
        names = [row["filename"] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source filenames; verify identities before splitting")
        sources.append(dict(path=str(source.resolve()), sha256=file_hash(source)))
        for split in manifests:
            subset = [row for row in rows if row["use"].strip().lower() == split]
            if not subset:
                raise ValueError(f"No {split} source cases for {task}")
            if max_per_task and len(subset) > max_per_task:
                subset = [subset[i] for i in sorted(rng.choice(len(subset), max_per_task, replace=False))]
            for row in subset:
                label = row[task].strip().lower()
                if label not in allowed[task]:
                    raise ValueError(f"Unknown {task} clinical label: {label}")
                manifests[split].append(dict(task=task, case_id=Path(row["filename"]).stem,
                    filename=f"{folder}/{split}/{row['filename']}", split=split,
                    y_true=allowed[task][label], age=row["age"], sex_gender=row["gender"],
                    race=row["race"], ethnicity=row["ethnicity"]))
    out_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in manifests.items():
        write_csv(out_dir / f"{split}.csv", rows)
    atomic_json(out_dir / "manifest_config.json", dict(
        sources=sources, seed=seed, max_per_task=max_per_task, label_mapping=allowed,
        sampling="Uniform within predefined disease/split, without using expert outputs; 0 means all source cases",
        counts={split: dict(Counter(row["task"] for row in rows)) for split, rows in manifests.items()},
        race_class_support={split: [{"task": task, "race": race,
                            "positive": sum(row["y_true"] for row in rows if row["task"] == task and row["race"] == race),
                            "negative": sum(1 - row["y_true"] for row in rows if row["task"] == task and row["race"] == race)}
                            for task, race in sorted({(row["task"], row["race"]) for row in rows})]
                            for split, rows in manifests.items()},
        note="DR positive means vision-threatening DR, matching the existing paper cohort. No test cases were selected. Verify patient identities separately."))
    print(f"Saved split manifests: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "OphthalmicAgent/data")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.max_per_task < 0:
        parser.error("max-per-task cannot be negative")
    make_manifests(args.metadata_root, args.out_dir, args.max_per_task, args.seed)
