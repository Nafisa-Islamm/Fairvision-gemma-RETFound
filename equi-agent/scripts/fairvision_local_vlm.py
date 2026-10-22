"""Local, resumable FairVision inference. No paid API or foundation-model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"medgemma": "google/medgemma-4b-it", "gemma": "google/gemma-3-4b-it",
          "gemma4": "google/gemma-4-12B", "gemma4-it": "google/gemma-4-12B-it"}
MODEL_LABELS = {"medgemma": "MedGemma 4B", "gemma": "Gemma 3 4B",
                "gemma4": "Gemma 4 12B", "gemma4-it": "Gemma 4 12B IT"}
TASKS = ("glaucoma", "amd", "dr")
VERSION = "fairvision-local-vlm-v1"
# Same disease instructions as the repository's standalone baselines.
PROMPTS = {
    "glaucoma": (
        "You are a standalone ophthalmic image evaluator. Determine whether glaucoma is "
        "present using only the supplied SLO/fundus image, representative OCT B-scans, "
        "and basic demographics. Demographics are contextual metadata only and must never "
        "be treated as anatomical evidence or as proof of disease. Examine the optic nerve "
        "head, cupping and neuroretinal rim when visible, retinal nerve fiber layer or other "
        "glaucoma-related structural patterns, image quality, and agreement between the two "
        "modalities. Do not invent measurements, history, symptoms, intraocular pressure, "
        "visual fields, CDR values, or model scores. Return JSON only with exactly these "
        "fields: glaucoma_detected (integer 0 or 1), confidence (low, moderate, or high), "
        "reasoning (brief evidence-based explanation). You must choose 0 or 1 even when "
        "uncertain, and express uncertainty in confidence and reasoning."
    ),
    "amd": (
        "You are a standalone ophthalmic image evaluator. Determine whether age-related macular "
        "degeneration (AMD) is present using only the supplied SLO/fundus image, representative OCT "
        "B-scans, and basic demographics. Demographics are contextual metadata only and must never "
        "be treated as anatomical proof of disease. Examine the macula for drusen or drusenoid "
        "elevations, pigmentary abnormalities, RPE disruption, geographic atrophy, subretinal or "
        "intraretinal fluid, pigment epithelial detachment, subretinal hyperreflective material, "
        "fibrosis, and other visible AMD-related changes. Consider image quality and agreement between "
        "the fundus and OCT evidence. Do not invent measurements, history, symptoms, visual acuity, "
        "angiography findings, or foundation-model scores. Return JSON only with exactly these fields: "
        "amd_detected (integer 0 or 1), confidence (low, moderate, or high), and reasoning (brief "
        "evidence-based explanation). You must choose 0 or 1 even when uncertain."
    ),
    "dr": (
        "You are a standalone ophthalmic image evaluator. Determine the binary diabetic retinopathy "
        "(DR) label using only the supplied SLO/fundus image, representative OCT B-scans, and basic "
        "demographics. Demographics are contextual metadata only and must never be treated as anatomical "
        "proof of disease. Examine visible retinal findings including microaneurysms, dot-blot or flame "
        "hemorrhages, hard exudates, cotton-wool spots, venous beading, IRMA, neovascularization, vitreous "
        "hemorrhage, and tractional changes. On OCT, assess intraretinal cysts or fluid, subretinal fluid, "
        "retinal thickening, and other diabetic macular changes. Consider image quality and agreement "
        "between modalities. Do not invent laboratory values, diabetes duration, visual acuity, treatment "
        "history, angiography findings, or foundation-model scores. Return JSON only with exactly these "
        "fields: dr_detected (integer 0 or 1), confidence (low, moderate, or high), and reasoning (brief "
        "evidence-based explanation). You must choose 0 or 1 even when uncertain."
    ),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def value(row, *names, default="unknown"):
    lookup = {key.lower(): str(val).strip() for key, val in row.items() if val is not None}
    for name in names:
        text = lookup.get(name.lower(), "")
        if text and text.lower() not in ("nan", "none", "null"):
            return text
    return default


def binary(text):
    number = float(text)
    if number not in (0, 1):
        raise ValueError(f"Expected binary label, got {text!r}")
    return int(number)


def prepare_manifest(args):
    """Preserve patient selection; resolve only missing glaucoma truth from source metadata."""
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source:
        raise ValueError("Do not overwrite the original manifest")
    rows = read_csv(source)
    glaucoma_source = args.glaucoma_summary.resolve()
    lookup = {}
    for row in read_csv(glaucoma_source):
        key = Path(row["filename"]).name
        if key in lookup:
            raise ValueError(f"Duplicate glaucoma source filename: {key}")
        lookup[key] = row
    changes = []
    for row in rows:
        task = value(row, "Task_Folder", "task").lower()
        truth = value(row, "Ground_Truth", "y_true")
        if task == "glaucoma":
            key = Path(value(row, "filename", "image_path")).name
            reference = lookup[key]
            if reference["use"].strip().lower() != "test":
                raise ValueError(f"Source case is not test: {key}")
            text = reference["glaucoma"].strip().lower()
            if text not in ("yes", "no"):
                raise ValueError(f"Invalid source glaucoma label: {key}/{text}")
            expected = int(text == "yes")
            if truth == "-1":
                if "Ground_Truth" not in row:
                    raise ValueError("Label recovery requires legacy Ground_Truth header")
                row["Ground_Truth"] = str(expected)
                changes.append(dict(task=task, filename=key, original_label=truth,
                                    corrected_label=expected, source=str(glaucoma_source)))
            elif binary(truth) != expected:
                raise ValueError(f"Manifest/source glaucoma label mismatch: {key}")
        else:
            binary(truth)
    if output.exists():
        if read_csv(output) != rows:
            raise ValueError("Output manifest already exists with different contents")
    else:
        write_csv(output, rows)
    cases = load_cohort(output)
    counts = {task: {"n": sum(c["task"] == task for c in cases),
                     "positive": sum(c["task"] == task and c["y_true"] == 1 for c in cases)}
              for task in TASKS}
    if any(item["n"] != 250 for item in counts.values()):
        raise ValueError(f"Expected 250 cases per task, got {counts}")
    audit = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 glaucoma_summary_sha256=hashlib.sha256(glaucoma_source.read_bytes()).hexdigest(),
                 changes=changes, counts=counts,
                 note="Source CSV label verification; NPZ clinical labels not independently verified.")
    atomic_json(output.with_suffix(".audit.json"), audit)
    print(json.dumps(audit, indent=2))


def age_group(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(age) or age < 0:
        return "unknown"
    return "younger" if age < 50 else "middle-aged" if age < 70 else "older"


def load_cohort(manifest, tasks=TASKS, limit=0):
    cases, seen, counts = [], set(), dict.fromkeys(tasks, 0)
    for row in read_csv(manifest):
        task = value(row, "task", "Task_Folder", "disease_folder", "disease").lower()
        if task not in tasks:
            continue
        split = value(row, "split", default="test").lower()
        if split not in ("test", "testing"):
            raise ValueError(f"Selected manifest contains non-test row: {task}/{split}")
        filename = value(row, "filename", "npz_path", "image_path", "filepath", "path", "image_id", default="")
        if not filename or Path(filename).suffix.lower() != ".npz":
            raise ValueError("Each selected manifest row needs an NPZ filename/path")
        case_id = value(row, "case_id", "patient_id", default=Path(filename).stem)
        key = (task, case_id)
        if key in seen:
            raise ValueError(f"Duplicate patient/case in task: {key}")
        seen.add(key)
        case = dict(task=task, case_id=case_id, filename=filename,
                    y_true=binary(value(row, "y_true", "Ground_Truth", "label", "gt")),
                    age=value(row, "age"), sex_gender=value(row, "sex_gender", "Gender", "Sex").lower(),
                    race=value(row, "race").lower(), ethnicity=value(row, "ethnicity").lower())
        case["age_group"] = age_group(case["age"])
        if not limit or counts[task] < limit:
            cases.append(case)
            counts[task] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Manifest is missing requested tasks: {counts}")
    return cases


def resolve_npz(filename, manifest, data_root):
    """Support legacy disease/Test and documented shared Test layouts; reject ambiguity."""
    source, root = Path(filename).expanduser(), Path(data_root).expanduser().resolve()
    candidates = [source] if source.is_absolute() else [Path(manifest).resolve().parent / source, root / source]
    if not source.is_absolute():
        parts = source.parts
        if parts and parts[0].lower() == "data":
            candidates.append(root.joinpath(*parts[1:]))
        candidates.append(root / "Test" / source.name)
        # Drive exports may use Amd/Dr while the paper manifest uses AMD/DR.
        # Match actual disease/split directory names; never alter source data.
        if root.is_dir():
            parents = [root]
            if len(parts) >= 3 and parts[-2].casefold() == "test":
                disease = parts[-3].casefold()
                parents.extend(p for p in root.iterdir() if p.is_dir() and p.name.casefold() == disease)
            for parent in parents:
                candidates.extend(p / source.name for p in parent.iterdir()
                                  if p.is_dir() and p.name.casefold() == "test")
    existing = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(existing) != 1:
        raise ValueError(f"Need exactly one file for {filename}; found {existing}. Set --data-root or fix manifest paths.")
    return existing[0]


def normalize_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if not finite.size:
        raise ValueError("Image contains no finite pixels")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.nan_to_num((array - low) / (high - low), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(255 * scaled, 0, 255).astype(np.uint8)


def jpeg_image(image):
    # Baseline API scripts send JPEG quality 95; preserve that preprocessing.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load_images(path, count=8):
    with np.load(path, allow_pickle=False) as container:
        fundus, volume = container["slo_fundus"], container["oct_bscans"]
    if fundus.ndim not in (2, 3) or (fundus.ndim == 3 and fundus.shape[-1] not in (3, 4)):
        raise ValueError(f"Expected SLO [H,W] or [H,W,3/4], got {fundus.shape}")
    if volume.ndim != 3 or not len(volume):
        raise ValueError(f"Expected OCT [slices,H,W], got {volume.shape}")
    indices = np.linspace(0, len(volume) - 1, min(count, len(volume)), dtype=int)
    images = [ImageOps.contain(Image.fromarray(normalize_uint8(volume[i])).convert("RGB"), (448, 224))
              for i in indices]
    width, height = max(im.width for im in images), max(im.height for im in images)
    columns = min(4, len(images))
    grid = Image.new("RGB", (columns * width, math.ceil(len(images) / columns) * height), "black")
    for position, image in enumerate(images):
        grid.paste(image, ((position % columns) * width + (width - image.width) // 2,
                          (position // columns) * height + (height - image.height) // 2))
    return jpeg_image(Image.fromarray(normalize_uint8(fundus)).convert("RGB")), jpeg_image(grid), indices.tolist()


def prompt_text(case, indices):
    return (PROMPTS[case["task"]] + "\n\nDemographics:\n"
            f"Age: {case['age']}\nGender: {case['sex_gender']}\nRace: {case['race']}\n"
            f"Ethnicity: {case['ethnicity']}\n\nFirst image: SLO/fundus. "
            f"Second image: evenly spaced OCT B-scans from slice indices {indices}. "
            f"Assess the binary {case['task']} label.")


def parse_prediction(raw, task):
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    # Do not search arbitrary reasoning for a label or coerce malformed values.
    result = json.loads(text)
    prediction = result.get(f"{task}_detected")
    if type(prediction) is not int or prediction not in (0, 1):
        raise ValueError(f"Expected integer {task}_detected: 0 or 1")
    if result.get("confidence") not in ("low", "moderate", "high"):
        raise ValueError("Expected confidence low/moderate/high")
    if not isinstance(result.get("reasoning"), str):
        raise ValueError("Expected reasoning string")
    return prediction, result["confidence"], result["reasoning"]


def download_model(args):
    from huggingface_hub import snapshot_download
    # HF_TOKEN or cached 'hf auth login' is used automatically; never log tokens.
    snapshot = snapshot_download(MODELS[args.model], revision=args.revision, cache_dir=args.cache_dir)
    print(f"Model snapshot: {snapshot}", flush=True)
    if args.snapshot_file:
        Path(args.snapshot_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.snapshot_file).write_text(str(Path(snapshot).resolve()) + "\n")


def prepare_cases(args):
    cases = load_cohort(args.manifest, args.tasks, args.limit_per_task)
    for case in cases:
        path = resolve_npz(case["filename"], args.manifest, args.data_root)
        load_images(path, args.oct_slices)
        case["npz_path"] = str(path)
        # Identity protects resume and cross-model comparison if data change.
        case["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cases


def infer(args):
    import fcntl
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if args.model in ("gemma4", "gemma4-it"):
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError("Gemma 4 needs its separate environment: run install_gemma4.sh") from exc
        model_loader = AutoModelForMultimodalLM
    else:
        model_loader = AutoModelForImageTextToText
    cases = prepare_cases(args)
    snapshot = Path(args.model_path).expanduser().resolve()
    if not snapshot.is_dir():
        raise ValueError("--model-path must be a downloaded Hugging Face snapshot directory")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        config = dict(version=VERSION, source_sha256=source_hash, model=args.model,
                      model_id=MODELS[args.model], model_path=str(snapshot),
                      model_config_sha256=hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
                      cohort_sha256=digest(cases), prompts=PROMPTS,
                      oct_slices=args.oct_slices, max_new_tokens=args.max_new_tokens,
                      dtype=args.dtype, device=args.device, do_sample=False, disable_compile=True, seed=2026,
                      python=platform.python_version(), torch=torch.__version__)
        import transformers
        config["transformers"] = transformers.__version__
        fingerprint = digest(config)
        config_path = out / "run_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Run configuration/cohort changed; choose a NEW --out-dir")
        if not config_path.exists() and (out / "predictions.csv").exists():
            raise ValueError("Existing predictions without configuration; choose a new output directory")
        atomic_json(config_path, config)
        atomic_json(out / "cohort.json", cases)
        previous = read_csv(out / "predictions.csv") if (out / "predictions.csv").exists() else []
        results = {}
        expected = {(c["task"], c["case_id"]): c for c in cases}
        for row in previous:
            key = (row["task"], row["case_id"])
            if key in results or key not in expected or row["run_fingerprint"] != fingerprint:
                raise ValueError("Duplicate or incompatible saved prediction")
            if any(str(row[field]) != str(item) for field, item in expected[key].items()):
                raise ValueError("Saved prediction/cohort metadata mismatch")
            if row["status"] not in ("ok", "error"):
                raise ValueError("Invalid saved prediction status")
            if row["status"] == "ok":
                binary(row["y_pred"])
            results[key] = row
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Run inference inside a GPU allocation on Newton.")
        if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16; use --dtype float16 and a new output directory")
        if args.device == "cpu" and args.dtype != "float32":
            raise ValueError("CPU inference requires --dtype float32 (GPU strongly recommended)")
        if any((c["task"], c["case_id"]) not in results or
               (args.retry_errors and results[c["task"], c["case_id"]]["status"] != "ok") for c in cases):
            torch.manual_seed(2026)
            processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            if not processor.chat_template:
                raise ValueError("Processor has no chat template. For Gemma 4 instruction-based evaluation, "
                                 "download google/gemma-4-12B-it and use --model gemma4-it with a new output directory.")
            model = model_loader.from_pretrained(
                snapshot, torch_dtype=dtype, local_files_only=True,
                device_map={"": args.device}, attn_implementation="eager").eval()
            model_type = model.config.model_type
            expected_type = "gemma4_unified" if args.model in ("gemma4", "gemma4-it") else "gemma3"
            if model_type != expected_type:
                raise ValueError(f"Expected {expected_type} checkpoint, got {model_type}")
            print(f"Loaded {args.model} on {args.device}; {len(cases)} cases", flush=True)
            for index, case in enumerate(cases, 1):
                key = (case["task"], case["case_id"])
                if key in results and (results[key]["status"] == "ok" or not args.retry_errors):
                    continue
                started, raw, prediction, confidence, reasoning, status, error = time.time(), "", "", "", "", "ok", ""
                try:
                    slo, oct_grid, indices = load_images(case["npz_path"], args.oct_slices)
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt_text(case, indices)},
                        {"type": "image", "image": slo}, {"type": "image", "image": oct_grid}]}]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt").to(model.device, dtype=dtype)
                    input_length = inputs["input_ids"].shape[-1]
                    with torch.inference_mode():
                        # Gemma checkpoints may enable compilation through their cache configuration.
                        # Eager generation avoids requiring Python development headers on compute nodes.
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                   do_sample=False, disable_compile=True)
                    raw = processor.decode(generated[0][input_length:], skip_special_tokens=True)
                    prediction, confidence, reasoning = parse_prediction(raw, case["task"])
                except torch.cuda.OutOfMemoryError:
                    # Fatal resource error: resume with a suitable allocation, not hundreds of failed rows.
                    raise
                except Exception as exc:
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                row = dict(**case, model_name=args.model, model_id=MODELS[args.model],
                           y_pred=prediction, confidence=confidence, reasoning=reasoning,
                           raw_response=raw, status=status, error=error,
                           elapsed_seconds=round(time.time() - started, 3), run_fingerprint=fingerprint)
                # Retain all attempts when --retry-errors is explicitly requested.
                with (out / "attempts.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                results[key] = row
                write_csv(out / "predictions.csv", [results[c["task"], c["case_id"]] for c in cases
                                                  if (c["task"], c["case_id"]) in results])
                print(f"{index}/{len(cases)} {key}: {status} prediction={prediction}", flush=True)
                if status != "ok":
                    print(error, flush=True)
        errors = sum(row["status"] != "ok" for row in results.values())
        print(f"Complete: {len(results)}/{len(cases)} rows; {errors} errors; {out}", flush=True)
        return 2 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-manifest", help="Preserve tracked cohort; audit source-metadata label recovery")
    prepare.add_argument("--source", type=Path, default=ROOT / "OphthalmicAgent/data/fairvision_250each.csv")
    prepare.add_argument("--glaucoma-summary", type=Path, default=ROOT / "OphthalmicAgent/data/Glaucoma/data_summary_glaucoma.csv")
    prepare.add_argument("--output", type=Path, default=ROOT / "equi-agent/outputs/local_vlm/cohort.csv")
    download = commands.add_parser("download", help="Download licensed model once, before GPU jobs")
    download.add_argument("--model", choices=MODELS, required=True)
    download.add_argument("--revision", default="main")
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--snapshot-file", type=Path)
    for command in ("check", "infer"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, default=ROOT / "equi-agent/outputs/local_vlm/cohort.csv")
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
        sub.add_argument("--limit-per-task", type=int, default=0, help="Smoke runs only; 0 means entire cohort")
        sub.add_argument("--oct-slices", type=int, default=8)
        if command == "infer":
            sub.add_argument("--model", choices=MODELS, required=True)
            sub.add_argument("--model-path", required=True)
            sub.add_argument("--out-dir", type=Path, required=True)
            sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            sub.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
            sub.add_argument("--max-new-tokens", type=int, default=512)
            sub.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-manifest":
        prepare_manifest(args)
        return 0
    if args.command == "download":
        download_model(args)
        return 0
    if args.oct_slices < 1 or args.limit_per_task < 0:
        parser.error("oct-slices must be positive and limit-per-task nonnegative")
    if args.command == "check":
        cases = prepare_cases(args)
        print(json.dumps({"cases": len(cases), "cohort_sha256": digest(cases),
                          "tasks": {task: {"n": sum(c["task"] == task for c in cases),
                                           "positive": sum(c["task"] == task and c["y_true"] == 1 for c in cases)}
                                    for task in args.tasks}}, indent=2))
        return 0
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    return infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
