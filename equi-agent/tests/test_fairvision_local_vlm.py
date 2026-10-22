"""Data, resume, and paper-table contracts for the local FairVision baselines."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fairvision_local_vlm as vlm
import report_fairvision_local_vlm as reporting


class LocalVLMTests(unittest.TestCase):
    def test_repository_cohort_recovery_preserves_750_patients(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = vlm.ROOT / "OphthalmicAgent/data/fairvision_250each.csv"
            args = argparse.Namespace(source=source,
                glaucoma_summary=vlm.ROOT / "OphthalmicAgent/data/Glaucoma/data_summary_glaucoma.csv",
                output=Path(tmp) / "cohort.csv")
            before = source.read_bytes()
            with patch("builtins.print"):
                vlm.prepare_manifest(args)
                vlm.prepare_manifest(args)
            cases = vlm.load_cohort(args.output)
            self.assertEqual(len(cases), 750)
            counts = {t: sum(c["y_true"] for c in cases if c["task"] == t) for t in vlm.TASKS}
            self.assertEqual(counts, dict(glaucoma=124, amd=125, dr=125))
            self.assertEqual(source.read_bytes(), before)
            audit = json.loads(args.output.with_suffix(".audit.json").read_text())
            self.assertEqual(len(audit["changes"]), 1)
            self.assertEqual(audit["changes"][0]["filename"], "data_07199.npz")

    def test_duplicate_and_non_test_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            row = dict(task="dr", filename="x.npz", y_true=0, split="test")
            vlm.write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                vlm.load_cohort(path, ["dr"])
            vlm.write_csv(path, [{**row, "split": "val"}])
            with self.assertRaisesRegex(ValueError, "non-test"):
                vlm.load_cohort(path, ["dr"])

    def test_labels_are_never_exposed_in_prompts(self):
        case = dict(task="dr", y_true=0, age="65", sex_gender="female", race="black", ethnicity="non-hispanic")
        original = vlm.prompt_text(case, [0, 1])
        self.assertEqual(original, vlm.prompt_text({**case, "y_true": 1}, [0, 1]))
        self.assertNotIn("y_true", original)

    def test_images_and_two_dataset_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "Test/data_1.npz"
            path.parent.mkdir()
            np.savez(path, slo_fundus=np.zeros((40, 40, 3), dtype=np.uint8),
                     oct_bscans=np.zeros((10, 20, 40), dtype=np.uint8))
            self.assertEqual(vlm.resolve_npz("data/DR/Test/data_1.npz", root / "cohort.csv", root), path)
            slo, grid, indices = vlm.load_images(path)
            self.assertEqual(slo.mode, "RGB")
            self.assertEqual(indices, [0, 1, 2, 3, 5, 6, 7, 9])
            self.assertEqual(grid.size, (1792, 448))
            legacy = root / "DR/Test/data_1.npz"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "exactly one"):
                vlm.resolve_npz("data/DR/Test/data_1.npz", root / "cohort.csv", root)

    def test_strict_json_parsing(self):
        valid = dict(dr_detected=1, confidence="low", reasoning="Visible finding.")
        self.assertEqual(vlm.parse_prediction("```json\n" + json.dumps(valid) + "\n```", "dr")[0], 1)
        for bad in (True, 0.9, "1", 2):
            with self.assertRaises(ValueError):
                vlm.parse_prediction(json.dumps({**valid, "dr_detected": bad}), "dr")
        with self.assertRaises(ValueError):
            vlm.parse_prediction("I think yes", "dr")

    def test_google_drive_disease_directory_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for disease, actual in (("AMD", "Amd"), ("DR", "Dr"), ("Glaucoma", "Glaucoma")):
                target = root / actual / "test/data_1.npz"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"fixture")
                self.assertEqual(vlm.resolve_npz(f"data/{disease}/Test/data_1.npz",
                                                root / "manifest.csv", root), target)

    def test_macro_f1_is_not_positive_f1_and_age_boundaries(self):
        rows = [dict(y_true=y, y_pred=p, race="white", ethnicity="non-hispanic",
                     sex_gender="female", age_group="older")
                for y, p in ((0, 0), (0, 0), (0, 0), (1, 0))]
        metrics, _ = reporting.summarize(rows)
        self.assertAlmostEqual(metrics["f1_macro"], 3 / 7)
        self.assertAlmostEqual(metrics["balanced_accuracy"], .5)
        self.assertEqual(metrics["worst_group_f1_macro"], metrics["f1_macro"])
        self.assertEqual([vlm.age_group(x) for x in (49, 50, 69, 70)], ["younger", "middle-aged", "middle-aged", "older"])
        self.assertEqual(vlm.age_group("nan"), "unknown")

    def make_run(self, root, model="gemma", failed=False, omit=False):
        root.mkdir()
        cohort = [dict(task="dr", case_id=str(i), filename=f"{i}.npz", y_true=i % 2,
                       age="65", sex_gender="female", race="white", ethnicity="non-hispanic",
                       age_group="middle-aged", npz_path=f"/data/{i}.npz", npz_sha256="hash")
                  for i in range(2)]
        config = dict(model=model, cohort_sha256=vlm.digest(cohort), prompts=vlm.PROMPTS, oct_slices=8)
        vlm.atomic_json(root / "cohort.json", cohort)
        vlm.atomic_json(root / "run_config.json", config)
        rows = [dict(**case, model_name=model, y_pred=case["y_true"], status="ok",
                     run_fingerprint=vlm.digest(config)) for case in cohort]
        if failed:
            rows[0].update(status="error", y_pred="")
        vlm.write_csv(root / "predictions.csv", rows[:1] if omit else rows)
        return rows

    def test_reports_fail_closed_on_errors_and_missing_patients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, failed, omit in (("failed", True, False), ("missing", False, True)):
                self.make_run(root / name, failed=failed, omit=omit)
                args = argparse.Namespace(run_dirs=[root / name], out_dir=root / (name + "_report"),
                                          min_group_size=1, allow_small_cohort=True)
                with patch("builtins.print"):
                    self.assertEqual(reporting.report(args), 2)
                result = vlm.read_csv(args.out_dir / "table1_local_models.csv")[0]
                self.assertEqual(result["f1_macro"], "")
                self.assertEqual(result["status"], "incomplete")

    def test_reports_match_cohorts_and_validate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.make_run(root / "gemma")
            self.make_run(root / "medgemma", model="medgemma")
            args = argparse.Namespace(run_dirs=[root / "gemma", root / "medgemma"], out_dir=root / "table",
                                      min_group_size=1, allow_small_cohort=True)
            with patch("builtins.print"):
                self.assertEqual(reporting.report(args), 0)
            for result in vlm.read_csv(args.out_dir / "table1_local_models.csv"):
                self.assertEqual(float(result["f1_macro"]), 1.)
            rows[0]["y_true"] = 1
            vlm.write_csv(root / "gemma/predictions.csv", rows)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                reporting.load_run(root / "gemma")

    def test_cpu_inference_resume_and_raw_response_audit(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        class Inputs(dict):
            def to(self, *args, **kwargs):
                return self

        class Processor:
            chat_template = "fixture"
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return Inputs(input_ids=torch.tensor([[1]]))

            def decode(self, *args, **kwargs):
                return '{"dr_detected":0,"confidence":"low","reasoning":"fixture"}'

        class Model:
            config = argparse.Namespace(model_type="gemma3")
            device = torch.device("cpu")

            def eval(self):
                return self

            def generate(self, **kwargs):
                if kwargs.get("disable_compile") is not True:
                    raise AssertionError("Cluster inference must disable automatic compilation")
                return torch.tensor([[1, 2]])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "case.npz", slo_fundus=np.zeros((20, 20, 3), dtype=np.uint8),
                     oct_bscans=np.zeros((8, 20, 20), dtype=np.uint8))
            vlm.write_csv(root / "manifest.csv", [dict(task="dr", filename="case.npz", y_true=0)])
            snapshot = root / "model"
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{"model_type":"gemma3"}')
            args = argparse.Namespace(manifest=root / "manifest.csv", data_root=root, tasks=["dr"],
                                      limit_per_task=0, oct_slices=8, model_path=str(snapshot),
                                      out_dir=root / "out", model="gemma", dtype="float32", device="cpu",
                                      max_new_tokens=64, retry_errors=False)
            with patch.object(AutoProcessor, "from_pretrained", return_value=Processor()), \
                 patch.object(AutoModelForImageTextToText, "from_pretrained", return_value=Model()) as loader, \
                 patch("builtins.print"):
                self.assertEqual(vlm.infer(args), 0)
                self.assertEqual(vlm.infer(args), 0)
                self.assertEqual(loader.call_count, 1)
            self.assertEqual(len((args.out_dir / "attempts.jsonl").read_text().splitlines()), 1)
            _, _, rows = reporting.load_run(args.out_dir)
            self.assertEqual(rows[0]["raw_response"], '{"dr_detected":0,"confidence":"low","reasoning":"fixture"}')
            args.max_new_tokens = 128
            with self.assertRaisesRegex(ValueError, "configuration/cohort changed"):
                vlm.infer(args)
            # The new family must use its own loader and retain its identity in reports.
            import transformers
            from unittest.mock import Mock
            gemma4_model = Model()
            gemma4_model.config = argparse.Namespace(model_type="gemma4_unified")
            multimodal_loader = Mock()
            multimodal_loader.from_pretrained.return_value = gemma4_model
            for name, model_id in (("gemma4", "google/gemma-4-12B"),
                                   ("gemma4-it", "google/gemma-4-12B-it"),
                                   ("muse", "meta-models/Muse-Glimmer-30B")):
                args.model, args.out_dir = name, root / name
                gemma4_model.config.model_type = "muse_glimmer" if name == "muse" else "gemma4_unified"
                multimodal_loader.reset_mock()
                with patch.object(transformers, "AutoModelForMultimodalLM", multimodal_loader, create=True), \
                     patch.object(AutoProcessor, "from_pretrained", return_value=Processor()), \
                     patch.object(AutoModelForImageTextToText, "from_pretrained") as old_loader, \
                     patch("builtins.print"):
                    self.assertEqual(vlm.infer(args), 0)
                    old_loader.assert_not_called()
                    multimodal_loader.from_pretrained.assert_called_once()
                config, _, rows = reporting.load_run(args.out_dir)
                self.assertEqual(config["model_id"], model_id)
                self.assertEqual(rows[0]["model_name"], name)
            # Fail before loading large weights when a downloaded processor lacks a template.
            args.out_dir = root / "no-template"
            processor = Processor()
            processor.chat_template = None
            multimodal_loader.reset_mock()
            with patch.object(transformers, "AutoModelForMultimodalLM", multimodal_loader, create=True), \
                 patch.object(AutoProcessor, "from_pretrained", return_value=processor):
                with self.assertRaisesRegex(ValueError, "Processor has no chat template"):
                    vlm.infer(args)
                multimodal_loader.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
