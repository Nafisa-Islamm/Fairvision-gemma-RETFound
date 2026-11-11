import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_expert_routing as audit
import fairvision_local_vlm as vlm


class ExpertAuditTests(unittest.TestCase):
    def fixture(self, root):
        cohort = [dict(task="glaucoma", case_id=str(i), y_true=i % 2,
                       race="asian" if i < 4 else "black", ethnicity="non-hispanic",
                       sex_gender="female", age_group="older") for i in range(8)]
        paths = []
        for name, pred in (("gemma", [0, 1, 0, 1, 1, 0, 1, 0]),
                           ("medgemma", [1, 0, 1, 0, 0, 1, 0, 1]),
                           ("gemma4-it", [0, 1, 1, 1, 0, 1, 0, 0])):
            path = root / name
            path.mkdir()
            config = dict(model=name, cohort_sha256=vlm.digest(cohort), oct_slices=8, prompts=vlm.PROMPTS)
            rows = [dict(**case, model_name=name, run_fingerprint=vlm.digest(config),
                         status="ok", y_pred=p) for case, p in zip(cohort, pred)]
            vlm.atomic_json(path / "run_config.json", config)
            vlm.atomic_json(path / "cohort.json", cohort)
            vlm.write_csv(path / "predictions.csv", rows)
            paths.append(path)
        return paths

    def test_metrics_match_fixed_class_macro_f1_and_missing_support(self):
        truth = np.asarray([0, 0, 1, 1])
        prediction = np.asarray([0, 1, 0, 0])
        result = audit.scores(truth, prediction)
        self.assertEqual(result["f1_macro"], f1_score(truth, prediction, labels=[0, 1], average="macro"))
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 2, 0))
        single = audit.scores(np.asarray([0, 0]), np.asarray([0, 0]))
        self.assertEqual(single["f1_macro"], .5)
        self.assertIsNone(single["sensitivity"])
        self.assertIsNone(audit.scores([], [])["f1_macro"])
        summary, _ = audit.racial_summary(truth, prediction, np.asarray(["unknown", "unknown", "asian", "asian"]))
        self.assertEqual(summary["unknown_race_cases"], 2)
        self.assertIsNone(summary["race_f1_gap"])

    def test_end_to_end_majority_pairing_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root)
            before = {str(p): p.read_bytes() for path in paths for p in path.iterdir()}
            args = argparse.Namespace(run_dirs=paths, out_dir=root / "audit", bootstrap=30,
                                      seed=2026, support_warning=20, allow_small_cohort=True)
            with patch("builtins.print"):
                audit.audit(args)
            self.assertEqual(before, {str(p): p.read_bytes() for path in paths for p in path.iterdir()})
            rows = vlm.read_csv(args.out_dir / "overall_and_race_gaps.csv")
            majority = next(row for row in rows if row["method"] == "majority_vote")
            # The first two experts always disagree, so the third determines every vote.
            third = next(row for row in rows if row["method"] == "gemma4-it")
            self.assertEqual(majority["f1_macro"], third["f1_macro"])
            pair = vlm.read_csv(args.out_dir / "expert_disagreement.csv")[0]
            self.assertEqual(int(pair["disagreements"]), 8)
            self.assertEqual(sum(int(pair[k]) for k in ("both_correct", "only_left_correct", "only_right_correct", "both_wrong")), 8)
            selective = vlm.read_csv(args.out_dir / "selective_coverage.csv")
            self.assertEqual(float(selective[0]["coverage"]), 0.)
            self.assertEqual(selective[0]["f1_macro"], "")
            intervals = vlm.read_csv(args.out_dir / "confidence_intervals.csv")
            self.assertTrue(all(int(row["valid_bootstrap_replicates"]) == 30 for row in intervals))
            pair_diffs = vlm.read_csv(args.out_dir / "paired_differences.csv")
            same = next(r for r in pair_diffs if r["left"] == "gemma4-it" and r["right"] == "majority_vote" and r["metric"] == "f1_macro")
            self.assertEqual((float(same["ci_lower"]), float(same["ci_upper"])), (0., 0.))
            config = json.loads((args.out_dir / "audit_config.json").read_text())
            self.assertEqual(config["stage"], "exploratory_test_audit_no_training")
            with self.assertRaisesRegex(ValueError, "NEW output"):
                audit.audit(args)

    def test_missing_failed_and_mismatched_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "250 cases"):
                audit.align_runs(paths)
            with self.assertRaisesRegex(ValueError, "distinct expert"):
                audit.align_runs([paths[0], paths[0]], True)
            original = vlm.read_csv(paths[0] / "predictions.csv")
            vlm.write_csv(paths[0] / "predictions.csv", original[:-1])
            with self.assertRaisesRegex(ValueError, "Missing or failed"):
                audit.align_runs(paths, True)
            original[0]["status"] = "error"
            vlm.write_csv(paths[0] / "predictions.csv", original)
            with self.assertRaisesRegex(ValueError, "Missing or failed"):
                audit.align_runs(paths, True)


if __name__ == "__main__":
    unittest.main()
