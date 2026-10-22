import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fairvision_local_vlm as vlm
import recover_fairvision_json as recovery
import report_fairvision_local_vlm as reporting


class FormattingRecoveryTests(unittest.TestCase):
    def test_only_quotes_confidence_and_keeps_diagnosis(self):
        raw = '```json\n{"amd_detected":1,"confidence": high,"reasoning":"fixture"}\n```'
        repaired, parsed = recovery.quote_confidence(raw, "amd")
        self.assertEqual(parsed, (1, "high", "fixture"))
        self.assertIn('"confidence": "high"', repaired)
        for invalid in (
            '{"amd_detected": true,"confidence": high,"reasoning":"fixture"}',
            '{"amd_detected":1,"confidence": uncertain,"reasoning":"fixture"}',
            '{"amd_detected":1,"confidence": high,"reasoning": broken}',
            '{"amd_detected":1,"confidence":"high","reasoning":"fixture"}',
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recovery.quote_confidence(invalid, "amd")

    def test_derived_report_preserves_raw_source_and_checks_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "source", root / "recovered"
            source.mkdir()
            cohort = [dict(task=task, case_id=str(i), y_true=0, race="asian",
                           ethnicity="non-hispanic", sex_gender="female", age_group="older")
                      for i, task in enumerate(("amd", "dr", "dr"))]
            config = dict(model="gemma4-it", cohort_sha256=vlm.digest(cohort),
                          oct_slices=8, prompts=vlm.PROMPTS)
            rows = []
            for i, case in enumerate(cohort):
                pred = 1 if case["task"] == "amd" else 0
                raw = ('{"%s_detected":%s,"confidence": high,"reasoning":"fixture"}'
                       % (case["task"], pred))
                if i == 2:
                    raw = raw.replace(': high', ': "high"')
                rows.append(dict(**case, model_name="gemma4-it", run_fingerprint=vlm.digest(config),
                                 raw_response=raw, status="ok" if i == 2 else "error",
                                 error="" if i == 2 else "JSONDecodeError", y_pred=0 if i == 2 else "",
                                 confidence="high" if i == 2 else "", reasoning="fixture" if i == 2 else "",
                                 elapsed_seconds=1.2))
            vlm.atomic_json(source / "cohort.json", cohort)
            vlm.atomic_json(source / "run_config.json", config)
            vlm.write_csv(source / "predictions.csv", rows)
            original = {p.name: p.read_bytes() for p in source.iterdir()}
            with patch("builtins.print"):
                self.assertEqual(recovery.recover(source, destination), 0)
            self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})
            _, _, repaired = reporting.load_run(destination)
            self.assertEqual([r["status"] for r in repaired], ["ok"] * 3)
            self.assertEqual([r["y_pred"] for r in repaired], ["1", "0", "0"])
            self.assertEqual([r["raw_response"] for r in repaired], [r["raw_response"] for r in rows])
            audit = json.loads((destination / "recovery_audit.json").read_text())
            self.assertEqual(len(audit["repairs"]), 2)
            with patch("builtins.print"):
                self.assertEqual(reporting.report(argparse.Namespace(
                    run_dirs=[destination], out_dir=root / "table", min_group_size=1,
                    allow_small_cohort=True)), 0)
            self.assertIn("Gemma 4 12B IT", (root / "table/table1_local_models.md").read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                recovery.recover(source, destination)
            with self.assertRaisesRegex(ValueError, "NEW sibling"):
                recovery.recover(source, source)
            (destination / "recovery_audit.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "audit hash mismatch"):
                reporting.load_run(destination)


if __name__ == "__main__":
    unittest.main()
