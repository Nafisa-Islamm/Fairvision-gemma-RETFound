import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fairroute_grpo as grpo
import fairroute_data as data
import fairvision_local_vlm as vlm
from make_fairroute_manifests import make_manifests


def fixture(split):
    cases = []
    for task in vlm.TASKS:
        for race in ("asian", "black"):
            for j in range(6):
                label = j % 2
                case_id = f"{split}-{task}-{race}-{j}"
                predictions = [label if race == "asian" else 1 - label,
                               label if race == "black" else 1 - label, 0]
                cases.append(dict(task=task, case_id=case_id, patient_id=case_id, y_true=label,
                                  race=race, age=40 + j, sex_gender="female", ethnicity="non-hispanic",
                                  npz_sha256=hashlib.sha256(case_id.encode()).hexdigest(), embedding=[],
                                  expert_preds=predictions, expert_confidence=["high", "moderate", "low"]))
    return dict(format_version=1, split=split, experts=["gemma", "medgemma", "gemma4-it"], cases=cases,
                provenance=dict(identity="patient_map", prompts_sha256=vlm.digest(vlm.PROMPTS), oct_slices=8))


class GRPORoutingTests(unittest.TestCase):
    def test_group_relative_advantages_and_clipping(self):
        rewards = torch.tensor([[1., 0., 1., 0.], [1., 1., 1., 1.]])
        advantages = grpo.group_advantages(rewards)
        self.assertTrue(torch.allclose(advantages[0], torch.tensor([1., -1., 1., -1.])))
        self.assertTrue(torch.equal(advantages[1], torch.zeros(4)))
        # Demonstrates why a broadcast demographic penalty cannot train fairness via GRPO normalization.
        self.assertTrue(torch.allclose(advantages, grpo.group_advantages(rewards - .4)))
        logits = torch.tensor([[np.log(.9), np.log(.1)]], requires_grad=True)
        loss = grpo.clipped_grpo_loss(logits, torch.tensor([[0]]), torch.tensor([[np.log(.5)]]),
                                     torch.ones(1, 1), torch.ones(1), .2)
        self.assertAlmostEqual(float(loss.detach()), -1.2)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_fairness_has_gradient_and_missing_groups_are_not_zero_fnr(self):
        router = torch.nn.Linear(1, 2)
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.tensor([np.log(3.), 0.]))
        features = torch.ones(4, 1)
        predictions = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        pools = {"glaucoma": [("asian", np.array([0, 1])), ("black", np.array([2, 3]))]}
        optimizer = torch.optim.SGD(router.parameters(), lr=.1)
        initial = grpo.fairness_loss(router, features, predictions, pools, .15, np.random.default_rng(1), 2)
        initial.backward()
        self.assertGreater(float(router.bias.grad.abs().sum()), 0.)
        optimizer.step()
        final = grpo.fairness_loss(router, features, predictions, pools, .15, np.random.default_rng(1), 2)
        self.assertLess(float(final.detach()), float(initial.detach()))
        supported, counts = grpo.fairness_pools(fixture("train"), 2)
        self.assertEqual(counts["glaucoma"]["white"], 0)
        self.assertEqual(len(supported["glaucoma"]), 2)

    def test_features_do_not_use_labels_or_unqueried_experts(self):
        original = fixture("train")
        modified = copy.deepcopy(original)
        for case in modified["cases"]:
            case["y_true"] = 1 - case["y_true"]
        np.testing.assert_array_equal(data.feature_matrix(original), data.feature_matrix(modified))
        for case in modified["cases"]:
            case["expert_preds"] = [1 - p for p in case["expert_preds"]]
        np.testing.assert_array_equal(data.feature_matrix(original, mode="prequery"), data.feature_matrix(modified, mode="prequery"))
        self.assertFalse(np.array_equal(data.feature_matrix(original), data.feature_matrix(modified)))
        original["cases"][0]["race"] = "unknown"
        matrix = data.feature_matrix(original, race_aware=True, mode="prequery")
        self.assertEqual(matrix[0, -1], 1.)
        self.assertEqual(matrix[0, -2], 0.)

    def test_split_and_missing_expert_guards(self):
        training, validation = fixture("train"), fixture("validation")
        data.check_separation(training, validation)
        validation["cases"][0]["patient_id"] = training["cases"][0]["patient_id"]
        with self.assertRaisesRegex(ValueError, "overlapping patient_id"):
            data.check_separation(training, validation)
        validation = fixture("validation")
        validation["cases"][0]["npz_sha256"] = training["cases"][0]["npz_sha256"]
        with self.assertRaisesRegex(ValueError, "overlapping npz_sha256"):
            data.check_separation(training, validation)
        training["cases"][0]["expert_preds"].pop()
        with self.assertRaisesRegex(ValueError, "Missing expert"):
            data.validate_bundle(training)
        with self.assertRaisesRegex(ValueError, "Expected train"):
            data.validate_bundle(fixture("test"), "train")

    def test_existing_cache_adapter_rejects_relabeling_test_as_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = fixture("test")
            cohort = [{k: v for k, v in c.items() if k not in ("expert_preds", "expert_confidence", "embedding")}
                      for c in cache["cases"]]
            paths = []
            for e, name in enumerate(cache["experts"]):
                path = root / name
                config = dict(model=name, cohort_sha256=vlm.digest(cohort), prompts=vlm.PROMPTS, oct_slices=8)
                vlm.atomic_json(path / "cohort.json", cohort)
                vlm.atomic_json(path / "run_config.json", config)
                rows = [dict(**case, model_name=name, run_fingerprint=vlm.digest(config), status="ok",
                             y_pred=source["expert_preds"][e], confidence=source["expert_confidence"][e])
                        for case, source in zip(cohort, cache["cases"])]
                vlm.write_csv(path / "predictions.csv", rows)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "cannot be called training"):
                data.prepare_bundle(paths, "train", root / "train.json", confirm_one_case_per_patient=True)
            result = data.prepare_bundle(paths, "test", root / "test.json", confirm_one_case_per_patient=True)
            self.assertEqual(result["cases"][0]["expert_preds"], cache["cases"][0]["expert_preds"])
            self.assertEqual(result["cases"][0]["embedding"], [])

    def test_manifest_builder_preserves_source_splits_and_default_test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("builtins.print"):
                make_manifests(vlm.ROOT / "OphthalmicAgent/data", root / "manifests", 5)
            train = vlm.load_cohort(root / "manifests/training.csv", expected_split="training")
            validation = vlm.load_cohort(root / "manifests/validation.csv", expected_split="validation")
            self.assertEqual(len(train), 15)
            self.assertFalse({(r["task"], r["case_id"]) for r in train} & {(r["task"], r["case_id"]) for r in validation})
            with self.assertRaisesRegex(ValueError, "non-test"):
                vlm.load_cohort(root / "manifests/training.csv")
            directory = root / "raw/Amd/Training"
            directory.mkdir(parents=True)
            (directory / "case.npz").write_bytes(b"fixture")
            path = vlm.resolve_npz("AMD/training/case.npz", root / "manifest.csv", root / "raw", "training")
            self.assertEqual(path, directory / "case.npz")
            with self.assertRaisesRegex(ValueError, "different split"):
                vlm.resolve_npz("AMD/test/case.npz", root / "manifest.csv", root / "raw", "training")

    def test_small_balanced_pilot_uses_real_cells_without_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "repeat"):
                with patch("builtins.print"):
                    make_manifests(vlm.ROOT / "OphthalmicAgent/data", root / name,
                                   training_per_cell=30, validation_per_cell=20)
            training = vlm.read_csv(root / "first/training.csv")
            validation = vlm.read_csv(root / "first/validation.csv")
            self.assertEqual((len(training), len(validation)), (540, 320))
            for split, rows in (("training", training), ("validation", validation)):
                self.assertEqual(len(rows), len({(r["task"], r["case_id"]) for r in rows}))
                self.assertEqual({r["split"] for r in rows}, {split})
                self.assertEqual((root / f"first/{split}.csv").read_bytes(),
                                 (root / f"repeat/{split}.csv").read_bytes())
            self.assertFalse({(r["task"], r["case_id"]) for r in training} &
                             {(r["task"], r["case_id"]) for r in validation})
            config = json.loads((root / "first/manifest_config.json").read_text())
            rare = next(r for r in config["cell_selection"] if r["split"] == "validation" and
                        r["task"] == "dr" and r["race"] == "asian" and r["label"] == 1)
            self.assertEqual((rare["available"], rare["selected"]), (2, 2))
            self.assertTrue(rare["below_target"])
            self.assertTrue(all(r["positive"] == 30 and r["negative"] == 30
                                for r in config["race_class_support"]["training"]))
            with self.assertRaisesRegex(ValueError, "Choose uniform"):
                make_manifests(vlm.ROOT / "OphthalmicAgent/data", root / "bad", max_per_task=10,
                               training_per_cell=30, validation_per_cell=20)

    def test_check_progress_does_not_change_cohort_or_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(2):
                name = f"case-{i}.npz"
                np.savez(root / name, slo_fundus=np.zeros((16, 16, 3), dtype=np.uint8),
                         oct_bscans=np.zeros((8, 16, 16), dtype=np.uint8))
                rows.append(dict(task="dr", filename=name, y_true=i, split="training"))
            vlm.write_csv(root / "manifest.csv", rows)
            args = argparse.Namespace(manifest=root / "manifest.csv", data_root=root, tasks=["dr"],
                                      limit_per_task=0, split="training", oct_slices=8, progress_every=1)
            with patch("builtins.print") as progress:
                cases = vlm.prepare_cases(args)
            printed = "\n".join(call.args[0] for call in progress.call_args_list)
            self.assertIn("Checked 1/2", printed)
            self.assertIn("Checked 2/2", printed)
            args.progress_every = 0
            with patch("builtins.print") as quiet:
                self.assertEqual(cases, vlm.prepare_cases(args))
                quiet.assert_not_called()

    def test_grpo_and_supervised_train_save_and_evaluate_without_test_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation", "test"):
                vlm.atomic_json(root / f"{split}.json", fixture(split))
            for algorithm in ("grpo", "supervised"):
                args = argparse.Namespace(train=root / "train.json", validation=root / "validation.json",
                    out_dir=root / algorithm, algorithm=algorithm, mode="arbitration", race_aware=False,
                    epochs=2, batch_size=18, hidden_dim=8, group_size=4, inner_updates=2,
                    learning_rate=.01, epsilon_clip=.2, kl_weight=.02, entropy_weight=.01,
                    fairness_weight=.5, fnr_gap_threshold=.15, minimum_positive=2, fair_samples=2,
                    tp_bonus=.3, fn_penalty=.5, cost_weight=0., expert_costs=None, seed=2026, device="cpu")
                with patch("builtins.print"):
                    grpo.train(args)
                checkpoint = root / algorithm / "best_router.pt"
                payload = torch.load(checkpoint, weights_only=True)
                self.assertEqual(payload["config"]["algorithm"], algorithm)
                self.assertNotIn("test_bundle_sha256", payload["config"])
                history = json.loads((root / algorithm / "history.json").read_text())
                self.assertEqual(len(history), 2)
                self.assertTrue(all(np.isfinite(row["loss"]) for row in history))
                evaluation = argparse.Namespace(checkpoint=checkpoint, data=root / "test.json",
                                                out_dir=root / f"{algorithm}-test", device="cpu", verification="disagreement")
                with patch("builtins.print"):
                    grpo.evaluate(evaluation)
                rows = vlm.read_csv(evaluation.out_dir / "routing_decisions.csv")
                self.assertEqual(len(rows), len(fixture("test")["cases"]))
                self.assertTrue(all(row["selected_expert"] in payload["config"]["experts"] for row in rows))
                selective = vlm.read_csv(evaluation.out_dir / "selective_metrics.csv")
                self.assertTrue(all(float(row["coverage"]) == 0. for row in selective))
                self.assertTrue(all(row["f1_macro"] == "" for row in selective))
                with self.assertRaisesRegex(ValueError, "leakage"):
                    grpo.check_test_identity(payload, fixture("train"))
                args.out_dir = root / f"{algorithm}-bad-cost"
                args.cost_weight = .1
                with self.assertRaisesRegex(ValueError, "Arbitration already uses all"):
                    grpo.train(args)


if __name__ == "__main__":
    unittest.main()
