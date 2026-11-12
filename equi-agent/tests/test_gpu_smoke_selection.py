import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import submit_available_gpu_smoke as selector


def node(name, gpu, state="MIXED", partitions="normal", memory=40000):
    return (f"NodeName={name} State={state} Gres=gpu:{gpu}:2(S:0-1) Partitions={partitions} "
            f"CPUTot=32 CPUAlloc=25 RealMemory=190884 AllocMem={190884-memory} "
            "CfgTRES=cpu=32,mem=190884M,gres/gpu=2 AllocTRES=cpu=25,mem=150884M,gres/gpu=1")


class GPUSmokeSelectionTests(unittest.TestCase):
    def test_capacity_health_and_precision_selection(self):
        text = "\n".join([
            node("evc105", "nvidia_h200_nvl", "IDLE", "short,ucfit"),
            node("evc38", "nvidia_h100_pcie"), node("evc15", "tesla_v100-pcie-32gb"),
            node("evc11", "tesla_v100-pcie-32gb", "IDLE"),
            node("evc9", "tesla_v100-pcie-16gb", "IDLE"),
            node("evc21", "tesla_v100-pcie-32gb", "MIXED+PLANNED"),
            node("evc40", "nvidia_h100_pcie", memory=30000)])
        chosen = selector.candidates(text, "gemma")
        self.assertEqual([(c.node, c.partition) for c in chosen],
                         [("evc105", "short"), ("evc105", "ucfit"), ("evc38", "normal"), ("evc15", "normal")])
        self.assertEqual([c.precision for c in chosen], ["bfloat16"] * 3 + ["float32"])
        self.assertFalse(any(c.node == "evc15" for c in selector.candidates(text, "gemma4-it")))

    def fixture(self, root):
        home, bins = root / "home", root / "bin"
        bins.mkdir()
        for path in (home / "fairvision-models/gemma-3-4b-it/config.json",
                     home / "fairvision-routing-research/manifests-v1/training.csv"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        python = home / ".venvs/fairvision-gpu-py311/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nexit 0\n")
        python.chmod(0o755)
        nodes = root / "nodes.txt"
        nodes.write_text("\n".join([node("evc105", "nvidia_h200_nvl", "IDLE", "short,ucfit"),
                                   node("evc38", "nvidia_h100_pcie")]))
        body = '''import json,os,sys
from pathlib import Path
tool=Path(sys.argv[0]).name
args=sys.argv[1:]
with Path(os.environ["FAKE_LOG"]).open("a") as f:
 f.write(json.dumps(dict(tool=tool,args=args,dtype=os.environ.get("VLM_DTYPE")))+"\\n")
mode=os.environ["FAKE_MODE"]
if tool=="scontrol": print(Path(os.environ["FAKE_NODES"]).read_text())
elif tool=="squeue": print("PENDING")
elif tool=="scancel": pass
elif "--test-only" in args:
 partition=next(a for a in args if a.startswith("--partition="))
 if mode=="deny_all" or (mode=="ucfit" and partition!="--partition=ucfit") or (mode=="normal" and partition!="--partition=normal"):
  print("Invalid account or account/partition combination",file=sys.stderr);sys.exit(1)
 if mode=="estimate":
  stamp="2026-11-12T01:00:00" if partition=="--partition=normal" else "2026-11-13T01:00:00"
  print("Job to start at "+stamp)
 else: print("Admission check passed")
elif mode=="submit_fail": print("Submission failed",file=sys.stderr);sys.exit(1)
else: print("1200999")
'''
        for tool in ("scontrol", "sbatch", "squeue", "scancel"):
            path = bins / tool
            path.write_text(f"#!{sys.executable}\n" + body)
            path.chmod(0o755)
        return home, dict(PATH=str(bins) + os.pathsep + os.environ["PATH"], FAKE_LOG=str(root / "calls.jsonl"),
                          FAKE_NODES=str(nodes), FAKE_MODE="ucfit")

    def test_permission_fallback_and_cancel_only_after_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home, env = self.fixture(root)
            with patch.object(selector.Path, "home", return_value=home), patch.dict(os.environ, env), patch("builtins.print"):
                self.assertEqual(selector.submit("1200227"), "1200999")
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            submitted = next(i for i, c in enumerate(calls) if c["tool"] == "sbatch" and "--parsable" in c["args"])
            cancelled = next(i for i, c in enumerate(calls) if c["tool"] == "scancel")
            self.assertGreater(cancelled, submitted)
            self.assertEqual(calls[cancelled]["args"], ["1200227"])
            self.assertIn("--partition=ucfit", calls[submitted]["args"])
            self.assertIn("--nodelist=evc105", calls[submitted]["args"])
            self.assertEqual(calls[submitted]["dtype"], "bfloat16")

    def test_denied_and_failed_submissions_preserve_old_job(self):
        for mode in ("deny_all", "submit_fail"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                home, env = self.fixture(root)
                env["FAKE_MODE"] = mode
                with patch.object(selector.Path, "home", return_value=home), patch.dict(os.environ, env), patch("builtins.print"):
                    with self.assertRaisesRegex(ValueError, "old job retained"):
                        selector.submit("1200227")
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                self.assertFalse(any(call["tool"] == "scancel" for call in calls))

    def test_normal_partition_fallback_uses_native_bf16_on_h100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home, env = self.fixture(root)
            env["FAKE_MODE"] = "normal"
            with patch.object(selector.Path, "home", return_value=home), patch.dict(os.environ, env), patch("builtins.print"):
                selector.submit("1200227")
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            submitted = next(c for c in calls if c["tool"] == "sbatch" and "--parsable" in c["args"])
            self.assertIn("--nodelist=evc38", submitted["args"])
            self.assertEqual(submitted["dtype"], "bfloat16")

    def test_earlier_scheduler_estimate_overrides_gpu_family_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home, env = self.fixture(root)
            env["FAKE_MODE"] = "estimate"
            with patch.object(selector.Path, "home", return_value=home), patch.dict(os.environ, env), patch("builtins.print"):
                selector.submit("1200227")
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            submitted = next(c for c in calls if c["tool"] == "sbatch" and "--parsable" in c["args"])
            self.assertIn("--nodelist=evc38", submitted["args"])


if __name__ == "__main__":
    unittest.main()
