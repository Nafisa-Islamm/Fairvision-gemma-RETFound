"""Submit one compatible short GPU smoke test, then replace an explicitly named old job.

Run on Newton's login node. Live capacity and admission checks do not override Slurm
priority, reservations or permissions and do not guarantee an immediate allocation.
"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess


@dataclass
class Candidate:
    node: str
    partition: str
    gpu_type: str
    precision: str
    seconds: int
    rank: int


def candidates(node_text, model):
    result = []
    blocked = {"DOWN", "DRAIN", "DRAINING", "FAIL", "FAILING", "NO_RESPOND", "NOT_RESPONDING",
               "MAINT", "RESERVED", "PLANNED", "POWERED_DOWN", "POWERING_DOWN", "POWERING_UP",
               "REBOOT_REQUESTED", "REBOOT_ISSUED"}
    for line in node_text.splitlines():
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
        name, state = fields.get("NodeName"), fields.get("State", "")
        if not name or name in {"evc11", "evc26"} or any(token in state.split("+") for token in blocked):
            continue
        if not state.startswith(("IDLE", "MIXED")):
            continue
        try:
            if int(fields["CPUTot"]) - int(fields["CPUAlloc"]) < 2:
                continue
            if int(fields["RealMemory"]) - int(fields["AllocMem"]) < 32 * 1024:
                continue
            configured = re.search(r"(?:^|,)gres/gpu=(\d+)", fields.get("CfgTRES", ""))
            allocated = re.search(r"(?:^|,)gres/gpu=(\d+)", fields.get("AllocTRES", ""))
            if configured is None or int(configured.group(1)) <= (int(allocated.group(1)) if allocated else 0):
                continue
        except (KeyError, ValueError):
            continue
        gres, partitions = fields.get("Gres", ""), fields.get("Partitions", "").split(",")
        if name == "evc105" and "nvidia_h200_nvl" in gres:
            for partition in ("short", "ucfit"):
                if partition in partitions:
                    result.append(Candidate(name, partition, "nvidia_h200_nvl", "bfloat16", 1200, 0))
        elif "normal" in partitions and "nvidia_h100_pcie" in gres:
            result.append(Candidate(name, "normal", "nvidia_h100_pcie", "bfloat16", 1200, 1))
        elif model != "gemma4-it" and "normal" in partitions and "tesla_v100-pcie-32gb" in gres:
            result.append(Candidate(name, "normal", "tesla_v100-pcie-32gb", "float32", 3600, 2))
    return sorted(result, key=lambda item: (item.rank, int(re.search(r"\d+", item.node).group()),
                                           0 if item.partition == "short" else 1))


def run(command, env=None):
    return subprocess.run(command, env=env, text=True, capture_output=True)


def submit(old_job, model="gemma"):
    if not re.fullmatch(r"\d+", str(old_job)):
        raise ValueError("Use the numeric old job ID")
    for tool in ("scontrol", "sbatch", "squeue", "scancel"):
        if shutil.which(tool) is None:
            raise ValueError(f"Missing {tool}; run on the Newton login node")
    root = Path(__file__).resolve().parents[2]
    home = Path.home()
    choices = {"gemma": ("gemma-3-4b-it", "fairvision-gpu-py311"),
               "medgemma": ("medgemma-4b-it", "fairvision-gpu-py311"),
               "gemma4-it": ("gemma-4-12B-it", "fairvision-gemma4-py311")}
    if model not in choices:
        raise ValueError("Use gemma, medgemma, or gemma4-it")
    model_folder, venv_name = choices[model]
    snapshot = home / "fairvision-models" / model_folder
    python = home / ".venvs" / venv_name / "bin/python"
    manifest = home / "fairvision-routing-research/manifests-v1/training.csv"
    if not (snapshot / "config.json").is_file() or not os.access(python, os.X_OK) or not manifest.is_file():
        raise ValueError("Missing model config, existing environment or training manifest; old job retained")
    info = run(["scontrol", "--oneliner", "show", "node"])
    if info.returncode:
        raise ValueError(f"Cannot inspect nodes; old job retained.\n{info.stderr}")
    options = candidates(info.stdout, model)
    admitted = []
    for choice in options:
        family = "h200" if choice.rank == 0 else "h100" if choice.rank == 1 else "v100"
        suffix = "bf16" if choice.precision == "bfloat16" else "fp32"
        output = home / f"fairvision-routing-research/cache-{family}-{suffix}/training"
        env = dict(os.environ, REPO_ROOT=str(root), FAIRVISION_MANIFEST=str(manifest),
                   FAIRVISION_DATA_ROOT=os.environ.get("FAIRVISION_DATA_ROOT", str(home / "datasets/FairVision30K/raw")),
                   VLM_OUTPUT_ROOT=str(output), VLM_SPLIT="training", VLM_DTYPE=choice.precision,
                   VLM_VENV=str(python.parent.parent), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        job_name = f"{model}-{family}-{suffix}-smoke"
        hours, remainder = divmod(choice.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        arguments = ["--account=ytian", f"--partition={choice.partition}", f"--nodelist={choice.node}",
                     f"--gres=gpu:{choice.gpu_type}:1", "--cpus-per-task=2", "--mem=32G",
                     f"--time={hours:02}:{minutes:02}:{seconds:02}", f"--job-name={job_name}", "--export=ALL,SMOKE=1",
                     str(root / "equi-agent/local_vlm/cluster_gpu.sbatch"), model]
        print(f"Testing admission: node={choice.node}, partition={choice.partition}, GPU={choice.gpu_type}, dtype={choice.precision}", flush=True)
        check = run(["sbatch", "--test-only", *arguments], env)
        if check.returncode:
            print(check.stderr.strip() or check.stdout.strip(), flush=True)
            continue
        message = (check.stdout + "\n" + check.stderr).strip()
        print(message, flush=True)
        estimate = re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b", message)
        # Cluster-local ISO estimates can be compared without inventing a timezone.
        admitted.append((estimate.group() if estimate else "9999", choice, output, env, arguments, job_name))
    for _, choice, output, env, arguments, job_name in sorted(admitted, key=lambda item: (item[0], item[1].rank)):
        output.mkdir(parents=True, exist_ok=True)
        pointer = output / f"{model}_snapshot.txt"
        if pointer.exists() and pointer.read_text().strip() != str(snapshot):
            raise ValueError("Existing snapshot pointer differs; old job retained")
        pointer.write_text(str(snapshot) + "\n")
        accepted = run(["sbatch", "--parsable", *arguments], env)
        if accepted.returncode:
            print(accepted.stderr.strip(), flush=True)
            continue
        new_id = accepted.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", new_id):
            raise ValueError(f"Unexpected submission result {accepted.stdout!r}; old job not cancelled")
        print(f"NEW_JOB_ID={new_id}", flush=True)
        current = run(["squeue", "-h", "-j", str(old_job), "-o", "%T"])
        if current.returncode:
            raise ValueError(f"New job {new_id} submitted, but old-job status could not be checked; inspect both jobs.\n{current.stderr}")
        if current.stdout.strip():
            cancelled = run(["scancel", str(old_job)])
            if cancelled.returncode:
                raise ValueError(f"New job {new_id} submitted, but old-job cancellation failed. Check both.\n{cancelled.stderr}")
            print(f"Cancelled old job {old_job}.", flush=True)
        else:
            print(f"Old job {old_job} is no longer queued; no cancellation needed.", flush=True)
        print(f"Requested {choice.node} in {choice.partition}. Slurm still controls the start time.", flush=True)
        print(f"Log: {root / f'slurm-{job_name}-{new_id}.out'}", flush=True)
        queue = run(["squeue", "-j", new_id])
        print(queue.stdout.strip(), flush=True)
        return new_id
    raise ValueError("No replacement submitted. No eligible node passed capacity/admission checks; old job retained. Account access/priority cannot be bypassed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-job", required=True)
    parser.add_argument("--model", choices=("gemma", "medgemma", "gemma4-it"), default="gemma")
    args = parser.parse_args()
    try:
        submit(args.replace_job, args.model)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
