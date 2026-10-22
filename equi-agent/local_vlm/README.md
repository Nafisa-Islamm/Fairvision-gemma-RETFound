# FairVision: local Gemma and MedGemma baselines

Run **google/gemma-3-4b-it** and **google/medgemma-4b-it** locally on the same
750 FairVision test patients (250 glaucoma, 250 AMD, 250 DR). These are standalone
multimodal baselines using SLO, an eight-slice OCT grid and demographic metadata.
No existing retinal foundation model is retrained and no paid inference API is used.

Start with **[the Newton/Stokes step-by-step guide](STOKES_NEWTON.md)**.
For **Gemma 4 12B**, use **[the separate Gemma 4 Newton guide](GEMMA4_NEWTON.md)**
and `install_gemma4.sh`; keep the original model environments unchanged.
It gives the clone, installation, dataset check, model-path setup, GPU smoke/full
jobs, resume and CPU reporting commands. The selected dataset root is configured
by FAIRVISION_DATA_ROOT; Amd/Dr/Glaucoma and case variations of Test are supported.

## Files

- `../scripts/fairvision_local_vlm.py`: manifest recovery, image check, optional
  licensed model download, offline inference, raw responses and resumable results.
- `../scripts/report_fairvision_local_vlm.py`: complete-cohort macro-F1, worst-group
  macro-F1, sensitivity, specificity and balanced accuracy; CSV/Markdown/LaTeX export.
- `install_cpu.sh`: lightweight Stokes CPU environment for checks and reporting.
- `install.sh`: separate Newton GPU environment (PyTorch 2.6 / CUDA 12.4,
  Transformers 4.53.3). Run installation on a CPU allocation with network access.
- `cluster_cpu.sbatch`, `cluster_gpu.sbatch`: Slurm templates. Supply the real
  partition and any required account on each cluster; neither is guessed.
  GPU jobs enter an `srun` task step and print node/device diagnostics before
  checking PyTorch CUDA access. They stop before inference if no GPU is visible.
- `../tests/test_fairvision_local_vlm.py`: ten targeted regression tests.

The original paper cohort CSV contains one glaucoma label of -1. Manifest
preparation writes a new copy with a correction to 0 supported by the tracked
source metadata and records it in an audit. It leaves the original CSV unchanged.
Expected class counts: glaucoma 124/126, AMD 125/125, DR 125/125.

Missing/invalid model predictions remain explicit errors; complete headline
metrics are withheld for incomplete cohorts. Worst-group reporting uses marginal
race, ethnicity, sex and age groups (<50, 50–69, >=70), excludes unknown groups,
and uses fixed labels [0,1] and zero_division=0. All observed groups qualify by
default, including single-class groups; the support policy is recorded/exported.
The paper does not fully specify a minimum subgroup support cutoff. Compare
existing rows under the same scoring convention.

Models need their Google licenses/access accepted before the optional download.
Already downloaded local folders are supported. Runtime inference is offline;
credentials, model weights, clinical images and generated results are not added
by this change. Keep data and result files in appropriate cluster storage.

## Validation

Ten targeted tests passed. A tiny randomly initialized Gemma3 model completed a
CPU multimodal generation check with the pinned libraries. Full pretrained
inference, actual VRAM usage, the clinical NPZ files and Newton scheduling still
need validation using the GPU smoke jobs. New model scores come from your runs;
the code does not reproduce another model's numerical results by assumption.

After GPU dependencies are installed, run from the repository root:

```bash
"$VLM_VENV/bin/python" -m unittest discover -s equi-agent/tests -p test_fairvision_local_vlm.py -v
```
# Cluster compilation

Generation explicitly sets `disable_compile=True` so Gemma runs without Python development headers or Triton compilation. This setting is recorded in each run configuration. After updating inference code, use a new output directory because saved runs include the script hash; keep the cohort and local model snapshot paths unchanged.

