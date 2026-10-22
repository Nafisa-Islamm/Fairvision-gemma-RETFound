# Muse Glimmer 30B on Newton

Requested checkpoint: `meta-models/Muse-Glimmer-30B`. The user's downloaded configuration identifies `muse_glimmer`, `MuseGlimmerForConditionalGeneration`, BF16, no quantization config, and saved Transformers version `5.15.0.dev0`. That version records the checkpoint's saving environment, not a sufficient compatibility guarantee. The official Transformers revision already pinned for Gemma 4 (`f348d3dcfa9317bed603e01b8ca22b11fe59615c`) includes this architecture and maps it to `AutoModelForMultimodalLM`.

Support is added as `--model muse`, retaining the same cohort, prompts, image grids, eager attention and disabled compilation. Do not use the optional assistant/drafter, quantization or CPU offloading in the initial baseline. Record any later change as a separate experimental configuration. The processor's default reasoning settings are used; smoke testing must determine whether a 512-token output budget yields valid responses.

Local verification: the original 10 pipeline tests plus two formatting-recovery tests passed; loader-selection tests now include Muse. A separate tiny randomly initialized Muse model successfully reloaded and generated from two image inputs on CPU. This does not validate pretrained 30B weights, the actual downloaded processor, or H100 memory use.

**Image-resizing compatibility:** the first actual processor check passed but warned that torchvision 0.21 substituted BICUBIC for LANCZOS. Muse has no registered PIL processor in the inspected implementation. The separate Muse installer now pins PyTorch 2.12.0 / torchvision 0.27.0 / CUDA 12.6 to support LANCZOS on CPU. The CPU counterparts passed real LANCZOS resizing and two-image generation tests locally; the CUDA installation and pretrained GPU inference still need validation on Newton. Gemma environments are unchanged. Muse inference requires torchvision >=0.27, checks CPU preprocessing, and records the image-processing version/mode in run_config.json.

## Check storage before large downloads

On the Newton login node:

```bash
df -h "$HOME"
quota -s
du -sh "$HOME/fairvision-models/"*
du -sh "$HOME/.venvs"
```

`df` reports shared capacity; personal/group quotas can be much smaller. An empty `quota` output does not establish unlimited space. Thirty billion BF16 parameters alone are approximately 60 GB decimal (56 GiB), and the environment, partial downloads and repository variants need additional space. Inspect the actual repository file sizes before allocating that space. Do not delete existing results or model folders automatically.

Metadata-only size/license query with existing tools:

```bash
"$HOME/.venvs/fairvision-gemma4-py311/bin/python" -c 'from huggingface_hub import HfApi; i=HfApi().model_info("meta-models/Muse-Glimmer-30B",files_metadata=True); files=i.siblings; print("Revision:",i.sha); print("Reported repository GiB:",round(sum(f.size or 0 for f in files)/1024**3,2)); print("Files with unknown size:",sum(f.size is None for f in files)); print("License:",(i.card_data or {}).get("license"))'
```

The total describes all files at that revision, not guaranteed local quota consumption. Review the model card/license yourself; accessible configuration files do not establish unrestricted weight access or permitted use.

## Verify the actual processor before downloading weights

Update the feature branch and restrict threads:

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
export REPO_ROOT="$HOME/Fairvision-gemma-RETFound"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

Use the existing modern environment for this lightweight metadata check, without changing its packages:

```bash
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:20:00 "$HOME/.venvs/fairvision-gemma4-py311/bin/python" -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="meta-models/Muse-Glimmer-30B",local_dir=os.path.expanduser("~/fairvision-models/Muse-Glimmer-30B"),allow_patterns=["*.json","*.jinja","*.model","*.txt","*.tiktoken","README.md","LICENSE*"],max_workers=2))'
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00 "$HOME/.venvs/fairvision-gemma4-py311/bin/python" equi-agent/scripts/verify_muse_processor.py --model-path "$HOME/fairvision-models/Muse-Glimmer-30B"
```

Require `Two-image processor check passed`. Missing templates, unsupported configuration, authentication and dependency errors must be investigated before continuing. The image preprocessing uses two synthetic images and does not load weights.

## Isolated runtime and weights (only after checks pass and quota is sufficient)

```bash
export VLM_VENV="$HOME/.venvs/fairvision-muse-py311"
export PYTHON_BIN=/usr/bin/python3.11
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=12G --time=00:45:00 bash equi-agent/local_vlm/install_muse.sh
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00 "$VLM_VENV/bin/python" equi-agent/scripts/verify_muse_processor.py --model-path "$HOME/fairvision-models/Muse-Glimmer-30B" --require-native-lanczos
srun --account=ytian --partition=normal --ntasks=1 --cpus-per-task=2 --mem=8G --time=03:00:00 "$VLM_VENV/bin/python" -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="meta-models/Muse-Glimmer-30B",local_dir=os.path.expanduser("~/fairvision-models/Muse-Glimmer-30B"),max_workers=2))'
```

Do not continue after installation or download errors. Completed existing metadata is reused.

## Six-case H100 test

```bash
export VLM_OUTPUT_ROOT="$HOME/fairvision-vlm-results-muse"
export FAIRVISION_DATA_ROOT="$HOME/datasets/FairVision30K/raw"
mkdir -p "$VLM_OUTPUT_ROOT"
cp -n "$HOME/fairvision-vlm-results-eager/cohort.csv" "$VLM_OUTPUT_ROOT/cohort.csv"
printf '%s\n' "$HOME/fairvision-models/Muse-Glimmer-30B" > "$VLM_OUTPUT_ROOT/muse_snapshot.txt"
MUSE_TEST_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 --cpus-per-task=2 --mem=96G --time=00:45:00 --export=ALL,SMOKE=1 equi-agent/local_vlm/cluster_gpu.sbatch muse)
echo "$MUSE_TEST_JOB_ID"
```

96 GB here is host RAM, not GPU VRAM. A single H100 has about 80 GB VRAM; whether the full model plus two-image activations fits must be measured. No fit guarantee is made. If out of memory, stop and choose a larger-memory GPU, a documented multi-GPU configuration, or a separately identified quantized baseline; do not retry unchanged allocations indefinitely.

Use `squeue -u "$USER"`, the printed job number, its Slurm log, and `sacct` to confirm six predictions and zero errors before a full run. No pretrained Muse smoke test has been run by the assistant.

## Full evaluation and CPU report

After successful smoke testing, the same submission command with `SMOKE=0` and a larger time allowance evaluates all 750 cases. Keep the same code/environment and separate output root. Do not compare timings extrapolated from the tiny architecture test to the real checkpoint.

After successful full inference, restore the lightweight CPU environment and select the Muse report:

```bash
export VLM_CPU_VENV="$HOME/.venvs/fairvision-cpu-py311"
export VLM_REPORT_MODELS=muse
MUSE_REPORT_JOB_ID=$(sbatch --parsable --account=ytian --partition=normal --cpus-per-task=1 --mem=4G --time=00:10:00 --export=ALL equi-agent/local_vlm/cluster_cpu.sbatch report)
echo "$MUSE_REPORT_JOB_ID"
```

Check exit `0:0`, then read `$HOME/fairvision-vlm-results-muse/table/table1_local_models.md` and the subgroup CSV in that directory. Formatting failures are retained and must be audited rather than scored as negative or silently excluded.
