#!/usr/bin/env bash
# Isolated Muse runtime; preserve every existing model environment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3.11}"
venv="${VLM_VENV:-$HOME/.venvs/fairvision-muse-py311}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" TOKENIZERS_PARALLELISM=false
"$python_bin" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
  touch "$venv/.fairvision-muse-install"
elif [[ ! -e "$venv/.fairvision-muse-install" ]]; then
  echo "Choose a NEW VLM_VENV; refusing to modify an unrelated environment: $venv" >&2
  exit 1
fi
"$venv/bin/python" -m pip install 'pip==25.1.1'
"$venv/bin/python" -m pip install 'torch==2.12.0' 'torchvision==0.27.0' --index-url https://download.pytorch.org/whl/cu126
"$venv/bin/python" -m pip install -r equi-agent/local_vlm/requirements-cpu.txt 'accelerate==1.9.0' 'sentencepiece==0.2.1' 'protobuf==5.29.5'
"$venv/bin/python" -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@f348d3dcfa9317bed603e01b8ca22b11fe59615c'
"$venv/bin/python" -m pip check
"$venv/bin/python" -c 'from transformers import AutoModelForMultimodalLM, AutoProcessor, MuseGlimmerConfig; import torch, torchvision, transformers; print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__, "Transformers:", transformers.__version__); assert MuseGlimmerConfig.model_type == "muse_glimmer"'
"$venv/bin/python" -m pip freeze > "$venv/fairvision-installed-packages.txt"
echo "Muse environment: $venv"
