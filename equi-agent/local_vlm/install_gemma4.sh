#!/usr/bin/env bash
# Separate Gemma 4 environment; never upgrade the completed Gemma 3/MedGemma environment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3.11}"
venv="${VLM_VENV:-$HOME/.venvs/fairvision-gemma4-py311}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
  touch "$venv/.fairvision-gemma4-install"
elif [[ ! -e "$venv/.fairvision-gemma4-install" ]]; then
  echo "Choose a NEW VLM_VENV; refusing to modify an unrelated environment: $venv" >&2
  exit 1
fi
"$venv/bin/python" -m pip install 'pip==25.1.1'
"$venv/bin/python" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
"$venv/bin/python" -m pip install -r equi-agent/local_vlm/requirements-cpu.txt 'accelerate==1.9.0' 'sentencepiece==0.2.1' 'protobuf==5.29.5'
# Pin the inspected official revision that includes gemma4_unified and its auto loader.
"$venv/bin/python" -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@f348d3dcfa9317bed603e01b8ca22b11fe59615c'
"$venv/bin/python" -m pip check
"$venv/bin/python" -c 'from transformers import AutoModelForMultimodalLM, AutoProcessor, Gemma4UnifiedConfig; import torch, transformers; print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__); assert Gemma4UnifiedConfig.model_type == "gemma4_unified"'
"$venv/bin/python" -m pip freeze > "$venv/fairvision-installed-packages.txt"
echo "Gemma 4 environment: $venv"
