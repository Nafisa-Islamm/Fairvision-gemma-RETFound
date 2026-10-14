#!/usr/bin/env bash
# Install the GPU inference environment on the chosen cluster using Python >=3.10.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
venv="${VLM_VENV:-$repo_root/.venv}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
  touch "$venv/.retfound-vlm-install"
elif [[ ! -e "$venv/.retfound-vlm-install" ]]; then
  echo "Existing environment at $venv. Choose a NEW VLM_VENV to avoid altering it." >&2
  exit 1
fi
"$venv/bin/python" -m pip install 'pip==25.1.1'
"$venv/bin/python" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
"$venv/bin/python" -m pip install -r equi-agent/local_vlm/requirements-gpu.txt
"$venv/bin/python" -m pip check
touch "$venv/.retfound-vlm-install"
"$venv/bin/python" -c 'import torch, transformers; print("PyTorch", torch.__version__, "Transformers", transformers.__version__)'
echo "Environment: $venv"
