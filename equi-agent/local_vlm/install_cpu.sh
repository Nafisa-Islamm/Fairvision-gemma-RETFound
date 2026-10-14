#!/usr/bin/env bash
# Lightweight CPU environment for cohort checks and table reporting on Stokes.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
venv="${VLM_CPU_VENV:-$repo_root/.venv-cpu}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
  touch "$venv/.retfound-vlm-cpu-install"
elif [[ ! -e "$venv/.retfound-vlm-cpu-install" ]]; then
  echo "Existing environment at $venv. Choose a NEW VLM_CPU_VENV to preserve it." >&2
  exit 1
fi
"$venv/bin/python" -m pip install 'pip==25.1.1'
"$venv/bin/python" -m pip install -r equi-agent/local_vlm/requirements-cpu.txt
"$venv/bin/python" -m pip check
echo "CPU environment: $venv"
