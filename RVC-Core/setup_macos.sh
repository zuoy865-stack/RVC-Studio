#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
if [[ "$(uname -m)" != "arm64" ]]; then
  print -u2 "RVC-Core requires a native arm64 shell, not Rosetta."
  exit 2
fi

mkdir -p runtime
python3 -m venv runtime/python-env
runtime/python-env/bin/python -m pip install --upgrade pip setuptools wheel
runtime/python-env/bin/python -m pip install -r requirements-macos.txt
runtime/python-env/bin/python -m rvc_core.server --precision float32 </dev/null
