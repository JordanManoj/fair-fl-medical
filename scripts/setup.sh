#!/usr/bin/env bash
# Reproducible environment + data setup for Fed-Heart-Disease.
#
# Usage:  bash scripts/setup.sh [path/to/venv]
#   PYTHON=/path/to/python3.11   interpreter to build the venv with (default: python3.11)
#   HEART_DATA=/path/to/folder   reuse an existing download instead of downloading
#
# Requires Python 3.10 or 3.11 (FLamby targets 3.10; tested here on 3.11).
set -euo pipefail

FLAMBY_COMMIT=edacf54d5211520583b0133d55ac39b6fda8324b
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Work from the repo root with relative paths: Git Bash on Windows reports
# POSIX-style absolute paths that Windows pip cannot parse.
cd "$ROOT"
VENV="${1:-.venv}"
PY="${PYTHON:-python3.11}"

"$PY" -m venv "$VENV"
if [ -x "$VENV/Scripts/python" ]; then VPY="$VENV/Scripts/python"; else VPY="$VENV/bin/python"; fi

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

mkdir -p external
if [ ! -d external/FLamby ]; then
  git clone https://github.com/owkin/FLamby.git external/FLamby
fi
git -C external/FLamby checkout --quiet "$FLAMBY_COMMIT"

# Only the heart extra; then re-apply our pins (FLamby's deps are unpinned).
"$VPY" -m pip install -e "./external/FLamby[heart]"
"$VPY" -m pip install -r requirements.txt

if [ -n "${HEART_DATA:-}" ]; then
  "$VPY" scripts/register_data.py "$HEART_DATA"
else
  # Download (prompts you to accept the UCI CC BY 4.0 data terms).
  mkdir -p data/heart
  DATA_DIR="$("$VPY" -c 'import os; print(os.path.abspath("data/heart"))')"
  VPY_ABS="$("$VPY" -c 'import sys; print(sys.executable)')"
  cd external/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
  "$VPY_ABS" download.py --output-folder "$DATA_DIR"
fi

echo "Done. Activate the venv and run:  python -m pytest tests && python scripts/run_fedavg.py"
