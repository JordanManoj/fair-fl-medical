#!/usr/bin/env bash
# Reproducible environment + data setup for Fed-Heart-Disease.
# Usage:  bash scripts/setup.sh [path/to/venv]
# Requires Python 3.10 or 3.11 (FLamby targets 3.10; tested here on 3.11).
set -euo pipefail

FLAMBY_COMMIT=edacf54d5211520583b0133d55ac39b6fda8324b
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$ROOT/.venv}"
PY="${PYTHON:-python3.11}"

"$PY" -m venv "$VENV"
if [ -x "$VENV/Scripts/python" ]; then VPY="$VENV/Scripts/python"; else VPY="$VENV/bin/python"; fi

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

mkdir -p "$ROOT/external"
if [ ! -d "$ROOT/external/FLamby" ]; then
  git clone https://github.com/owkin/FLamby.git "$ROOT/external/FLamby"
fi
git -C "$ROOT/external/FLamby" checkout --quiet "$FLAMBY_COMMIT"

# Only the heart extra; then re-apply our pins (FLamby's deps are unpinned).
"$VPY" -m pip install -e "$ROOT/external/FLamby[heart]"
"$VPY" -m pip install -r "$ROOT/requirements.txt"

# Download (prompts you to accept the UCI CC BY 4.0 data terms).
mkdir -p "$ROOT/data/heart"
cd "$ROOT/external/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts"
"$VPY" download.py --output-folder "$ROOT/data/heart"
