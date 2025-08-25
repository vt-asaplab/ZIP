#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."

SRC="$REPO_ROOT/src/piecewise_polynomial_approximation/precomputed_lookup_tables_ieee754/elu_approx.py"
DST_DIR="$REPO_ROOT/src/CNN/UTKFace_MAE"

mkdir -p "$DST_DIR"
cp -f "$SRC" "$DST_DIR/elu_approx.py"

cd "$DST_DIR"

if [[ -d "$DST_DIR/Dataset/utkface_aligned_cropped/crop_part1" ]]; then
  echo "Dataset already extracted. Skipping unzip."
elif [[ -f "$DST_DIR/Dataset.zip" ]]; then
  unzip -q "$DST_DIR/Dataset.zip" -d "$DST_DIR"
else
  echo "ERROR: Dataset.zip not found at $DST_DIR"; exit 1
fi

echo ""
echo "Running with native ELU activation"
python CNN_MAE.py 0 110
echo "Finished run"

echo ""
echo "Running with piecewise polynomial approximation of ELU activation"
python CNN_MAE.py 1 110
echo "Finished run"
