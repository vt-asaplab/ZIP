#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for act in elu selu gelu softmax layernorm; do
  echo
  echo "Computing piecewise polynomial approximation for activation: $act"
  python3 -W ignore ../src/piecewise_polynomial_approximation/approx.py "$act"
  python3 ../src/piecewise_polynomial_approximation/extract.py "$act"
  rm -f -- "../src/piecewise_polynomial_approximation/${act}_approx.py"
done

src="../src/piecewise_polynomial_approximation/precomputed_lookup_tables_ieee754_hex"
dst="../src/proof_generation/ZIP_proof_generation"

mkdir -p -- "$dst"
rm -rf -- "$dst/precomputed_lookup_tables_ieee754_hex"
mv -- "$src" "$dst/"
