#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."

# ---------------- Paths ----------------
EXAMPLES_SRC="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup/examples/y_yprime_examples"
EXAMPLES_TMP="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup/examples/table1_2"
CIRCUIT_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/non-linear"
ZIP_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup"
LOOKUP_TABLE_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/precomputed_lookup_tables_ieee754_hex"

cleanup() {
  rm -rf -- "$EXAMPLES_TMP"
}
trap cleanup EXIT

# ---------------- Preset values ----------------
declare -A GELU_PRESET=(
  [TABLE_SIZE]=70
  [PRIVATE_VECTOR_SIZE]=10
  [TABLE_PRIME_SIZE]=8
  [PRIVATE_VECTOR_PRIME_SIZE]=2
)
declare -A SELU_PRESET=(
  [TABLE_SIZE]=35
  [PRIVATE_VECTOR_SIZE]=7
  [TABLE_PRIME_SIZE]=6
  [PRIVATE_VECTOR_PRIME_SIZE]=2
)
declare -A ELU_PRESET=(
  [TABLE_SIZE]=36
  [PRIVATE_VECTOR_SIZE]=6
  [TABLE_PRIME_SIZE]=7
  [PRIVATE_VECTOR_PRIME_SIZE]=2
)

echo "Running table1/2 evaluations..."

mkdir -p "$EXAMPLES_TMP"
cd "$CIRCUIT_DIR"

for act in gelu selu elu; do
  python3 "$ZIP_DIR/add_interval_index.py" \
    --pairs "$EXAMPLES_SRC/${act}_y_yprime.txt" \
    --intervals "$LOOKUP_TABLE_DIR/${act}_intervals_ieee754.txt"

  echo ""
  echo "Starting evaluations for activation: $act"

  src_file="$EXAMPLES_SRC/${act}_y_yprime.txt"
  if [[ ! -f "$src_file" ]]; then
    echo "ERROR: source pairs file not found: $src_file" >&2
    exit 1
  fi

  base_line="$(awk 'NF && $0 !~ /^[[:space:]]*#/{print; exit}' "$src_file")"
  if [[ -z "$base_line" ]]; then
    echo "ERROR: no valid (non-empty, non-comment) line found in $src_file" >&2
    exit 1
  fi

  for i in 0 4 8 12 16; do
    instances=$((1<<i))
    echo ""
    echo "Running for $act with 2^$i instances (instances=$instances):"

    tmp_pairs="$EXAMPLES_TMP/${act}_y_yprime.txt"
    awk -v line="$base_line" -v n="$instances" 'BEGIN{for(i=0;i<n;i++) print line}' > "$tmp_pairs"

    preset_name="${act^^}_PRESET"
    declare -n PRESET="$preset_name"

    python3 generate_config.py \
      --preset "$act" \
      --instances "$instances" \
      --mode "table1/2" \
      --proving false \
      --table_size "${PRESET[TABLE_SIZE]}" \
      --table_prime_size "${PRESET[TABLE_PRIME_SIZE]}" \
      --values_dir "table1_2"

    go run main.go config.go
  done
done
