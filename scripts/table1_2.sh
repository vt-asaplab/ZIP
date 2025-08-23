#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

CIRCUIT_DIR="../src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/non-linear"
cd "$CIRCUIT_DIR"

echo "Running table1/2 evaluations..."

for act in gelu selu elu; do
    echo ""
    echo "Starting evaluations for activation: $act"
    for i in 0 4 8 12 16; do
        instances=$((2**i))
        echo ""
        echo "Running for $act with 2^$i instances:"
        python3 generate_config.py --preset "$act" --instances "$instances" --mode table1/2 --proving false
        wait
        go run main.go config.go
    done
done