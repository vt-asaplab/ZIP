#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

CIRCUIT_DIR="../src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/non-linear"
cd "$CIRCUIT_DIR"

echo "Running fig1 evaluations..."

for act in gelu selu elu; do
    echo ""
    echo "Starting evaluations for activation: $act"
    for ((i=0; i<=14; i++)); do
        instances=$((2**i))
        echo ""
        echo "Running for $act with 2^$i instances:"
        python3 generate_config.py --preset "$act" --instances "$instances" --mode fig1 --proving false
        wait
        go run main.go config.go
    done
done

