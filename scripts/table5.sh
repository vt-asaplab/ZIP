#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Work in LeNet directory
LENET_DIR="$REPO_ROOT/src/CNN/LeNet"
LINEAR_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear"
TIMES_FILE="$LINEAR_DIR/proof_times.txt" 
rm -f "$TIMES_FILE"
DEST="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear/lenet_output_add_mult"

pushd "$LENET_DIR" >/dev/null

echo
echo "Performance of ZIP on LeNet-5"
echo

echo "=============================================================="
echo "== Proving non-linear layers (activations): GeLU, ELU, SeLU =="
echo "=============================================================="

for act in gelu elu selu; do
  python lenet.py --act "$act"
  python infer_lenet.py --act "$act"

  python act_ops.py --act "$act" --key act1_in
  python act_ops.py --act "$act" --key act2_in
  python act_ops.py --act "$act" --key act3_in
  python act_ops.py --act "$act" --key act4_in

  echo "Proving 1st activation, $act"
  "$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_lenet_act1 "$act"

  echo "Proving 2nd activation, $act"
  "$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_lenet_act2 "$act"

  echo "Proving 3rd activation, $act"
  "$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_lenet_act3 "$act"

  echo "Proving 4th activation, $act"
  "$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_lenet_act4 "$act"
done

echo ""
echo "==========================="
echo "== Proving linear layers =="
echo "==========================="

# Linear layers
rm -rf lenet_output
python conv1_ops.py
python pool1_ops.py
python conv2_ops.py
python pool2_ops.py
python fc1_ops.py
python fc2_ops.py
python fc3_ops.py

CHUNK=60000

rm -rf "$DEST"
mkdir -p "$DEST/addition" "$DEST/multiplication"

pushd "$LENET_DIR/lenet_output" >/dev/null

split -d -a 5 -l "$CHUNK" --additional-suffix=.txt addition.txt       "$DEST/addition/addition_"
split -d -a 5 -l "$CHUNK" --additional-suffix=.txt multiplication.txt "$DEST/multiplication/multiplication_"

popd >/dev/null

pushd "$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear" >/dev/null

for i in {1..6}; do
  python generate_config.py --num-add 60000 --num-mul 60000 --size 18
  go run main.go config.go
done

python generate_config.py --num-add 60000 --num-mul 58096 --size 18
go run main.go config.go

python generate_config.py --num-add 9342 --num-mul 0 --size 16
go run main.go config.go

popd >/dev/null

echo
echo "*********************************************"
# ---- Totals from proof_times.txt ----
if [[ -s "$TIMES_FILE" ]]; then
  awk -F',' '
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); p += $1 + 0; v += $2 + 0 }
    END {
      printf("total proving time for all linear layers : %.6f sec\n", p + 0);
      printf("total verification time for all linear layers: %.6f sec\n", v + 0);
    }' "$TIMES_FILE"
else
  echo "No proof times found at $TIMES_FILE"
fi

echo "*********************************************"
echo

popd >/dev/null
